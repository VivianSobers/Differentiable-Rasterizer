#!/usr/bin/env python3
"""Train a network to predict triangle scenes from images.

    # single GPU
    python python/train.py --data path/to/images --triangles 128 --epochs 40

    # both GPUs
    torchrun --nproc_per_node=2 python/train.py --data path/to/images

    # no data on hand — verifiable synthetic task
    python python/train.py --synthetic --epochs 5

## Where the time goes

Rendering is the slow part of an end-to-end step. Three levers:

- **`--raster-device`** selects the rasterizer's own backend. `auto` picks from
  the measured crossover: the GPU only pulls ahead once there is enough geometry
  to spread gradient-accumulator contention across, and below that a many-core
  CPU parallelizing over batch items wins outright. This is separate from the
  torch device — the extension carries its own wgpu context.
- **Render loss on a sub-batch.** The photometric term is computed on a slice
  of each batch (`--render-fraction`), while the whole batch still gets the
  cheap GPU-side terms. The gradient is noisier per step but steps are far
  faster, which wins comfortably.
- **`--pretrain`**, which trains against precomputed fits — no rasterizer in the
  loop at all. Use it to get most of the way, then fine-tune end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from diffrast.data import ImageFolderDataset, PrecomputedFitDataset, SyntheticShapeDataset
from diffrast.model import TriangleNet, count_parameters
from diffrast.torch_layer import psnr, rasterize


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, torch.device]:
    """Returns `(rank, world_size, device)`, initializing the process group."""
    if not is_distributed():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def sigma_at(epoch: int, epochs: int, start: float, end: float) -> float:
    """Anneal softness geometrically across training.

    Same reasoning as the standalone fitter: a sharp sigma gives a triangle no
    gradient unless it already overlaps its target region, so training starts
    blurry and tightens.
    """
    if epochs <= 1:
        return end
    t = epoch / (epochs - 1)
    return start * (end / start) ** t


def build_dataset(args):
    if args.pretrain:
        return PrecomputedFitDataset(args.pretrain)
    if args.synthetic:
        return SyntheticShapeDataset(
            length=args.synthetic_size, size=args.size, triangles=args.triangles
        )
    return ImageFolderDataset(args.data, size=args.size, limit=args.limit)


def train_step(
    model, batch, args, sigma: float, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    """One optimization step. Returns `(loss, metrics)`."""
    supervised = isinstance(batch, (list, tuple))
    if supervised:
        images, target_params = batch
        images = images.to(device, non_blocking=True)
        target_params = target_params.to(device, non_blocking=True)
    else:
        images = batch.to(device, non_blocking=True)
        target_params = None

    predicted = model(images)
    metrics: dict[str, float] = {}
    loss = torch.zeros((), device=device)

    if target_params is not None:
        # Parameter supervision is a *warm start*, not the objective. Many
        # different triangle sets render to the same image, so matching the
        # fitter's particular solution is a weaker signal than matching its
        # output — useful early, then decayed away.
        param_loss = F.smooth_l1_loss(predicted, target_params)
        loss = loss + args.param_weight * param_loss
        metrics["param_loss"] = param_loss.item()

    if args.render_weight > 0:
        # Render only a slice: rendering dominates the step even on the GPU
        # path, and a noisier photometric gradient is a good trade for
        # substantially cheaper steps.
        #
        # This trade only holds when there is a second loss term covering the
        # rest of the batch. Without parameter supervision the unrendered
        # images contribute nothing at all — they are loaded, moved to the
        # device and discarded. `main` warns about that configuration.
        n = max(1, int(len(images) * args.render_fraction))
        rendered = rasterize(
            predicted[:n],
            images.shape[-2],
            images.shape[-1],
            sigma=sigma,
            device=args.raster_device,
        )
        render_loss = F.mse_loss(rendered, images[:n])
        loss = loss + args.render_weight * render_loss
        metrics["render_loss"] = render_loss.item()
        with torch.no_grad():
            metrics["psnr"] = psnr(rendered, images[:n]).item()

    metrics["loss"] = loss.item()
    return loss, metrics


def main() -> int:
    args = parse_args()
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        # TF32 is a large free speedup on Ada-generation cards and the
        # precision loss is irrelevant next to the noise in this objective.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataset = build_dataset(args)
    triangles = getattr(dataset, "triangles", args.triangles)

    # Without --pretrain there is no parameter-supervision term, so the render
    # loss is the *only* loss and a fractional render silently throws most of
    # each batch away. Cheap to get wrong, invisible in the loss curve, and it
    # costs exactly the data the model is short of.
    if is_main and args.render_fraction < 1.0 and not args.pretrain:
        used = max(1, int(args.batch * args.render_fraction))
        print(
            f"warning: --render-fraction {args.render_fraction} with no parameter "
            f"supervision means {args.batch - used} of every {args.batch} images "
            f"contribute nothing.\n"
            f"         Pass --render-fraction 1.0 unless you are deliberately "
            f"trading data for speed."
        )

    sampler = DistributedSampler(dataset) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = TriangleNet(triangles=triangles, width=args.width, pool=args.pool).to(device)
    if is_main:
        print(f"model: {count_parameters(model):,} parameters, {triangles} triangles")
        print(f"data:  {len(dataset):,} items, batch {args.batch} x {world_size} rank(s)")
        print(f"device: {device}")

    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(loader) * args.epochs)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.15
    )

    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best = math.inf

    for epoch in range(args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        sigma = sigma_at(epoch, args.epochs, args.sigma_start, args.sigma_end)
        # Decay parameter supervision so the image loss takes over.
        args.param_weight = args.param_weight_initial * (1 - epoch / max(1, args.epochs - 1))

        totals: dict[str, float] = {}
        count = 0
        start = time.perf_counter()

        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = train_step(model, batch, args, sigma, device)

            if not torch.isfinite(loss):
                # Skip rather than abort: one bad batch should not end a run
                # that may have been going for hours.
                if is_main:
                    print("  warning: non-finite loss, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            schedule.step()

            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1

        elapsed = time.perf_counter() - start
        means = {k: v / max(count, 1) for k, v in totals.items()}

        if world_size > 1:
            # Average metrics across ranks so the log reflects the whole run.
            packed = torch.tensor(list(means.values()), device=device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            means = dict(zip(means.keys(), (packed / world_size).tolist()))

        if is_main:
            record = {"epoch": epoch, "sigma": sigma, "seconds": elapsed, **means}
            history.append(record)
            summary = "  ".join(f"{k} {v:.5f}" for k, v in means.items())
            print(f"epoch {epoch:>3}  sigma {sigma:.4f}  {summary}  ({elapsed:.1f}s)")

            score = means.get("render_loss", means.get("loss", math.inf))
            state = model.module.state_dict() if world_size > 1 else model.state_dict()
            checkpoint = {
                "model": state,
                "triangles": triangles,
                "width": args.width,
                "pool": args.pool,
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(checkpoint, out_dir / "last.pt")
            if score < best:
                best = score
                torch.save(checkpoint, out_dir / "best.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print(f"\nbest score {best:.6f}\ncheckpoints in {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="directory of training images")
    source.add_argument("--synthetic", action="store_true", help="generated triangle scenes")
    source.add_argument("--pretrain", help="precomputed fit dataset (.pt) for supervised warm start")

    p.add_argument("--triangles", type=int, default=128)
    p.add_argument("--size", type=int, default=64, help="training resolution")
    p.add_argument("--width", type=int, default=32, help="model channel width")
    p.add_argument(
        "--pool",
        type=int,
        default=4,
        help="adaptive pool side before the head; 1 discards spatial layout",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="cap dataset size")
    p.add_argument("--synthetic-size", type=int, default=20_000)
    p.add_argument("--out", default="runs/amortized")

    p.add_argument("--sigma-start", type=float, default=0.02)
    p.add_argument("--sigma-end", type=float, default=0.003)
    p.add_argument("--render-weight", type=float, default=1.0)
    p.add_argument(
        "--render-fraction",
        type=float,
        default=None,
        help="share of each batch that gets the render loss "
        "(default: 0.25 with --pretrain, 1.0 otherwise)",
    )
    p.add_argument("--param-weight", type=float, default=1.0)
    p.add_argument(
        "--raster-device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="backend for the rasterizer itself, independent of the torch device",
    )

    args = p.parse_args()

    # The default depends on whether a second loss term exists. With
    # --pretrain, parameter supervision covers the unrendered part of the batch
    # and rendering a slice is a good trade. Without it the render loss is the
    # only loss, so a fractional render just discards data — which is a
    # mistake this defaulted everyone into, and which cost a real experiment
    # here before it was noticed.
    if args.render_fraction is None:
        args.render_fraction = 0.25 if args.pretrain else 1.0
    if not 0 < args.render_fraction <= 1:
        p.error("--render-fraction must be in (0, 1]")
    args.param_weight_initial = args.param_weight
    return args


if __name__ == "__main__":
    raise SystemExit(main())
