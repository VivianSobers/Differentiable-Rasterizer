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
import multiprocessing as mp
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
from diffrast.torch_layer import fused_mse


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


def save_atomically(payload: dict, path: Path) -> None:
    """Write a checkpoint so that `path` is either the old file or the new one.

    `torch.save` straight to the destination leaves a truncated file if the
    process dies mid-write, and the resume path cannot tell that from a valid
    one until it fails to load — potentially hours later.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def build_dataset(args):
    if args.pretrain:
        return PrecomputedFitDataset(args.pretrain)
    if args.synthetic:
        return SyntheticShapeDataset(
            length=args.synthetic_size, size=args.size, triangles=args.triangles
        )
    return ImageFolderDataset(args.data, size=args.size, limit=args.limit)


@torch.no_grad()
def validation_score(model, images: torch.Tensor, sigma: float, args, device) -> float:
    """Photometric loss on a fixed held-out set, at one fixed resolution.

    Model selection compares one epoch's loss against another's, which silently
    assumes they are on the same scale. With `--sizes` they are not: each epoch
    trains at a different resolution and MSE is not comparable across them, so
    `best.pt` ends up chosen by whichever resolution happened to score lowest
    rather than by which model is better.

    This is also a genuinely held-out set, unlike the training loss it
    replaces — a distinct seed, never trained on.
    """
    was_training = model.training
    model.eval()
    images = images.to(device, non_blocking=True)
    score = fused_mse(model(images), images, sigma=sigma, device=args.raster_device).item()
    model.train(was_training)
    return score


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
        # `fused_mse` rather than `F.mse_loss(rasterize(...), ...)`: identical
        # value and gradient, one render per step instead of two. The obvious
        # form renders once for the forward and again inside the backward,
        # which rebuilds the tape it did not keep — and rendering is most of
        # the step. Pinned by `test_fused_mse_matches_rasterize_then_mse`.
        render_loss = fused_mse(
            predicted[:n], images[:n], sigma=sigma, device=args.raster_device
        )
        loss = loss + args.render_weight * render_loss
        metrics["render_loss"] = render_loss.item()
        # PSNR is a pure function of the MSE at a fixed peak, so it comes out
        # of the loss rather than costing another render.
        mse = metrics["render_loss"]
        metrics["psnr"] = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)

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

    # Shared across the fork so persistent workers can pick up a per-epoch
    # resolution change without being re-forked — see the note on
    # `persistent_workers` below for why re-forking is the thing to avoid.
    size_value = None
    if args.sizes:
        size_value = mp.Value("i", args.sizes[0])
        dataset.size_value = size_value

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
        # Workers are forked once and never again; a resolution change under
        # `--sizes` reaches them through `size_value`, a `multiprocessing.Value`
        # in memory shared across the fork, rather than through re-forking.
        persistent_workers=args.workers > 0,
    )
    if args.workers > 0:
        # Fork the workers now, before anything below touches the
        # rasterizer's GPU/wgpu context. `SyntheticShapeDataset.__getitem__`
        # forks safely today because the parent hasn't used wgpu yet; the
        # validation build below and every training step do use it, and
        # forking *after* that deadlocks the child on a lock the parent's
        # (now-absent) other thread was holding. Reproduced reliably with
        # `--sizes` (which used to re-fork every epoch, guaranteeing a fork
        # after wgpu use) — persistent workers plus this one early fork
        # avoid it entirely.
        next(iter(loader))

    model = TriangleNet(triangles=triangles, width=args.width, pool=args.pool).to(device)
    if args.init_from:
        # A warm start into a *new* training phase — different data, often a
        # different epoch count — not a resumed run of the same one. `--resume`
        # cannot do this: it restores the optimizer and OneCycle schedule too,
        # both defined over the checkpoint's own step count, and a schedule
        # already at the end of its cycle has nothing left to anneal. This
        # loads only the weights and leaves the epoch counter, optimizer and
        # schedule to start fresh for the new run.
        blob = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        if is_main:
            print(f"initialized weights from {args.init_from}")
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

    # Only built when resolutions vary, because that is the case where the
    # training loss stops being a usable selection signal.
    validation = None
    if args.sizes:
        val_set = SyntheticShapeDataset(
            length=args.val_size, size=args.size, triangles=triangles, seed=9973
        )
        validation = torch.stack([val_set[i] for i in range(args.val_size)])
        if is_main:
            print(f"validation: {args.val_size} held-out scenes at {args.size}px")

    history: list[dict] = []
    best = math.inf
    first_epoch = 0

    # A twelve-hour run that dies at hour nine and restarts from zero has
    # wasted nine hours. Resuming restores the optimizer and scheduler as well
    # as the weights: Adam's moments and a OneCycle schedule's position are
    # part of the training state, and reloading only the weights silently
    # restarts the learning-rate cycle from its warmup.
    if args.resume:
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        target = model.module if world_size > 1 else model
        target.load_state_dict(blob["model"])
        if "optimizer" in blob:
            # OneCycle is defined over a *fixed* horizon — it anneals to a
            # floor at exactly `total_steps`. Resuming into a run with a
            # different `--epochs` is therefore not the same schedule, and
            # silently continuing steps past the old total raises deep inside
            # the scheduler with nothing pointing back to here.
            saved_total = blob["schedule"].get("total_steps")
            if saved_total is not None and saved_total != steps:
                raise SystemExit(
                    f"cannot resume: this checkpoint was trained on a schedule of "
                    f"{saved_total} steps, and --epochs {args.epochs} gives {steps}.\n"
                    f"Resume with the same --epochs and --batch as the original run "
                    f"(it was --epochs {blob.get('args', {}).get('epochs', '?')}), or "
                    f"start a fresh run to train for longer."
                )
            opt.load_state_dict(blob["optimizer"])
            schedule.load_state_dict(blob["schedule"])
            first_epoch = blob.get("epoch", -1) + 1
            best = blob.get("best", math.inf)
            history = blob.get("history", [])
        elif is_main:
            print("warning: checkpoint has no optimizer state, restarting the schedule")
        if is_main:
            print(f"resumed from {args.resume} at epoch {first_epoch}, best {best:.6f}")

    if first_epoch >= args.epochs and is_main:
        print(f"nothing to do: checkpoint is already at epoch {first_epoch} of {args.epochs}")

    for epoch in range(first_epoch, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Resolution jitter. A model trained at one resolution transfers badly
        # to another — measured, not assumed: a 128px-trained model went from
        # +4.44 dB of margin on its own resolution to -1.17 dB at 96px. The
        # adaptive pool makes varying resolution *possible*; only training
        # across resolutions makes the model actually invariant to it.
        if args.sizes:
            chosen = args.sizes[torch.randint(len(args.sizes), (1,)).item()]
            dataset.size = chosen
            size_value.value = chosen
            if is_main:
                print(f"  epoch {epoch}: training at {chosen}px")

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

            score = (
                validation_score(model, validation, sigma, args, device)
                if validation is not None
                else means.get("render_loss", means.get("loss", math.inf))
            )
            if validation is not None:
                record["val_loss"] = score
            improved = score < best
            if improved:
                best = score

            state = model.module.state_dict() if world_size > 1 else model.state_dict()
            checkpoint = {
                "model": state,
                "triangles": triangles,
                "width": args.width,
                "pool": args.pool,
                "epoch": epoch,
                "args": vars(args),
                # Enough to resume rather than merely to evaluate. `best` is
                # recorded after this epoch's comparison, so a resumed run does
                # not overwrite a better checkpoint with a worse one.
                "optimizer": opt.state_dict(),
                "schedule": schedule.state_dict(),
                "best": best,
                "history": history,
            }
            # Written to a temporary name and renamed, because a 45 MB save is
            # a wide enough window to be interrupted in — by a crash, or by
            # something as ordinary as `pkill -9` cleaning up a previous job.
            # A partial file left at either of these paths is worse than no
            # file: `--resume` and `evaluate.py` both look here, and an
            # unloadable checkpoint fails much later than a missing one.
            save_atomically(checkpoint, out_dir / "last.pt")
            if improved:
                save_atomically(checkpoint, out_dir / "best.pt")
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
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="train on a different resolution each epoch, chosen from this list; "
        "the model pools adaptively but nothing has ever made it *use* that",
    )
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
    p.add_argument(
        "--val-size",
        type=int,
        default=256,
        help="held-out scenes used for model selection when --sizes varies the "
        "training resolution",
    )
    p.add_argument("--out", default="runs/amortized")
    p.add_argument("--resume", help="continue from a checkpoint written by a previous run")
    p.add_argument(
        "--init-from",
        help="load model weights from a checkpoint as a warm start for a new "
        "training phase, e.g. --pretrain followed by end-to-end fine-tuning; "
        "unlike --resume this does not restore the optimizer, schedule or "
        "epoch counter",
    )

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
    if args.sizes:
        if args.pretrain:
            p.error("--sizes cannot be used with --pretrain: those images are already rendered")
        if any(s <= 0 for s in args.sizes):
            p.error("--sizes must all be positive")
    if args.render_fraction is None:
        args.render_fraction = 0.25 if args.pretrain else 1.0
    if not 0 < args.render_fraction <= 1:
        p.error("--render-fraction must be in (0, 1]")
    if args.resume and args.init_from:
        p.error("--resume and --init-from are mutually exclusive")
    args.param_weight_initial = args.param_weight
    return args


if __name__ == "__main__":
    raise SystemExit(main())
