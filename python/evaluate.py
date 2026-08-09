#!/usr/bin/env python3
"""Measure whether an amortized model actually learned the mapping.

    python python/evaluate.py --checkpoint runs/amortized/best.pt --synthetic

A falling training loss is not evidence that this works. The model emits a
scene, the scene is rendered, and the render is scored against the target — and
a model that ignored its input entirely and emitted one generic scene would
still drive that loss down, because most images in a set look somewhat alike.
Reporting PSNR alone cannot tell the two apart.

So this reports PSNR against three controls, each of which answers a question
the headline number cannot:

- **Shuffled control.** Score each predicted render against a *different*
  target. Whatever score survives is what the model gets for free without
  looking at its input; the gap between the two is the part attributable to
  actually reading the image. A near-zero gap means the model is a very
  expensive way to store a constant.

- **Mean-colour baseline.** Fill each image with its own average colour. This
  is the cheapest possible "prediction" that still adapts to the input. A model
  below this line is worse than useless, however good its PSNR looks in
  isolation.

- **Refinement.** The claim in the README is not that one-shot output is good,
  it is that it is a *better starting point* than random — that the fitter
  reaches a given quality in fewer iterations from a prediction. So run the
  same optimizer for the same budget from both, and compare.

None of these are expensive. Their absence is what let a model that loses to a
flat colour fill look like a success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from diffrast.data import ImageFolderDataset, SyntheticShapeDataset
from diffrast.model import TriangleNet
from diffrast.torch_layer import clamp_params, psnr, random_params, rasterize


def load_model(path: str | Path) -> tuple[TriangleNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model = TriangleNet(
        triangles=checkpoint["triangles"],
        width=checkpoint.get("width", args.get("width", 32)),
        pool=checkpoint.get("pool", args.get("pool", 1)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def refine(
    params: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    sigma: float,
    lr: float = 0.02,
) -> tuple[torch.Tensor, list[float]]:
    """Run the fitter from `params` and report PSNR after every step."""
    p = params.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    trace: list[float] = []

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        rendered = rasterize(p, targets.shape[-2], targets.shape[-1], sigma=sigma)
        loss = F.mse_loss(rendered, targets)
        loss.backward()
        opt.step()
        with torch.no_grad():
            p.copy_(clamp_params(p))
            trace.append(psnr(rendered, targets).item())

    return p.detach(), trace


def evaluate(model: TriangleNet, images: torch.Tensor, args) -> dict:
    size = images.shape[-1]
    out: dict[str, float | list[float]] = {}

    with torch.no_grad():
        predicted = model(images)
        rendered = rasterize(predicted, size, size, sigma=args.sigma)

        out["one_shot_psnr"] = psnr(rendered, images).item()

        # Rolling by one pairs every render with someone else's target. Rolling
        # rather than a random permutation keeps it deterministic and
        # guarantees no image is paired with itself.
        out["shuffled_psnr"] = psnr(rendered, images.roll(1, 0)).item()
        out["input_gain_db"] = out["one_shot_psnr"] - out["shuffled_psnr"]

        mean_colour = images.mean(dim=(2, 3), keepdim=True).expand_as(images)
        out["mean_colour_psnr"] = psnr(mean_colour, images).item()
        # Absolute PSNR depends on how hard the eval set is; the margin over a
        # baseline measured on the *same* set does not, which makes this the
        # number to compare between configurations.
        out["margin_db"] = out["one_shot_psnr"] - out["mean_colour_psnr"]

        # A mirrored image is a different scene — every triangle has to move.
        # A model reading spatial layout should respond about as strongly as it
        # does to an unrelated image; one that pooled the layout away will not.
        mirrored = model(torch.flip(images, dims=[3]))
        moved_by_mirror = (predicted - mirrored).abs().mean()
        moved_by_other = (predicted - predicted.roll(1, 0)).abs().mean()
        out["mirror_response"] = (moved_by_mirror / moved_by_other.clamp_min(1e-12)).item()

    if args.refine_steps > 0:
        _, from_model = refine(predicted, images, args.refine_steps, args.sigma)
        scratch = random_params(
            len(images), model.triangles, generator=torch.Generator().manual_seed(0)
        )
        _, from_random = refine(scratch, images, args.refine_steps, args.sigma)

        out["refined_from_model_psnr"] = from_model[-1]
        out["refined_from_random_psnr"] = from_random[-1]
        out["refine_trace_model"] = from_model
        out["refine_trace_random"] = from_random

        # The amortization claim, as a number: how many steps a random start
        # needs to reach what the model produced with none.
        target = out["one_shot_psnr"]
        reached = next((i + 1 for i, v in enumerate(from_random) if v >= target), None)
        out["steps_for_random_to_match_model"] = reached

    return out


def report(out: dict) -> None:
    print(f"\none-shot PSNR         {out['one_shot_psnr']:8.2f} dB")
    print(f"  vs shuffled targets {out['shuffled_psnr']:8.2f} dB")
    print(f"  input gain          {out['input_gain_db']:8.2f} dB  <- how much reading the image is worth")
    print(f"mean-colour baseline  {out['mean_colour_psnr']:8.2f} dB  <- must be beaten to be useful")
    print(f"  margin over it      {out['margin_db']:8.2f} dB  <- comparable across eval sets")
    print(f"mirror response       {out['mirror_response']:8.2f}     <- 1.0 = fully spatially aware, 0 = blind")

    verdict = []
    if out["input_gain_db"] < 1.0:
        verdict.append("model barely uses its input")
    if out["one_shot_psnr"] < out["mean_colour_psnr"]:
        verdict.append("loses to a flat colour fill")
    if out["mirror_response"] < 0.5:
        verdict.append("largely spatially blind")

    if "refined_from_model_psnr" in out:
        steps = out["steps_for_random_to_match_model"]
        print(f"\nafter {len(out['refine_trace_model'])} refinement steps")
        print(f"  from prediction     {out['refined_from_model_psnr']:8.2f} dB")
        print(f"  from random         {out['refined_from_random_psnr']:8.2f} dB")
        if steps is None:
            print("  a random start never caught the one-shot prediction in this budget")
        else:
            print(f"  random needed {steps} steps to match the one-shot prediction")
        if out["refined_from_model_psnr"] <= out["refined_from_random_psnr"]:
            verdict.append("prediction is not a better starting point than random")

    print("\nverdict: " + ("; ".join(verdict) if verdict else "learned a real mapping"))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    model, checkpoint = load_model(args.checkpoint)
    size = args.size or checkpoint.get("args", {}).get("size", 64)

    if args.data:
        dataset = ImageFolderDataset(args.data, size=size, limit=args.count)
    else:
        # A different seed from training's default, so this is held out rather
        # than a re-run of what the model was trained on.
        # The eval set's complexity is a property of the *benchmark*, not of
        # the model being benchmarked. Defaulting it to `model.triangles` gave
        # every model its own private exam and made the resulting PSNRs
        # incomparable across configurations — a 128-triangle model was scored
        # against harder targets than a 64-triangle one, and the mean-colour
        # baseline moved with it. Pass --eval-triangles to fix the exam.
        eval_triangles = args.eval_triangles or model.triangles
        dataset = SyntheticShapeDataset(
            length=args.count, size=size, triangles=eval_triangles, seed=args.seed + 1
        )

    images = torch.stack([dataset[i] for i in range(min(args.count, len(dataset)))])
    print(f"model: {model.triangles} triangles, pool {model.pool_size}")
    print(f"eval:  {len(images)} images at {size}px"
          + (f", {args.eval_triangles} triangles" if args.eval_triangles else ""))

    out = evaluate(model, images, args)
    report(out)

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="a .pt written by train.py")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data", help="directory of evaluation images")
    source.add_argument("--synthetic", action="store_true", help="held-out synthetic scenes")

    p.add_argument("--count", type=int, default=64, help="images to evaluate")
    p.add_argument("--size", type=int, default=None, help="defaults to the training size")
    p.add_argument(
        "--eval-triangles",
        type=int,
        default=None,
        help="triangles in the held-out scenes; fix it to compare models with "
        "different triangle budgets (default: the model's own count)",
    )
    p.add_argument("--sigma", type=float, default=0.003)
    p.add_argument("--refine-steps", type=int, default=0, help="0 skips the refinement study")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the metrics as JSON")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
