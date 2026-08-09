#!/usr/bin/env python3
"""How well does a trained model transfer off its training distribution?

    python python/transfer.py --checkpoint runs/sweep/data-160k/best.pt

Two claims in this repo have never been tested, and both can be checked with
evaluation alone — no training, minutes rather than hours.

**Resolution.** `TriangleNet` pools adaptively specifically so that a model
"can be trained at 64px and run at 256px without reshaping the head". That is
an argument about shapes fitting together, which is not the same as the model
still working. Nothing has ever measured the second part.

**Scene complexity.** Every training scene has exactly N triangles and the model
always emits exactly N. Whether a model trained on 64-triangle scenes handles a
16-triangle or a 256-triangle one is entirely unknown, and it is the difference
between a scene predictor and a lookup table for one particular corner of the
distribution.

The number reported is **margin over the mean-colour baseline**, in dB. Absolute
PSNR cannot be compared across these cells — a 16-triangle scene is intrinsically
easier to approximate than a 256-triangle one, and both the model and the
baseline get easier with it. The margin is measured against a baseline computed
on the same images, so it survives the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from diffrast.data import SyntheticShapeDataset
from evaluate import evaluate, load_model


def grid(model, sizes: list[int], triangle_counts: list[int], args) -> dict:
    results: dict[str, dict[str, float]] = {}

    for size in sizes:
        for triangles in triangle_counts:
            dataset = SyntheticShapeDataset(
                length=args.count, size=size, triangles=triangles, seed=args.seed + 1
            )
            images = torch.stack([dataset[i] for i in range(args.count)])
            metrics = evaluate(model, images, args)
            results[f"{size}px/{triangles}tri"] = {
                "size": size,
                "eval_triangles": triangles,
                "margin_db": metrics["margin_db"],
                "one_shot_psnr": metrics["one_shot_psnr"],
                "input_gain_db": metrics["input_gain_db"],
                "mirror_response": metrics["mirror_response"],
            }
            print(
                f"  {size:>4}px {triangles:>4} tri   "
                f"margin {metrics['margin_db']:+6.2f} dB   "
                f"gain {metrics['input_gain_db']:5.2f}   "
                f"mirror {metrics['mirror_response']:.2f}"
            )
    return results


def report(results: dict, sizes: list[int], triangle_counts: list[int], trained: dict) -> None:
    print(f"\nmargin over the mean-colour baseline, dB "
          f"(trained at {trained['size']}px on {trained['triangles']}-triangle scenes)\n")
    print(f"{'':>8}" + "".join(f"{t:>10}" for t in triangle_counts) + "   <- eval triangles")
    for size in sizes:
        cells = []
        for triangles in triangle_counts:
            value = results[f"{size}px/{triangles}tri"]["margin_db"]
            marker = "*" if size == trained["size"] and triangles == trained["triangles"] else " "
            cells.append(f"{value:>9.2f}{marker}")
        print(f"{size:>6}px" + "".join(cells))
    print("\n* = the configuration the model was trained on")
    print("positive = beats a flat per-image colour fill; negative = worse than useless")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    model, checkpoint = load_model(args.checkpoint)
    trained = {
        "size": checkpoint.get("args", {}).get("size", 96),
        "triangles": checkpoint["triangles"],
    }
    print(f"model: {model.triangles} triangles, pool {model.pool_size}, "
          f"trained at {trained['size']}px")

    results = grid(model, args.sizes, args.eval_triangles, args)
    report(results, args.sizes, args.eval_triangles, trained)

    if args.out:
        Path(args.out).write_text(json.dumps({"trained": trained, "grid": results}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[48, 96, 144, 192])
    p.add_argument("--eval-triangles", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--count", type=int, default=96)
    p.add_argument("--sigma", type=float, default=0.003)
    p.add_argument("--refine-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the grid as JSON")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
