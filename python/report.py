#!/usr/bin/env python3
"""Run a fit and produce every artifact for it: images, loss curve, GIF.

    python python/report.py                      # synthetic target
    python python/report.py photo.jpg --tris 300 --iters 2500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffrast import (  # noqa: E402
    FitError,
    comparison_strip,
    loss_curve,
    make_gif,
    run_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="image to fit (default: synthetic)")
    parser.add_argument("--tris", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument("--frames", type=int, default=15, help="save a frame every N iters")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    print(f"running fit: {args.tris} triangles, {args.iters} iters, size {args.size}")
    try:
        result = run_fit(
            args.target,
            out_dir=args.out,
            triangles=args.tris,
            iters=args.iters,
            size=args.size,
            seed=args.seed,
            save_every=0 if args.no_gif else args.frames,
        )
    except FitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"  {result.duration_s:.1f}s  "
        f"loss {result.initial_loss:.6f} -> {result.best_loss:.6f} "
        f"({result.improvement:.0f}x)  "
        f"{len(result.scene)} triangles, mean alpha {result.scene.mean_alpha:.2f}"
    )

    out = result.out_dir
    written = [loss_curve(result.losses, out / "loss.png")]
    written.append(comparison_strip(result.target_png, result.fit_png, out / "comparison.png"))

    if not args.no_gif:
        try:
            written.append(make_gif(result.frames_dir, out / "convergence.gif"))
        except FileNotFoundError as exc:
            # A missing GIF should not sink a report whose other artifacts are
            # perfectly good.
            print(f"warning: skipping GIF ({exc})", file=sys.stderr)

    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
