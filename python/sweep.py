#!/usr/bin/env python3
"""Fit the same target at several triangle counts and compare.

Answers the question the project invites: how much does more geometry buy you?

    python python/sweep.py --counts 32 128 512 --iters 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffrast import FitError, run_fit, sweep_curves  # noqa: E402
from diffrast.plots import SERIES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="image to fit (default: synthetic)")
    parser.add_argument("--counts", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="out/sweep")
    args = parser.parse_args()

    if len(args.counts) > len(SERIES):
        print(
            f"error: at most {len(SERIES)} counts can share one chart; "
            "run separate sweeps instead",
            file=sys.stderr,
        )
        return 1

    out_root = Path(args.out)
    runs: dict[str, list[float]] = {}
    summary: list[tuple[int, float, float]] = []

    for count in args.counts:
        print(f"fitting {count} triangles...", flush=True)
        try:
            result = run_fit(
                args.target,
                out_dir=out_root / f"tris-{count}",
                triangles=count,
                iters=args.iters,
                size=args.size,
                seed=args.seed,
            )
        except FitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        runs[f"{count} tris"] = result.losses
        summary.append((count, result.best_loss, result.duration_s))
        print(f"  best loss {result.best_loss:.6f} in {result.duration_s:.1f}s")

    chart = sweep_curves(runs, out_root / "sweep.png")

    print("\ntriangles   best loss    time")
    for count, loss, secs in summary:
        print(f"{count:>9}   {loss:.6f}   {secs:>5.1f}s")
    print(f"\nwrote {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
