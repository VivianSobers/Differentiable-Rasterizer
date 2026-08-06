#!/usr/bin/env python3
"""Fit a corpus of images and save `(image, params)` pairs for supervised training.

    python python/precompute.py --data photos/ --out data/fits.pt --tris 128

Fitting is CPU-bound and embarrassingly parallel; training on the results is
GPU-bound. Doing the fits once, up front, means the GPUs are never waiting on a
rasterizer — and the same fits are reused across every epoch and every
hyperparameter sweep instead of being recomputed.

The fits produced here are targets for a *warm start*, not ground truth. Many
different triangle sets render to the same image, so a network that matches the
rendered output while diverging from these parameters is doing fine.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from diffrast import FitError, run_fit
from diffrast.data import IMAGE_SUFFIXES


def fit_one(job: tuple[str, str, int, int, int, int]) -> tuple[str, list] | None:
    """Fit a single image in a worker process. Returns `(path, params)`."""
    path, out_root, triangles, iters, size, seed = job
    work_dir = Path(out_root) / f"w{abs(hash(path)) % 100000:05d}"
    try:
        result = run_fit(
            path,
            out_dir=work_dir,
            triangles=triangles,
            iters=iters,
            size=size,
            seed=seed,
            export=64,
        )
    except FitError as exc:
        print(f"  skipped {Path(path).name}: {exc}", file=sys.stderr)
        return None

    scene = result.scene
    if len(scene) != triangles:
        print(f"  skipped {Path(path).name}: got {len(scene)} triangles", file=sys.stderr)
        return None

    rows = []
    for tri in scene.triangles:
        rows.append(
            [
                tri.verts[0][0], tri.verts[0][1],
                tri.verts[1][0], tri.verts[1][1],
                tri.verts[2][0], tri.verts[2][1],
                tri.color[0], tri.color[1], tri.color[2],
                tri.alpha,
            ]
        )
    return path, rows


def load_image(path: str, size: int) -> torch.Tensor:
    from PIL import Image

    from diffrast.data import srgb_to_linear

    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.LANCZOS)
        arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    srgb = arr.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    return srgb_to_linear(srgb)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="directory of source images")
    p.add_argument("--out", default="data/fits.pt")
    p.add_argument("--tris", type=int, default=128)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--fit-size", type=int, default=96, help="resolution the fits run at")
    p.add_argument("--size", type=int, default=64, help="resolution images are stored at")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=0, help="0 = one per core")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        print(f"error: no such directory: {root}", file=sys.stderr)
        return 1

    paths = sorted(str(q) for q in root.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"error: no images found under {root}", file=sys.stderr)
        return 1

    import os
    import tempfile

    workers = args.workers or os.cpu_count() or 4
    print(f"fitting {len(paths)} images with {workers} workers ({args.tris} triangles, {args.iters} iters)")

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        jobs = [
            (path, scratch, args.tris, args.iters, args.fit_size, args.seed)
            for path in paths
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [r for r in pool.map(fit_one, jobs) if r is not None]

    if not results:
        print("error: every fit failed", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - start
    print(f"fitted {len(results)}/{len(paths)} in {elapsed:.1f}s ({elapsed/len(results):.2f}s each)")

    images = torch.stack([load_image(path, args.size) for path, _ in results])
    params = torch.tensor([rows for _, rows in results], dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": images,
            "params": params,
            "meta": {
                "triangles": args.tris,
                "iters": args.iters,
                "fit_size": args.fit_size,
                "size": args.size,
                "count": len(results),
            },
        },
        out_path,
    )
    print(f"wrote {out_path}  images {tuple(images.shape)}  params {tuple(params.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
