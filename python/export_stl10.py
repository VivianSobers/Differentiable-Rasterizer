#!/usr/bin/env python3
"""Export STL-10's unlabeled split to a folder of PNGs.

    python python/export_stl10.py --out data/photos --count 40000

Every claim this project has made about the amortized model was measured on
synthetic triangle scenes rendered by the same rasterizer it is trained
against — inverting a renderer, not approximating a photograph.
`ImageFolderDataset` and the `--pretrain` path exist to close that gap but
have never been exercised. STL-10's unlabeled split is 100k images at 96x96,
which matches the training resolution used elsewhere in this repo with no
resize needed, and downloads through torchvision without any manual corpus
collection. Exporting to a plain folder of PNGs means the existing, tested
`ImageFolderDataset` path is used completely unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/photos", help="output directory for PNGs")
    p.add_argument("--count", type=int, default=40_000, help="how many images to export")
    p.add_argument("--root", default="data/stl10", help="torchvision download cache")
    args = p.parse_args()

    from torchvision.datasets import STL10

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading/loading STL-10 unlabeled split into {args.root} ...")
    dataset = STL10(root=args.root, split="unlabeled", download=True)

    count = min(args.count, len(dataset))
    print(f"exporting {count}/{len(dataset)} images to {out_dir} ...")
    for i in range(count):
        img, _ = dataset[i]
        img.save(out_dir / f"{i:06d}.png")
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{count}")

    print(f"wrote {count} PNGs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
