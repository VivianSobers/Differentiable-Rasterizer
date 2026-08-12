#!/usr/bin/env python3
"""Render the showcase figure: what the trained model actually produces.

    python python/figure.py --checkpoint runs/photos/best.pt --data photos/ \
        --count 6 --refine-steps 100 --out docs/amortized.png

Every result about the amortized model in this repository is a table. Tables are
the right way to *check* a claim and the wrong way to *see* one, and the whole
ML half of the project had no picture at all.

Three columns, which is exactly the argument the project makes:

    target | one forward pass | + N refinement steps

The middle column is the amortization claim — a scene predicted with no
optimization at all. The right column is what the fitter does starting from it.
Read them together: the middle is what the network learned, the gap to the
right is what it still leaves on the table.

Images are converted from linear light to sRGB before saving. The rasterizer
works in linear light throughout, and writing those values straight to a PNG
would show a picture noticeably darker than what the loss actually sees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from diffrast.data import ImageFolderDataset, SyntheticShapeDataset, linear_to_srgb
from diffrast.torch_layer import psnr, rasterize
from evaluate import load_model, refine

#: Matches the charts in `plots.py` so the figures read as one set.
SURFACE = (18, 18, 22)
INK = (232, 232, 236)
GAP = 10
LABEL_HEIGHT = 26


def to_pil(image: torch.Tensor):
    """A `(3, H, W)` linear-light tensor as a displayable RGB image."""
    from PIL import Image

    srgb = linear_to_srgb(image.clamp(0, 1)).permute(1, 2, 0)
    return Image.fromarray((srgb.numpy() * 255).round().astype("uint8"), mode="RGB")


def build_grid(columns: list[list], labels: list[str], scale: int):
    """Lay the panels out as a labelled grid, one row per sample."""
    from PIL import Image, ImageDraw

    rows = len(columns[0])
    cell = columns[0][0].width * scale
    width = len(columns) * cell + (len(columns) - 1) * GAP
    height = LABEL_HEIGHT + rows * cell + (rows - 1) * GAP

    sheet = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(sheet)

    for col, (panels, label) in enumerate(zip(columns, labels)):
        x = col * (cell + GAP)
        draw.text((x + 2, 6), label, fill=INK)
        for row, panel in enumerate(panels):
            if scale != 1:
                panel = panel.resize((cell, cell), Image.NEAREST)
            sheet.paste(panel, (x, LABEL_HEIGHT + row * (cell + GAP)))

    return sheet


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    model, checkpoint = load_model(args.checkpoint)
    size = args.size or checkpoint.get("args", {}).get("size", 96)

    if args.data:
        dataset = ImageFolderDataset(args.data, size=size, limit=args.count * 4)
    else:
        dataset = SyntheticShapeDataset(
            length=args.count * 4, size=size, triangles=model.triangles, seed=args.seed + 1
        )

    # Spread the picks across the set rather than taking the first N, which on
    # a sorted image folder means one directory and one kind of subject.
    stride = max(1, len(dataset) // args.count)
    images = torch.stack([dataset[i * stride] for i in range(args.count)])

    with torch.no_grad():
        predicted = model(images)
        one_shot = rasterize(predicted, size, size, sigma=args.sigma)

    refined_params, trace = refine(predicted, images, args.refine_steps, args.sigma)
    with torch.no_grad():
        refined = rasterize(refined_params, size, size, sigma=args.sigma)

    print(f"model:    {model.triangles} triangles, pool {model.pool_size}, {size}px")
    print(f"one-shot: {psnr(one_shot, images).item():.2f} dB")
    print(f"refined:  {psnr(refined, images).item():.2f} dB "
          f"after {args.refine_steps} steps ({trace[0]:.2f} -> {trace[-1]:.2f})")

    columns = [
        [to_pil(img) for img in images],
        [to_pil(img) for img in one_shot],
        [to_pil(img) for img in refined],
    ]
    labels = ["target", "one forward pass", f"+ {args.refine_steps} refine steps"]

    sheet = build_grid(columns, labels, args.scale)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out} ({sheet.width}x{sheet.height})")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data", help="directory of images")
    source.add_argument("--synthetic", action="store_true")
    p.add_argument("--count", type=int, default=6, help="samples down the page")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--sigma", type=float, default=0.003)
    p.add_argument("--refine-steps", type=int, default=100)
    p.add_argument("--scale", type=int, default=2, help="upscale factor for the panels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="docs/amortized.png")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
