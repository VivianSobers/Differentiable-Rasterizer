"""Datasets for training the amortized scene predictor.

Two sources, both useful:

- `ImageFolderDataset` — real photos from a directory tree.
- `SyntheticShapeDataset` — procedurally generated triangle scenes, rendered by
  the same rasterizer being trained against.

The synthetic set exists because it makes the task *verifiable*. Its images are
exactly representable by N triangles, so a model that cannot drive the loss near
zero on it has a bug rather than a hard problem. Real photos never offer that
guarantee, which makes them a poor first signal.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def srgb_to_linear(x: Tensor) -> Tensor:
    """Decode 0-1 sRGB values to linear light, matching the Rust rasterizer.

    Training in the wrong color space is a silent accuracy tax: the loss would
    weight highlights and shadows differently than the renderer does, and every
    prediction would carry a systematic bias.
    """
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: Tensor) -> Tensor:
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


class ImageFolderDataset(Dataset):
    """Every image under `root`, resized to a square and returned in linear light."""

    def __init__(self, root: str | Path, size: int = 64, limit: int | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"no such directory: {self.root}")

        self.size = size
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no images found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        from PIL import Image

        path = self.paths[index]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((self.size, self.size), Image.LANCZOS)
                arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill training
            raise RuntimeError(f"could not read {path}: {exc}") from exc

        srgb = arr.view(self.size, self.size, 3).permute(2, 0, 1).float() / 255.0
        return srgb_to_linear(srgb)


class SyntheticShapeDataset(Dataset):
    """Randomly generated triangle scenes, rendered on demand.

    Deterministic in the index rather than in a shared generator, so a
    DataLoader with several workers produces the same dataset regardless of how
    the work is split — a shared RNG would silently give each worker a different
    stream and make runs unreproducible.
    """

    def __init__(
        self,
        length: int = 10_000,
        size: int = 64,
        triangles: int = 32,
        sigma: float = 0.004,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.size = size
        self.triangles = triangles
        self.sigma = sigma
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        from .torch_layer import random_params, rasterize

        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        params = random_params(1, self.triangles, generator=gen)
        background = torch.rand(3, generator=gen) * 0.3
        with torch.no_grad():
            image = rasterize(
                params,
                self.size,
                self.size,
                sigma=self.sigma,
                background=tuple(background.tolist()),
            )
        return image[0]


class PrecomputedFitDataset(Dataset):
    """`(image, fitted_params)` pairs written by `python/precompute.py`.

    Fitting is expensive and entirely CPU-bound; training a network on the
    results is GPU-bound. Splitting them means neither waits on the other, and
    the fits are computed once instead of every epoch.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — generate it with python/precompute.py"
            )

        blob = torch.load(self.path, map_location="cpu")
        self.images: Tensor = blob["images"]
        self.params: Tensor = blob["params"]
        self.meta: dict = blob.get("meta", {})

        if len(self.images) != len(self.params):
            raise ValueError(
                f"corrupt dataset: {len(self.images)} images but {len(self.params)} parameter sets"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.params[index]

    @property
    def triangles(self) -> int:
        return int(self.params.shape[1])
