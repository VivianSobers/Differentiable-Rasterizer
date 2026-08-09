"""A network that predicts a triangle scene from an image in one forward pass.

Fitting an image takes hundreds of gradient steps. This learns the *mapping* —
amortized inverse graphics — so a trained model produces a scene in a single
forward pass, which the fitter can then refine in a fraction of the iterations.

The output head matters more than the backbone. Triangle parameters have hard
constraints (colors in `[0, 1]`, alpha strictly inside it, vertices near the
canvas), and the clean way to enforce them is in the activation rather than by
clamping afterwards: a clamp kills the gradient exactly when a parameter is out
of range and most needs one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

N_PARAMS = 10


class ConvBlock(nn.Module):
    """Conv → GroupNorm → SiLU, halving resolution.

    GroupNorm rather than BatchNorm: batches here are small enough that batch
    statistics get noisy, and GroupNorm behaves identically at train and eval
    time, which removes a whole class of "works in training, fails at inference"
    bug.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.norm(self.conv(x)))


class TriangleNet(nn.Module):
    """Image → `(B, triangles, 10)` scene parameters.

    Args:
        triangles: how many triangles to emit. Fixed per model — the count is
            baked into the output head.
        width: channel width of the first block; the backbone doubles it at
            each stage.
        vertex_range: how far outside the canvas a vertex may land. Triangles
            clipped by the frame edge are legitimate and common, so the head
            must be able to express them.
        pool: side length of the adaptive pool before the head. See below —
            this is the single most consequential choice in the architecture.
    """

    def __init__(
        self,
        triangles: int = 128,
        width: int = 32,
        vertex_range: float = 0.15,
        in_channels: int = 3,
        pool: int = 4,
    ) -> None:
        super().__init__()
        if triangles <= 0:
            raise ValueError("triangles must be positive")
        if pool <= 0:
            raise ValueError("pool must be positive")

        self.triangles = triangles
        self.vertex_range = vertex_range
        self.pool_size = pool

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            ConvBlock(width, width * 2),
            ConvBlock(width * 2, width * 4),
            ConvBlock(width * 4, width * 8),
            ConvBlock(width * 8, width * 8),
        )
        # Adaptive pooling lets one model accept any input resolution: it can
        # be trained at 64px and run at 256px without reshaping the head.
        #
        # That is a statement about tensor shapes, and it is worth being clear
        # that it is *not* a statement about accuracy. Measured: a model
        # trained at 128px scored +4.44 dB of margin over baseline at 128px
        # and -1.17 dB at 96px, from the same weights. The shapes fit; the
        # model does not transfer. Train with `--sizes` if you need it to.
        #
        # The size matters enormously, and pooling to 1x1 — the obvious choice,
        # and what this was — is close to the worst possible one *for this
        # task*. Global average pooling is invariant to spatial permutation: it
        # reports how much of each feature is present anywhere in the image and
        # discards where. Inverse graphics is almost entirely a question of
        # where, so that bottleneck throws away the signal the head needs and
        # keeps the one it needs least.
        #
        # A 4x4 pool keeps the shape-flexibility that motivated pooling in the
        # first place while preserving coarse layout. `pool=1` is kept so the
        # two can be compared rather than argued about; see
        # `docs/AMORTIZED.md` for what that comparison found.
        self.pool = nn.AdaptiveAvgPool2d(pool)

        hidden = max(512, triangles * 4)
        self.head = nn.Sequential(
            nn.Linear(width * 8 * pool * pool, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, triangles * N_PARAMS),
        )

        self._init_head()

    def _init_head(self) -> None:
        """Start near a sane scene rather than at random.

        With default initialization the head's pre-activations sit near zero,
        which after the sigmoids below puts every triangle at the image center
        with identical mid-gray color — a degenerate start the optimizer wastes
        many steps escaping. Zeroing the final layer and biasing it instead
        gives every triangle a distinct, well-spread starting position.
        """
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)

        bias = torch.empty(self.triangles, N_PARAMS)
        # Vertices: scattered centers, small offsets. Inverse-sigmoid space, so
        # these are logits rather than coordinates.
        centers = torch.rand(self.triangles, 1, 2) * 4.0 - 2.0
        offsets = torch.randn(self.triangles, 3, 2) * 0.5
        bias[:, :6] = (centers + offsets).reshape(self.triangles, 6)
        bias[:, 6:9] = torch.randn(self.triangles, 3) * 0.5
        bias[:, 9] = 0.0  # sigmoid(0) = 0.5 alpha
        with torch.no_grad():
            final.bias.copy_(bias.reshape(-1))

    def forward(self, images: Tensor) -> Tensor:
        """Predict scene parameters for a batch of `(B, 3, H, W)` images."""
        if images.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(images.shape)}")

        x = self.stem(images)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        raw = self.head(x).view(-1, self.triangles, N_PARAMS)
        return self.activate(raw)

    def activate(self, raw: Tensor) -> Tensor:
        """Map unconstrained network output into valid parameter ranges.

        Every output is squashed rather than clamped, so a parameter pinned at
        the edge of its range still receives gradient and can move back.
        """
        lo, hi = -self.vertex_range, 1.0 + self.vertex_range
        verts = torch.sigmoid(raw[..., :6]) * (hi - lo) + lo
        colors = torch.sigmoid(raw[..., 6:9])
        # Alpha squeezed strictly inside (0, 1): the rasterizer gives a fully
        # transparent or fully opaque triangle no alpha gradient, so a triangle
        # that saturated could never recover.
        alpha = torch.sigmoid(raw[..., 9:10]) * 0.998 + 0.001
        return torch.cat([verts, colors, alpha], dim=-1)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
