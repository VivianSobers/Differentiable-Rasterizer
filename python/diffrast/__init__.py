"""Python tooling for the differentiable rasterizer.

Rust does the rendering, gradients, and optimization. This package drives it:
running experiments, reading back results, and turning them into charts and
animations.
"""

from .anim import frame_paths, make_gif
from .plots import comparison_strip, loss_curve, sweep_curves
from .runner import (
    FitError,
    FitResult,
    Scene,
    Triangle,
    ensure_binary,
    load_losses,
    load_scene,
    parse_scene,
    run_fit,
)

__all__ = [
    "FitError",
    "FitResult",
    "Scene",
    "Triangle",
    "comparison_strip",
    "ensure_binary",
    "frame_paths",
    "load_losses",
    "load_scene",
    "loss_curve",
    "make_gif",
    "parse_scene",
    "run_fit",
    "sweep_curves",
]
