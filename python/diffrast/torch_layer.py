"""The rasterizer as a PyTorch operation.

This is what makes the renderer trainable *through*. `rasterize` is a normal
differentiable op: a network can emit triangle parameters, render them, compare
against a photo, and backpropagate all the way to its own weights. The forward
and backward passes are the same Rust code the CLI uses, called through PyO3.

    params = model(image)                    # (B, T, 10)
    render = rasterize(params, 128, 128)     # (B, 3, H, W)
    loss = F.mse_loss(render, image)
    loss.backward()                          # gradients reach `model`

Layout note: Rust works in `(B, H, W, 3)` because that is the memory order the
rasterizer writes; PyTorch wants `(B, 3, H, W)`. The conversion happens at this
boundary and nowhere else, so the rest of the Python code is idiomatic torch.
"""

from __future__ import annotations

import atexit

import torch
from torch import Tensor

try:
    import diffrast_rs
except ImportError as exc:  # pragma: no cover - import-time guidance
    raise ImportError(
        "the compiled extension `diffrast_rs` is not installed — build it with:\n"
        "    cd crates/diffrast-py && maturin develop --release"
    ) from exc

#: Parameters per triangle: 6 position + 3 color + 1 alpha.
N_PARAMS: int = diffrast_rs.params_per_triangle()


class _Rasterize(torch.autograd.Function):
    """Bridges Rust's gradients into PyTorch's autograd graph."""

    @staticmethod
    def forward(ctx, params: Tensor, height: int, width: int, sigma: float, background, device):
        # Tensors always cross the boundary as CPU float32 — the extension owns
        # its own GPU context (wgpu), which is separate from torch's CUDA
        # context. `device` selects which backend the *extension* uses, not
        # where the torch tensors live.
        params_np = params.detach().to("cpu", torch.float32).contiguous().numpy()
        images = diffrast_rs.render_batch(params_np, height, width, sigma, background, device)

        ctx.save_for_backward(params)
        ctx.sigma = sigma
        ctx.background = background
        ctx.device = device
        return torch.from_numpy(images)

    @staticmethod
    def backward(ctx, grad_images: Tensor):
        (params,) = ctx.saved_tensors
        params_np = params.detach().to("cpu", torch.float32).contiguous().numpy()
        grad_np = grad_images.detach().to("cpu", torch.float32).contiguous().numpy()

        grads = diffrast_rs.backward_batch(
            params_np, grad_np, ctx.sigma, ctx.background, ctx.device
        )
        grad_params = torch.from_numpy(grads).to(params.device, params.dtype)
        # One gradient per forward input; the other five are non-tensors.
        return grad_params, None, None, None, None, None


def rasterize(
    params: Tensor,
    height: int,
    width: int,
    sigma: float = 0.0015,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    channels_first: bool = True,
    device: str = "auto",
) -> Tensor:
    """Render a batch of triangle scenes, differentiably.

    Args:
        params: `(B, T, 10)` — `[x0, y0, x1, y1, x2, y2, r, g, b, a]` per
            triangle, in normalized image coordinates.
        height, width: output resolution.
        sigma: edge softness in normalized units. Anneal this downward during
            training exactly as the standalone fitter does — starting sharp
            leaves triangles with no gradient to follow.
        background: color the canvas is cleared to.
        channels_first: return `(B, 3, H, W)` for torch, rather than the
            `(B, H, W, 3)` the renderer produces natively.
        device: which backend the rasterizer uses — `"cpu"`, `"gpu"` or
            `"auto"`. Independent of where the torch tensors live: the
            extension has its own wgpu context.

    Returns:
        The rendered batch, in linear light.
    """
    if params.dim() != 3 or params.shape[-1] != N_PARAMS:
        raise ValueError(
            f"expected params of shape (B, T, {N_PARAMS}), got {tuple(params.shape)}"
        )
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not (sigma > 0):
        raise ValueError("sigma must be positive")

    images = _Rasterize.apply(params, height, width, sigma, background, device)
    images = images.to(params.device, params.dtype)
    return images.permute(0, 3, 1, 2).contiguous() if channels_first else images


class _FusedMSE(torch.autograd.Function):
    """Photometric MSE against a target, in one Rust call.

    `F.mse_loss(rasterize(params, ...), target)` costs **two** renders per
    training step: `rasterize` renders once for the forward, and the backward
    renders again to rebuild the tape it did not keep. The loss is a pure
    function of the parameters and the target, so both can be had from a single
    call — which is exactly what `fused_loss_backward` already computes.
    """

    @staticmethod
    def forward(ctx, params: Tensor, targets: Tensor, sigma: float, background, device):
        params_np = params.detach().to("cpu", torch.float32).contiguous().numpy()
        targets_np = targets.detach().to("cpu", torch.float32).contiguous().numpy()
        losses, grads = diffrast_rs.fused_loss_backward(
            params_np, targets_np, sigma, background, device
        )
        ctx.save_for_backward(torch.from_numpy(grads).to(params.device, params.dtype))
        ctx.batch = len(params)
        # `fused_loss_backward` returns per-item losses as a plain list.
        return torch.tensor(losses, device=params.device, dtype=params.dtype).mean()

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (grads,) = ctx.saved_tensors
        # `fused_loss_backward` differentiates the *sum* of per-item MSE; the
        # forward above returns their mean, so the chain rule needs the 1/B.
        return grad_out * grads / ctx.batch, None, None, None, None


def fused_mse(
    params: Tensor,
    targets: Tensor,
    sigma: float = 0.0015,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: str = "auto",
) -> Tensor:
    """Mean squared error between the rendered scenes and `targets`.

    Equivalent to `F.mse_loss(rasterize(params, H, W, sigma), targets)` and
    differentiable back to `params` — but with one render per step instead of
    two, which is most of the cost of a training step.

    Args:
        params: `(B, T, 10)` scene parameters.
        targets: `(B, 3, H, W)`, matching what `rasterize` returns.
    """
    if params.dim() != 3 or params.shape[-1] != N_PARAMS:
        raise ValueError(
            f"expected params of shape (B, T, {N_PARAMS}), got {tuple(params.shape)}"
        )
    if targets.dim() != 4 or targets.shape[1] != 3:
        raise ValueError(f"expected targets of shape (B, 3, H, W), got {tuple(targets.shape)}")
    if not (sigma > 0):
        raise ValueError("sigma must be positive")

    hwc = targets.permute(0, 2, 3, 1).contiguous()
    return _FusedMSE.apply(params, hwc, sigma, background, device)


def gpu_adapter() -> str | None:
    """Name of the GPU the extension will use, or `None` if there is none."""
    return diffrast_rs.gpu_adapter()


def last_device() -> str:
    """Device the most recent `fused_loss` call actually ran on.

    Worth having: with `device="auto"` and a silent CPU fallback, "it got
    faster" is not evidence that the GPU was used. Tests assert on this
    instead — a bug that quietly routed everything to the CPU passed a
    timing-based check once already.
    """
    return diffrast_rs.last_device()


def shutdown_gpu() -> bool:
    """Release the rasterizer's GPU device.

    Registered with `atexit` below, so callers never need this. Exposed for
    the case where a process wants the device gone before it exits.
    """
    return diffrast_rs.shutdown_gpu()


# A live Vulkan device torn down during library unload — in a process that is
# simultaneously shutting CUDA down — segfaulted at interpreter exit, after
# training had completed and every checkpoint was written. Releasing it at a
# point where the interpreter is still healthy avoids that ordering entirely.
atexit.register(shutdown_gpu)


def policy_prefers_gpu(
    batch: int, triangles: int, pixels: int, discrete: bool | None = None
) -> bool:
    """Whether `device="auto"`'s size rule favours the GPU for this shape.

    The size rule only — `auto` also requires an adapter. Useful for answering
    "why did auto pick the CPU here?" without timing anything.

    The rule differs for discrete and integrated GPUs, which measure as
    genuinely different regimes rather than the same one scaled. `discrete`
    defaults to whichever kind is actually present.
    """
    return diffrast_rs.policy_prefers_gpu(batch, triangles, pixels, discrete)


def fused_loss(
    params: Tensor,
    targets: Tensor,
    sigma: float = 0.0015,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: str = "auto",
) -> tuple[Tensor, Tensor]:
    """Per-item MSE and parameter gradients in one call, without autograd.

    Use this when fitting parameters directly — it skips building a graph and
    never materializes the rendered images in Python, which is a meaningful
    saving when the batch is large. For training a *network*, use `rasterize`
    instead: this returns gradients rather than routing them.

    Args:
        device: `"cpu"`, `"gpu"`, or `"auto"`. `"auto"` picks from the measured
            crossover — the GPU only pulls ahead once there is enough geometry
            to spread gradient-accumulator contention across, and below that a
            many-core CPU parallelizing over batch items wins outright.

    Returns:
        `(losses, grads)` — `(B,)` and `(B, T, 10)`.
    """
    if targets.dim() != 4:
        raise ValueError(f"expected targets of shape (B, H, W, 3), got {tuple(targets.shape)}")

    params_np = params.detach().to("cpu", torch.float32).contiguous().numpy()
    targets_np = targets.detach().to("cpu", torch.float32).contiguous().numpy()

    losses, grads = diffrast_rs.fused_loss_backward(
        params_np, targets_np, sigma, background, device
    )
    return (
        torch.tensor(losses, device=params.device),
        torch.from_numpy(grads).to(params.device, params.dtype),
    )


def random_params(
    batch: int,
    triangles: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Plausible starting parameters: small triangles scattered over the canvas.

    Sized deliberately small. Triangles initialized to span the whole image
    overlap so heavily that early gradients mostly cancel, and the fit spends
    its first hundred iterations untangling them.
    """
    centers = torch.rand(batch, triangles, 1, 2, device=device, generator=generator)
    offsets = (
        torch.rand(batch, triangles, 3, 2, device=device, generator=generator) - 0.5
    ) * 0.3
    verts = (centers + offsets).reshape(batch, triangles, 6)

    colors = torch.rand(batch, triangles, 3, device=device, generator=generator)
    alphas = (
        torch.rand(batch, triangles, 1, device=device, generator=generator) * 0.4 + 0.3
    )
    return torch.cat([verts, colors, alphas], dim=-1)


def clamp_params(params: Tensor) -> Tensor:
    """Project parameters back into their valid ranges.

    Alpha is held strictly inside `[0, 1]`: a clamped alpha receives no
    gradient, so a triangle that reached exactly 0 could never come back.
    """
    verts = params[..., :6].clamp(-0.25, 1.25)
    colors = params[..., 6:9].clamp(0.0, 1.0)
    alpha = params[..., 9:10].clamp(1e-3, 0.999)
    return torch.cat([verts, colors, alpha], dim=-1)


def psnr(a: Tensor, b: Tensor, max_value: float = 1.0) -> Tensor:
    """Peak signal-to-noise ratio in dB — the usual reconstruction metric.

    Reported alongside MSE because MSE values like `2.4e-5` are hard to compare
    across resolutions, while PSNR is directly interpretable.
    """
    mse = torch.mean((a - b) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"), device=a.device)
    return 10.0 * torch.log10(max_value**2 / mse)
