"""Tests for the PyTorch layer.

Skipped wholesale when torch or the compiled extension is absent, so the rest
of the Python suite still runs on a machine without them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import torch.nn.functional as F

    from diffrast.torch_layer import (
        N_PARAMS,
        clamp_params,
        fused_loss,
        gpu_adapter,
        last_device,
        policy_prefers_gpu,
        psnr,
        random_params,
        rasterize,
    )

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - environment-dependent
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch or diffrast_rs not installed")
class TestRasterize(unittest.TestCase):
    def test_output_shape_and_layout(self) -> None:
        p = random_params(3, 8, generator=torch.Generator().manual_seed(0))
        self.assertEqual(rasterize(p, 16, 24).shape, (3, 3, 16, 24))
        self.assertEqual(
            rasterize(p, 16, 24, channels_first=False).shape, (3, 16, 24, 3)
        )

    def test_render_is_differentiable(self) -> None:
        p = random_params(1, 4, generator=torch.Generator().manual_seed(0))
        p.requires_grad_(True)
        rasterize(p, 16, 16, sigma=0.03).sum().backward()

        self.assertIsNotNone(p.grad)
        self.assertEqual(p.grad.shape, p.shape)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum(), 0, "gradient should not be identically zero")

    def test_gradient_matches_numerical_differentiation(self) -> None:
        """The load-bearing test: Rust's analytic gradient vs torch's own.

        `eps` is deliberately 5e-4. Finite-difference error is U-shaped —
        smaller steps drown in float32 cancellation, larger ones measure the
        function's curvature rather than its slope. This value sits at the
        bottom of that curve for this loss scale.
        """
        eps = 5e-4
        sigma = 0.03
        gen = torch.Generator().manual_seed(0)
        p = random_params(1, 5, generator=gen).requires_grad_(True)
        target = torch.rand(1, 3, 24, 24, generator=torch.Generator().manual_seed(1))

        loss = F.mse_loss(rasterize(p, 24, 24, sigma=sigma), target)
        loss.backward()
        analytic = p.grad.clone()

        numeric = torch.zeros_like(p)
        flat = numeric.view(-1)
        with torch.no_grad():
            base = p.detach().clone()
            for i in range(base.numel()):
                for sign in (1, -1):
                    probe = base.clone().view(-1)
                    probe[i] += sign * eps
                    flat[i] += sign * F.mse_loss(
                        rasterize(probe.view_as(base), 24, 24, sigma=sigma), target
                    )
            flat /= 2 * eps

        rel = ((analytic - numeric).norm() / numeric.norm()).item()
        cos = F.cosine_similarity(analytic.view(1, -1), numeric.view(1, -1)).item()
        self.assertLess(rel, 0.02, f"relative error {rel:.4%}")
        self.assertGreater(cos, 0.999, f"cosine similarity {cos:.6f}")

    def test_gradient_descent_reduces_loss(self) -> None:
        """End-to-end: optimizing through the layer must actually work."""
        gen = torch.Generator().manual_seed(3)
        target = rasterize(random_params(1, 6, generator=gen), 32, 32, sigma=0.002).detach()

        p = random_params(1, 6, generator=torch.Generator().manual_seed(9))
        p.requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.02)

        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = F.mse_loss(rasterize(p, 32, 32, sigma=0.03), target)
            loss.backward()
            opt.step()
            with torch.no_grad():
                p.copy_(clamp_params(p))
            if first is None:
                first = loss.item()

        self.assertLess(loss.item(), first * 0.7, f"loss {first} -> {loss.item()}")

    def test_gradients_reach_an_upstream_network(self) -> None:
        """The point of the whole layer: a network trains through the render."""
        net = torch.nn.Linear(4, 6 * N_PARAMS)
        latent = torch.randn(2, 4)
        params = net(latent).view(2, 6, N_PARAMS)

        rasterize(params, 16, 16, sigma=0.03).mean().backward()

        self.assertIsNotNone(net.weight.grad)
        self.assertTrue(torch.isfinite(net.weight.grad).all())
        self.assertGreater(net.weight.grad.abs().sum(), 0)

    def test_rasterize_routes_to_the_requested_device(self) -> None:
        gen = torch.Generator().manual_seed(0)
        p = random_params(8, 128, generator=gen).requires_grad_(True)

        rasterize(p, 64, 64, sigma=0.02, device="cpu").sum().backward()
        self.assertEqual(last_device(), "cpu")

        if gpu_adapter() is not None:
            p2 = random_params(8, 128, generator=torch.Generator().manual_seed(0))
            p2.requires_grad_(True)
            # Both the forward and the backward must route; the backward is the
            # one that matters for training and takes a different code path.
            out = rasterize(p2, 64, 64, sigma=0.02, device="gpu")
            self.assertEqual(last_device(), "gpu")
            out.sum().backward()
            self.assertEqual(last_device(), "gpu")

            rel = (p.grad - p2.grad).abs().max() / p.grad.norm().clamp_min(1e-12)
            self.assertLess(rel.item(), 1e-2, f"cpu/gpu gradient gap {rel.item():.2e}")

    def test_rejects_bad_shapes_and_sigma(self) -> None:
        p = random_params(1, 4, generator=torch.Generator().manual_seed(0))
        with self.assertRaises(ValueError):
            rasterize(p[..., :4], 16, 16)
        with self.assertRaises(ValueError):
            rasterize(p, 0, 16)
        with self.assertRaises(ValueError):
            rasterize(p, 16, 16, sigma=0.0)

    def test_background_is_visible_where_nothing_is_drawn(self) -> None:
        # A single degenerate triangle far off-canvas leaves the background.
        p = torch.tensor([[[5.0, 5.0, 5.1, 5.0, 5.05, 5.1, 1.0, 1.0, 1.0, 1.0]]])
        img = rasterize(p, 8, 8, background=(0.25, 0.5, 0.75))
        self.assertAlmostEqual(img[0, 0].mean().item(), 0.25, places=5)
        self.assertAlmostEqual(img[0, 2].mean().item(), 0.75, places=5)


@unittest.skipUnless(HAVE_TORCH, "torch or diffrast_rs not installed")
class TestFusedLoss(unittest.TestCase):
    def test_matches_autograd_path(self) -> None:
        gen = torch.Generator().manual_seed(0)
        p = random_params(2, 5, generator=gen)
        targets = torch.rand(2, 20, 20, 3, generator=torch.Generator().manual_seed(1))

        losses, grads = fused_loss(p, targets, sigma=0.03)

        pg = p.clone().requires_grad_(True)
        rendered = rasterize(pg, 20, 20, sigma=0.03, channels_first=False)
        # Per-item MSE, matching what the fused path reports.
        per_item = ((rendered - targets) ** 2).flatten(1).mean(1)
        per_item.sum().backward()

        torch.testing.assert_close(losses, per_item.detach(), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grads, pg.grad, rtol=1e-3, atol=1e-7)

    def test_device_routing_is_honoured(self) -> None:
        """Assert on the device actually used, not on whether it got faster.

        An earlier version of the binding parsed `device` and then ignored it,
        routing everything to the CPU. Timings still looked plausible and the
        gradients still matched, because they were the *same* CPU gradients —
        only `last_device()` exposed it.
        """
        gen = torch.Generator().manual_seed(0)
        p = random_params(8, 128, generator=gen)
        targets = torch.rand(8, 64, 64, 3, generator=torch.Generator().manual_seed(1))

        fused_loss(p, targets, sigma=0.02, device="cpu")
        self.assertEqual(last_device(), "cpu")

        if gpu_adapter() is not None:
            fused_loss(p, targets, sigma=0.02, device="gpu")
            self.assertEqual(last_device(), "gpu")

    def test_gpu_and_cpu_gradients_agree(self) -> None:
        if gpu_adapter() is None:
            self.skipTest("no GPU adapter")

        gen = torch.Generator().manual_seed(2)
        p = random_params(8, 128, generator=gen)
        targets = torch.rand(8, 64, 64, 3, generator=torch.Generator().manual_seed(3))

        cpu_loss, cpu_grads = fused_loss(p, targets, sigma=0.02, device="cpu")
        gpu_loss, gpu_grads = fused_loss(p, targets, sigma=0.02, device="gpu")

        torch.testing.assert_close(cpu_loss, gpu_loss, rtol=1e-4, atol=1e-7)
        rel = (cpu_grads - gpu_grads).abs().max() / cpu_grads.norm().clamp_min(1e-12)
        self.assertLess(rel.item(), 1e-3, f"relative gradient difference {rel.item():.2e}")
        # Bit-identical would mean one path silently ran the other's code.
        self.assertFalse(
            torch.equal(cpu_grads, gpu_grads),
            "identical to the last bit — the GPU path probably did not run",
        )

    def test_unknown_device_is_rejected(self) -> None:
        gen = torch.Generator().manual_seed(0)
        p = random_params(2, 8, generator=gen)
        with self.assertRaises(ValueError):
            fused_loss(p, torch.rand(2, 16, 16, 3), device="banana")

    def test_rejects_bad_target_shape(self) -> None:
        p = random_params(1, 4, generator=torch.Generator().manual_seed(0))
        with self.assertRaises(ValueError):
            fused_loss(p, torch.rand(1, 3, 8, 8, 1))


@unittest.skipUnless(HAVE_TORCH, "torch or diffrast_rs not installed")
class TestDispatchPolicy(unittest.TestCase):
    """Pin `device="auto"` to the crossover it claims to be derived from.

    The table is `gpu_bench`'s crossover on an idle RTX 4090 against a 26-core
    i9-13900, batch of 16, as recorded in `docs/gpu-report.txt`. Values are
    GPU-over-CPU speedups: above 1.0 the GPU won. Cells inside +/-10% are
    genuine ties and are not asserted on — either choice is defensible there,
    and pinning a tie would make the test fail on noise rather than on a
    regression.

    This runs without a GPU, which is the point: it checks the *rule*, not the
    hardware. `test_device_routing_is_honoured` covers the rule being obeyed.
    """

    MEASURED = [
        # (pixels, triangles, gpu speedup over cpu)
        (64 * 64, 32, 0.83),
        (64 * 64, 128, 2.04),
        (64 * 64, 512, 2.37),
        (128 * 128, 32, 1.02),
        (128 * 128, 128, 3.61),
        (128 * 128, 512, 3.52),
        (256 * 256, 32, 0.79),
        (256 * 256, 128, 2.41),
        (256 * 256, 512, 3.38),
    ]

    def test_policy_matches_the_measured_crossover(self) -> None:
        for pixels, tris, speedup in self.MEASURED:
            if 0.9 <= speedup <= 1.1:
                continue
            with self.subTest(pixels=pixels, triangles=tris):
                self.assertEqual(
                    policy_prefers_gpu(16, tris, pixels),
                    speedup > 1.0,
                    f"{tris} tris at {pixels} px measured {speedup:.2f}x",
                )

    def test_tiny_batches_stay_on_the_cpu(self) -> None:
        # One image cannot amortize a dispatch no matter how much geometry it
        # carries; batching is the whole reason the GPU path exists.
        self.assertFalse(policy_prefers_gpu(1, 512, 256 * 256))
        self.assertTrue(policy_prefers_gpu(16, 512, 256 * 256))

    def test_trivial_workloads_stay_on_the_cpu(self) -> None:
        # Below the floor the fixed overhead dominates whatever it saves.
        self.assertFalse(policy_prefers_gpu(4, 64, 16 * 16))


@unittest.skipUnless(HAVE_TORCH, "torch or diffrast_rs not installed")
class TestHelpers(unittest.TestCase):
    def test_random_params_are_in_range(self) -> None:
        p = random_params(4, 10, generator=torch.Generator().manual_seed(0))
        self.assertEqual(p.shape, (4, 10, N_PARAMS))
        self.assertTrue((p[..., 6:9] >= 0).all() and (p[..., 6:9] <= 1).all())
        self.assertTrue((p[..., 9] > 0).all() and (p[..., 9] < 1).all())

    def test_clamp_keeps_alpha_off_the_boundary(self) -> None:
        p = torch.full((1, 1, N_PARAMS), 5.0)
        p[..., 9] = 1.0
        out = clamp_params(p)
        self.assertLess(out[0, 0, 9].item(), 1.0, "alpha at exactly 1 gets no gradient")
        self.assertGreater(out[0, 0, 9].item(), 0.0)
        self.assertLessEqual(out[0, 0, 0].item(), 1.25)

    def test_psnr_is_infinite_for_identical_images(self) -> None:
        a = torch.rand(1, 3, 8, 8)
        self.assertTrue(torch.isinf(psnr(a, a)))
        self.assertGreater(psnr(a, a * 0.99).item(), 20.0)


if __name__ == "__main__":
    unittest.main()
