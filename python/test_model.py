"""Tests for the amortized model and the evaluation harness.

Skipped wholesale when torch or the compiled extension is absent, matching the
rest of the Python suite.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    from torch import nn

    from diffrast.model import N_PARAMS, TriangleNet, count_parameters
    from diffrast.torch_layer import psnr, random_params, rasterize
    from evaluate import evaluate, refine

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - environment-dependent
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch or diffrast_rs not installed")
class TestTriangleNet(unittest.TestCase):
    def test_output_shape(self) -> None:
        net = TriangleNet(triangles=12, width=8)
        self.assertEqual(net(torch.rand(2, 3, 32, 32)).shape, (2, 12, N_PARAMS))

    def test_outputs_are_always_in_valid_ranges(self) -> None:
        """The activation, not a clamp, is what keeps parameters legal.

        Driven with an absurd input so the head's pre-activations are large:
        the ranges have to hold by construction, not because the values
        happened to be small.
        """
        net = TriangleNet(triangles=8, width=8, vertex_range=0.15)
        out = net(torch.randn(4, 3, 32, 32) * 50)

        verts, colors, alpha = out[..., :6], out[..., 6:9], out[..., 9]
        self.assertTrue((verts >= -0.15).all() and (verts <= 1.15).all())
        self.assertTrue((colors >= 0).all() and (colors <= 1).all())
        # Strictly inside (0, 1): a saturated alpha gets no gradient and could
        # never recover.
        self.assertTrue((alpha > 0).all() and (alpha < 1).all())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            TriangleNet(triangles=0)
        with self.assertRaises(ValueError):
            TriangleNet(triangles=4, pool=0)
        with self.assertRaises(ValueError):
            TriangleNet(triangles=4, width=8)(torch.rand(3, 16, 16))

    def test_runs_at_a_resolution_it_was_not_built_for(self) -> None:
        # The reason for pooling at all: fit at one size, infer at another.
        net = TriangleNet(triangles=8, width=8, pool=4)
        self.assertEqual(net(torch.rand(1, 3, 32, 32)).shape, (1, 8, N_PARAMS))
        self.assertEqual(net(torch.rand(1, 3, 96, 96)).shape, (1, 8, N_PARAMS))

    def test_pooling_to_one_discards_spatial_layout(self) -> None:
        """The defect that cost this model most of its usable signal.

        Global average pooling is invariant to spatial permutation: it reports
        how much of a feature is present, never where. For inverse graphics —
        a task that is almost entirely *where* — that discards the signal the
        head needs and keeps the one it needs least.

        Asserted on the pooling layer directly rather than through training,
        because it is a property of the operation and not of any particular
        run.
        """
        features = torch.randn(1, 8, 6, 6)
        mirrored = features.flip(-1)

        blind = TriangleNet(triangles=4, width=8, pool=1).pool
        seeing = TriangleNet(triangles=4, width=8, pool=4).pool

        # Identical for a rearranged feature map — the layout is simply gone.
        torch.testing.assert_close(blind(features), blind(mirrored))
        self.assertFalse(torch.allclose(seeing(features), seeing(mirrored)))

    def test_head_starts_spread_out_rather_than_degenerate(self) -> None:
        # Zero weights and a scattered bias: every triangle starts somewhere
        # different, instead of all of them stacked at the image centre.
        net = TriangleNet(triangles=64, width=8)
        centers = net(torch.rand(1, 3, 32, 32))[0, :, :6].view(64, 3, 2).mean(1)
        self.assertGreater(centers.std(0).mean().item(), 0.1)

    def test_parameter_count_is_reported(self) -> None:
        net = TriangleNet(triangles=8, width=8)
        self.assertEqual(count_parameters(net), net.n_parameters)


class ConstantScene(nn.Module):
    """A model that ignores its input entirely.

    The exact failure the evaluation controls exist to catch: it still renders
    something plausible, and still drives a training loss down.
    """

    def __init__(self, triangles: int = 8) -> None:
        super().__init__()
        self.triangles = triangles
        self.pool_size = 1
        gen = torch.Generator().manual_seed(0)
        scene = torch.rand(1, triangles, N_PARAMS, generator=gen)
        scene[..., 9] = 0.9
        self.register_buffer("scene", scene)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.scene.expand(len(images), -1, -1)


@unittest.skipUnless(HAVE_TORCH, "torch or diffrast_rs not installed")
class TestEvaluationControls(unittest.TestCase):
    """Check that the diagnostic detects the thing it was written to detect.

    An instrument that cannot fail on a known-bad input is not evidence about
    a good one.
    """

    def images(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(1)
        return torch.rand(8, 3, 24, 24, generator=gen)

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(sigma=0.01, refine_steps=0)

    def test_input_gain_is_zero_for_a_model_that_ignores_its_input(self) -> None:
        out = evaluate(ConstantScene(), self.images(), self.args())
        # Same render for every image, so pairing renders with the wrong
        # targets must score identically up to which targets got averaged.
        self.assertLess(abs(out["input_gain_db"]), 0.5, "control failed to spot a constant model")

    def test_reports_the_baseline_it_must_beat(self) -> None:
        out = evaluate(ConstantScene(), self.images(), self.args())
        self.assertIn("mean_colour_psnr", out)
        # Uniform noise: a per-image flat fill should beat one fixed scene.
        self.assertGreater(out["mean_colour_psnr"], out["one_shot_psnr"])

    def test_refine_trace_is_indexed_by_step_count(self) -> None:
        """`trace[i]` must be the quality after exactly `i` steps.

        The natural loop renders, steps, then records the PSNR of what it just
        rendered — which describes the state *before* that step. Every entry is
        then credited to one step too many: "after 100 steps" reports 99, and
        the count of steps a random start needs to catch a prediction comes out
        one low, in the direction that flatters the model.
        """
        targets = rasterize(
            random_params(2, 6, generator=torch.Generator().manual_seed(1)),
            24, 24, sigma=0.01,
        ).detach()
        start = random_params(2, 6, generator=torch.Generator().manual_seed(2))

        final, trace = refine(start, targets, steps=5, sigma=0.02)

        self.assertEqual(len(trace), 6, "expected steps + 1 entries")
        with torch.no_grad():
            at_start = psnr(rasterize(start, 24, 24, sigma=0.02), targets).item()
            at_end = psnr(rasterize(final, 24, 24, sigma=0.02), targets).item()
        self.assertAlmostEqual(trace[0], at_start, places=3, msg="trace[0] is not the start")
        self.assertAlmostEqual(trace[-1], at_end, places=3, msg="trace[-1] is not the result")

    def test_refinement_study_runs_and_improves(self) -> None:
        args = argparse.Namespace(sigma=0.02, refine_steps=5)
        out = evaluate(ConstantScene(), self.images(), args)
        self.assertEqual(len(out["refine_trace_model"]), 6)  # steps + 1
        self.assertGreaterEqual(out["refined_from_model_psnr"], out["one_shot_psnr"])


if __name__ == "__main__":
    unittest.main()
