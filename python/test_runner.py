"""Tests for parsing and validation.

Deliberately free of any dependency on the compiled binary, so CI can run them
on a machine without a Rust toolchain and still catch the failure mode that
matters most here: silently misreading a fit's output.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffrast import FitError, load_losses, load_scene, parse_scene  # noqa: E402
from diffrast.runner import FitResult, Scene, Triangle  # noqa: E402

VALID_SCENE = {
    "version": 1,
    "background": [0.1, 0.2, 0.3],
    "triangles": [
        {"verts": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], "color": [1.0, 0.5, 0.25], "alpha": 0.75},
        {"verts": [[0.1, 0.1], [0.9, 0.2], [0.4, 0.8]], "color": [0.2, 0.4, 0.6], "alpha": 0.25},
    ],
}


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestParseScene(unittest.TestCase):
    def test_parses_a_valid_scene(self) -> None:
        scene = parse_scene(VALID_SCENE)
        self.assertEqual(len(scene), 2)
        self.assertEqual(scene.background, [0.1, 0.2, 0.3])
        self.assertAlmostEqual(scene.mean_alpha, 0.5)

    def test_rejects_an_unknown_version(self) -> None:
        raw = dict(VALID_SCENE, version=99)
        with self.assertRaises(FitError) as ctx:
            parse_scene(raw)
        self.assertIn("version", str(ctx.exception))

    def test_rejects_a_missing_version(self) -> None:
        raw = {k: v for k, v in VALID_SCENE.items() if k != "version"}
        with self.assertRaises(FitError):
            parse_scene(raw)

    def test_rejects_malformed_triangles(self) -> None:
        raw = dict(VALID_SCENE, triangles=[{"verts": [[0, 0]]}])
        with self.assertRaises(FitError):
            parse_scene(raw)

    def test_empty_scene_is_valid(self) -> None:
        scene = parse_scene({"version": 1, "background": [0, 0, 0], "triangles": []})
        self.assertEqual(len(scene), 0)
        self.assertEqual(scene.mean_alpha, 0.0)


class TestLoadScene(TempDirCase):
    def test_round_trips_through_a_file(self) -> None:
        path = self.tmp / "scene.json"
        path.write_text(json.dumps(VALID_SCENE))
        self.assertEqual(len(load_scene(path)), 2)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FitError):
            load_scene(self.tmp / "absent.json")

    def test_invalid_json_raises_with_context(self) -> None:
        path = self.tmp / "broken.json"
        path.write_text("{not json")
        with self.assertRaises(FitError) as ctx:
            load_scene(path)
        self.assertIn("valid JSON", str(ctx.exception))


class TestLoadLosses(TempDirCase):
    def write_csv(self, text: str) -> Path:
        path = self.tmp / "loss.csv"
        path.write_text(text)
        return path

    def test_reads_values_in_order(self) -> None:
        path = self.write_csv("iter,loss\n0,0.5\n1,0.25\n2,0.125\n")
        self.assertEqual(load_losses(path), [0.5, 0.25, 0.125])

    def test_header_only_is_empty_not_an_error(self) -> None:
        self.assertEqual(load_losses(self.write_csv("iter,loss\n")), [])

    def test_wrong_columns_raise(self) -> None:
        path = self.write_csv("step,value\n0,1.0\n")
        with self.assertRaises(FitError):
            load_losses(path)

    def test_malformed_value_raises(self) -> None:
        path = self.write_csv("iter,loss\n0,not-a-number\n")
        with self.assertRaises(FitError):
            load_losses(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FitError):
            load_losses(self.tmp / "absent.csv")


class TestFitResult(unittest.TestCase):
    def make(self, losses: list[float]) -> FitResult:
        return FitResult(
            out_dir=Path("out"),
            losses=losses,
            scene=Scene(background=[0, 0, 0], triangles=[]),
            triangles=4,
            seed=0,
            duration_s=1.0,
        )

    def test_reports_best_not_last(self) -> None:
        # Sigma annealing can make late iterations slightly worse, so "best"
        # must not be a synonym for "final".
        result = self.make([1.0, 0.1, 0.05, 0.08])
        self.assertEqual(result.best_loss, 0.05)
        self.assertEqual(result.initial_loss, 1.0)
        self.assertAlmostEqual(result.improvement, 20.0)

    def test_empty_losses_do_not_crash(self) -> None:
        result = self.make([])
        self.assertNotEqual(result.initial_loss, result.initial_loss)  # NaN
        self.assertNotEqual(result.improvement, result.improvement)

    def test_paths_are_derived_from_out_dir(self) -> None:
        result = self.make([1.0])
        self.assertEqual(result.fit_png.name, "fit.png")
        self.assertEqual(result.target_png.name, "target.png")
        self.assertEqual(result.frames_dir.name, "frames")


class TestSceneModel(unittest.TestCase):
    def test_mean_alpha_averages_triangles(self) -> None:
        scene = Scene(
            background=[0, 0, 0],
            triangles=[
                Triangle(verts=[[0, 0], [1, 0], [0, 1]], color=[1, 1, 1], alpha=0.2),
                Triangle(verts=[[0, 0], [1, 0], [0, 1]], color=[1, 1, 1], alpha=0.8),
            ],
        )
        self.assertAlmostEqual(scene.mean_alpha, 0.5)


if __name__ == "__main__":
    unittest.main()
