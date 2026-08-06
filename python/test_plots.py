"""Tests for chart and animation output."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diffrast import comparison_strip, frame_paths, loss_curve, make_gif, sweep_curves  # noqa: E402
from diffrast.plots import SERIES  # noqa: E402


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestPalette(unittest.TestCase):
    def test_slots_are_unique_and_well_formed(self) -> None:
        self.assertEqual(len(SERIES), len(set(SERIES)), "duplicate hue would merge two series")
        for hex_code in SERIES:
            self.assertRegex(hex_code, r"^#[0-9a-f]{6}$")


class TestLossCurve(TempDirCase):
    def test_writes_a_png(self) -> None:
        path = loss_curve([1.0, 0.5, 0.2, 0.1], self.tmp / "loss.png")
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1000)

    def test_creates_missing_parent_directories(self) -> None:
        path = loss_curve([1.0, 0.5], self.tmp / "nested" / "deep" / "loss.png")
        self.assertTrue(path.exists())

    def test_empty_losses_raise(self) -> None:
        with self.assertRaises(ValueError):
            loss_curve([], self.tmp / "loss.png")

    def test_single_point_is_plottable(self) -> None:
        self.assertTrue(loss_curve([0.5], self.tmp / "one.png").exists())


class TestSweepCurves(TempDirCase):
    def test_writes_a_png_for_several_runs(self) -> None:
        runs = {"32 tris": [1.0, 0.4, 0.2], "128 tris": [1.0, 0.3, 0.1]}
        self.assertTrue(sweep_curves(runs, self.tmp / "sweep.png").exists())

    def test_refuses_more_series_than_palette_slots(self) -> None:
        # Inventing a hue for a 5th series is exactly what the palette rule
        # forbids, so this must fail loudly rather than cycle colors.
        runs = {f"{i}": [1.0, 0.5] for i in range(len(SERIES) + 1)}
        with self.assertRaises(ValueError) as ctx:
            sweep_curves(runs, self.tmp / "sweep.png")
        self.assertIn("palette", str(ctx.exception))

    def test_empty_runs_raise(self) -> None:
        with self.assertRaises(ValueError):
            sweep_curves({}, self.tmp / "sweep.png")


class TestFrames(TempDirCase):
    def make_frames(self, count: int, size: tuple[int, int] = (16, 16)) -> Path:
        from PIL import Image

        frames = self.tmp / "frames"
        frames.mkdir()
        for i in range(count):
            Image.new("RGB", size, (i * 7 % 256, 100, 150)).save(
                frames / f"frame_{i:05d}.png"
            )
        return frames

    def test_frames_sort_numerically_not_lexically(self) -> None:
        frames = self.make_frames(12)
        names = [p.stem for p in frame_paths(frames)]
        # Lexical ordering would put frame 10 before frame 2 without padding;
        # the numeric key must hold regardless of padding.
        self.assertEqual(names[-1], "frame_00011")
        self.assertEqual(len(names), 12)

    def test_missing_directory_returns_empty(self) -> None:
        self.assertEqual(frame_paths(self.tmp / "nope"), [])

    def test_makes_a_gif(self) -> None:
        frames = self.make_frames(5)
        gif = make_gif(frames, self.tmp / "out.gif", fps=10)
        self.assertTrue(gif.exists())

        from PIL import Image

        with Image.open(gif) as img:
            self.assertEqual(img.n_frames, 5)

    def test_gif_downscales_wide_frames(self) -> None:
        frames = self.make_frames(3, size=(1200, 600))
        gif = make_gif(frames, self.tmp / "wide.gif", max_width=200)

        from PIL import Image

        with Image.open(gif) as img:
            self.assertEqual(img.width, 200)
            self.assertEqual(img.height, 100, "aspect ratio must be preserved")

    def test_no_frames_raises(self) -> None:
        (self.tmp / "empty").mkdir()
        with self.assertRaises(FileNotFoundError):
            make_gif(self.tmp / "empty", self.tmp / "out.gif")


class TestComparisonStrip(TempDirCase):
    def test_joins_two_images_of_different_sizes(self) -> None:
        from PIL import Image

        a = self.tmp / "a.png"
        b = self.tmp / "b.png"
        Image.new("RGB", (64, 64), (200, 50, 50)).save(a)
        Image.new("RGB", (128, 128), (50, 50, 200)).save(b)

        strip = comparison_strip(a, b, self.tmp / "strip.png")
        with Image.open(strip) as img:
            # Both scaled to the taller height, plus the gap between them.
            self.assertEqual(img.height, 128)
            self.assertEqual(img.width, 128 + 128 + 12)


if __name__ == "__main__":
    unittest.main()
