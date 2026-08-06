"""Drive the Rust `fit` binary and read back what it produced.

The split of responsibilities across the project: Rust owns the rasterizer, the
gradients, and the optimizer; Python owns experiments and presentation. This
module is the seam — it launches fits, parses their artifacts, and hands back
plain Python objects so nothing downstream needs to know a subprocess was
involved.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class FitError(RuntimeError):
    """A fit could not be run, or ran and failed."""


@dataclass
class Triangle:
    verts: list[list[float]]
    color: list[float]
    alpha: float


@dataclass
class Scene:
    background: list[float]
    triangles: list[Triangle] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.triangles)

    @property
    def mean_alpha(self) -> float:
        if not self.triangles:
            return 0.0
        return sum(t.alpha for t in self.triangles) / len(self.triangles)


@dataclass
class FitResult:
    """Everything a single fit produced."""

    out_dir: Path
    losses: list[float]
    scene: Scene
    triangles: int
    seed: int
    duration_s: float

    @property
    def initial_loss(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def best_loss(self) -> float:
        return min(self.losses) if self.losses else float("nan")

    @property
    def improvement(self) -> float:
        """How many times smaller the best loss is than the first."""
        if not self.losses or self.best_loss == 0:
            return float("nan")
        return self.initial_loss / self.best_loss

    @property
    def fit_png(self) -> Path:
        return self.out_dir / "fit.png"

    @property
    def target_png(self) -> Path:
        return self.out_dir / "target.png"

    @property
    def frames_dir(self) -> Path:
        return self.out_dir / "frames"


def binary_path(release: bool = True) -> Path:
    """Locate the compiled `fit` binary."""
    profile = "release" if release else "debug"
    return REPO_ROOT / "target" / profile / "fit"


def ensure_binary(release: bool = True) -> Path:
    """Build the `fit` binary if it is missing, and return its path.

    Building on demand keeps the Python entry points usable from a fresh clone
    without a documented two-step dance.
    """
    path = binary_path(release)
    if path.exists():
        return path

    if shutil.which("cargo") is None:
        raise FitError(
            f"{path} not found and cargo is not on PATH — "
            "install Rust or build the binary manually"
        )

    cmd = ["cargo", "build", "--bin", "fit"]
    if release:
        cmd.append("--release")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FitError(f"cargo build failed:\n{proc.stderr.strip()}")
    if not path.exists():
        raise FitError(f"build reported success but {path} is missing")
    return path


def run_fit(
    target: str | Path | None = None,
    *,
    out_dir: str | Path = "out",
    triangles: int = 128,
    iters: int = 1500,
    size: int = 192,
    seed: int = 0,
    save_every: int = 0,
    export: int = 1024,
    release: bool = True,
    timeout: float | None = None,
) -> FitResult:
    """Run one fit and return its results.

    Raises `FitError` with the binary's own message on failure, rather than a
    bare non-zero exit code — the Rust side already writes good diagnostics and
    swallowing them here would be a step backwards.
    """
    binary = ensure_binary(release)
    out_dir = Path(out_dir)

    cmd = [str(binary)]
    if target is not None:
        target = Path(target)
        if not target.exists():
            raise FitError(f"target image does not exist: {target}")
        cmd.append(str(target))
    cmd += [
        "--out", str(out_dir),
        "--tris", str(triangles),
        "--iters", str(iters),
        "--size", str(size),
        "--seed", str(seed),
        "--export", str(export),
        "--quiet",
    ]
    if save_every > 0:
        cmd += ["--save-every", str(save_every)]

    import time

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise FitError(f"fit timed out after {timeout}s") from exc
    duration = time.perf_counter() - start

    if proc.returncode != 0:
        raise FitError(proc.stderr.strip() or "fit failed with no message")

    resolved = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    return FitResult(
        out_dir=resolved,
        losses=load_losses(resolved / "loss.csv"),
        scene=load_scene(resolved / "scene.json"),
        triangles=triangles,
        seed=seed,
        duration_s=duration,
    )


def load_losses(path: str | Path) -> list[float]:
    """Read the `iter,loss` CSV the fit binary writes."""
    path = Path(path)
    if not path.exists():
        raise FitError(f"no loss file at {path}")

    losses: list[float] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["iter", "loss"]:
            raise FitError(f"unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            try:
                losses.append(float(row["loss"]))
            except (TypeError, ValueError) as exc:
                raise FitError(f"malformed loss value in {path}: {row}") from exc
    return losses


def load_scene(path: str | Path) -> Scene:
    """Read the JSON scene the fit binary writes."""
    path = Path(path)
    if not path.exists():
        raise FitError(f"no scene file at {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FitError(f"{path} is not valid JSON: {exc}") from exc

    return parse_scene(raw)


def parse_scene(raw: dict) -> Scene:
    """Convert decoded JSON into a `Scene`, checking the schema version.

    The version check is cheap insurance: a scene written by a future format
    would otherwise be read as garbage geometry rather than reported clearly.
    """
    version = raw.get("version")
    if version != 1:
        raise FitError(f"unsupported scene version: {version!r}")

    try:
        triangles = [
            Triangle(verts=t["verts"], color=t["color"], alpha=float(t["alpha"]))
            for t in raw["triangles"]
        ]
        return Scene(background=raw["background"], triangles=triangles)
    except (KeyError, TypeError, ValueError) as exc:
        raise FitError(f"malformed scene: {exc}") from exc
