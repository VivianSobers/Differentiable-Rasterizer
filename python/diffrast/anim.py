"""Assemble fit frames into an animated GIF.

The frames are what makes the project legible at a glance: a still image of
triangles proves the renderer works, but watching them slide into place is what
shows the optimization is real.
"""

from __future__ import annotations

from pathlib import Path


def frame_paths(frames_dir: str | Path) -> list[Path]:
    """Frames in numeric order.

    Sorted by the integer in the filename rather than lexically, so frame 100
    does not land between 10 and 11 once a run exceeds a hundred frames.
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        return []

    def index(p: Path) -> int:
        digits = "".join(c for c in p.stem if c.isdigit())
        return int(digits) if digits else -1

    return sorted((p for p in frames_dir.glob("*.png")), key=index)


def make_gif(
    frames_dir: str | Path,
    path: str | Path,
    *,
    fps: int = 20,
    max_width: int = 480,
    hold_last_ms: int = 1200,
) -> Path:
    """Write a GIF from a directory of frames.

    The final frame is held for `hold_last_ms` so a looping animation ends on
    the result instead of snapping back to noise the instant it converges.
    """
    from PIL import Image

    paths = frame_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"no frames found in {frames_dir}")

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.width > max_width:
            height = max(1, round(img.height * max_width / img.width))
            img = img.resize((max_width, height), Image.LANCZOS)
        frames.append(img)

    per_frame = max(20, round(1000 / max(fps, 1)))
    durations = [per_frame] * len(frames)
    durations[-1] = max(per_frame, hold_last_ms)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return path
