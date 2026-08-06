"""Charts for fit results.

Color use follows one rule: hues are assigned in a fixed order and never
cycled, so a series keeps its color no matter how many others are present. The
palette below is validated for colorblind separation; two of its slots fall
below 3:1 contrast on the chart surface, which is why every series is also
directly labeled rather than relying on a legend swatch alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in CI; must be set before pyplot is imported.

import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order — index by series position, never rotate.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e0"


def _style_axes(ax) -> None:
    """Recede the frame so the data is the most prominent thing on the chart."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(INK_MUTED)


def loss_curve(losses: list[float], path: str | Path, title: str | None = None) -> Path:
    """Plot loss against iteration on a log y-axis.

    Log scale is not decoration here: a good fit drops two or three orders of
    magnitude, and on a linear axis everything after the first hundred
    iterations collapses onto the floor and looks like nothing is happening.
    """
    if not losses:
        raise ValueError("no losses to plot")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    _style_axes(ax)

    # One series, so no legend — the title names it.
    ax.plot(range(len(losses)), losses, color=SERIES[0], linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("iteration", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("MSE loss (log)", color=INK_MUTED, fontsize=10)

    best_i = min(range(len(losses)), key=losses.__getitem__)
    ax.plot([best_i], [losses[best_i]], "o", color=SERIES[0], markersize=8)
    ax.annotate(
        f"best {losses[best_i]:.2e} @ {best_i}",
        xy=(best_i, losses[best_i]),
        xytext=(6, 10),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )

    improvement = losses[0] / losses[best_i] if losses[best_i] else float("nan")
    ax.set_title(
        title or f"Loss over training — {improvement:.0f}x reduction",
        color=INK,
        fontsize=12,
        loc="left",
        pad=12,
    )

    return _save(fig, path)


def sweep_curves(
    runs: dict[str, list[float]], path: str | Path, title: str = "Loss by triangle count"
) -> Path:
    """Overlay several loss curves, one per configuration.

    Each line is directly labeled at its right end in addition to the legend,
    so identity never rests on color alone.
    """
    if not runs:
        raise ValueError("no runs to plot")
    if len(runs) > len(SERIES):
        raise ValueError(
            f"{len(runs)} series exceeds the {len(SERIES)}-slot palette; "
            "facet into small multiples instead of inventing hues"
        )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    _style_axes(ax)

    for i, (label, losses) in enumerate(runs.items()):
        if not losses:
            continue
        color = SERIES[i]
        ax.plot(range(len(losses)), losses, color=color, linewidth=2, label=label)
        ax.annotate(
            label,
            xy=(len(losses) - 1, losses[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
        )

    ax.set_yscale("log")
    ax.set_xlabel("iteration", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("MSE loss (log)", color=INK_MUTED, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)

    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)

    # Leave room for the right-edge labels.
    ax.margins(x=0.12)
    return _save(fig, path)


def comparison_strip(
    target_png: str | Path, fit_png: str | Path, path: str | Path
) -> Path:
    """Target and result side by side, at matching height."""
    from PIL import Image

    target = Image.open(target_png).convert("RGB")
    fitted = Image.open(fit_png).convert("RGB")

    height = max(target.height, fitted.height)
    scaled = []
    for img in (target, fitted):
        if img.height != height:
            width = max(1, round(img.width * height / img.height))
            img = img.resize((width, height), Image.LANCZOS)
        scaled.append(img)

    gap = 12
    total = sum(i.width for i in scaled) + gap
    strip = Image.new("RGB", (total, height), SURFACE)
    x = 0
    for img in scaled:
        strip.paste(img, (x, 0))
        x += img.width + gap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path)
    return path


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
