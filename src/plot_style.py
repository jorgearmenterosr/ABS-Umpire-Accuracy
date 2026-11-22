"""Shared look for the study's charts: colours, fonts and small helpers.

The colours are a validated set: three categorical hues that stay distinct
for colour-blind readers, and a five-step blue ramp for the ordered tiers
(Tier 1 is the darkest step).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
TIER_COLORS = {
    1: "#104281",
    2: "#1c5cab",
    3: "#2a78d6",
    4: "#5598e7",
    5: "#86b6ef",
}
PREFERRED_FONTS = ("Helvetica Neue", "Arial", "DejaVu Sans")
SAVE_DPI = 200
POINTS_PER_INCH = 72
CORNER_RADIUS_POINTS = 4.0


def use_study_font() -> None:
    """Use the first preferred sans-serif font this computer has."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in PREFERRED_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            return


def new_figure(
    width: float = 8.0, height: float = 4.5, top: float = 0.78
) -> tuple[Figure, Axes]:
    """Return a figure and axes with the study's quiet styling."""
    use_study_font()
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.95, top=top, bottom=0.12)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=INK_MUTED, labelsize=10)
    return fig, ax


def add_title(fig: Figure, title: str, subtitle: str = "") -> None:
    """Put a left-aligned title and a quieter subtitle above the plot."""
    fig.text(
        0.02,
        0.96,
        title,
        fontsize=14,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    if subtitle:
        fig.text(
            0.02,
            0.885,
            subtitle,
            fontsize=10.5,
            color=INK_SECONDARY,
            ha="left",
            va="top",
        )


def rounded_columns(
    ax: Axes,
    positions: list,
    heights: list,
    colors: list,
    width: float = 0.26,
) -> None:
    """Draw thin columns with rounded tops that stand on the baseline.

    Set the axis limits before calling this: the corner radius is worked
    out from the final scale of the axes.
    """
    fig = ax.figure
    fig.canvas.draw()
    box = ax.get_window_extent()
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    x_per_pixel = (x_high - x_low) / box.width
    y_per_pixel = (y_high - y_low) / box.height
    aspect = y_per_pixel / x_per_pixel
    radius = CORNER_RADIUS_POINTS * fig.dpi / POINTS_PER_INCH * x_per_pixel
    overhang = 2 * radius * aspect
    for position, height, color in zip(
        positions, heights, colors, strict=True
    ):
        ax.add_patch(
            FancyBboxPatch(
                (position - width / 2, y_low - overhang),
                width,
                height - y_low + overhang,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=aspect,
                facecolor=color,
                edgecolor="none",
            )
        )


def label_columns(
    ax: Axes, positions: list, heights: list, labels: list
) -> None:
    """Write each column's value just above its top."""
    y_low, y_high = ax.get_ylim()
    gap = (y_high - y_low) * 0.015
    for position, height, label in zip(
        positions, heights, labels, strict=True
    ):
        ax.text(
            position,
            height + gap,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color=INK,
        )


def save_figure(fig: Figure, name: str) -> Path:
    """Save the figure to reports/figures as a PNG and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=SAVE_DPI, facecolor=fig.get_facecolor())
    return path
