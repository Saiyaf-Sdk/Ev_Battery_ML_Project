"""Shared, colour-blind-checked chart styling for every Member A figure.

Healthy is always blue and Failure is always orange, in every chart, so a
reader never has to re-learn the colour key.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HEALTHY = "#2a78d6"   # categorical slot 1 (blue)
FAILURE = "#eb6834"   # categorical slot 2 (orange)
CLASS_COLORS = {0: HEALTHY, 1: FAILURE}
CLASS_NAMES = {0: "Healthy (0)", 1: "Failure (1)"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#b9b8b1"

# Diverging blue <-> grey <-> red for correlations (-1 .. 0 .. +1).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a85", "#a8201f"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
