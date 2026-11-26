"""Shared, reproducible figure settings for the final report."""

import colorsys
from hashlib import sha256
from pathlib import Path

import numpy as np

REPORT_WIDTH = 6.8  # inches: full-width figure in the report
FIGURE_DPI = 180
BEFORE_BC = "#66788A"
AFTER_BC = "#207F78"


def use_report_style() -> None:
    """Apply the same type, grid and export settings in every report script."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (REPORT_WIDTH, 4.0),
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def run_color(label: str) -> str:
    """A run keeps its colour when other curves are added or removed."""
    hue = int.from_bytes(sha256(label.encode("utf-8")).digest()[:8], "big") / 2**64
    red, green, blue = colorsys.hls_to_rgb(hue, 0.39, 0.55)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def moving_average(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Trailing mean: uses only updates available at each point."""
    if window <= 0:
        raise ValueError("moving-average window must be positive")
    values = np.asarray(values, dtype=float)
    total = np.concatenate(([0.0], np.cumsum(values)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (total[end] - total[start]) / (end - start)


def save_report_figure(fig, path: str | Path) -> Path:
    """Write matching raster and vector versions of one figure."""
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return path
