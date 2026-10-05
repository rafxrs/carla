"""Draw the static board topology with tile/corner/edge ids and port slots (docs/img/board.png).

Usage: python scripts/render_board.py [output.png]
"""

import sys
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from carla.engine.topology import (
    CORNER_XY,
    EDGE_CORNERS,
    EDGE_XY,
    PORT_EDGES,
    TILE_CORNERS,
    TILE_XY,
)

SURFACE = "#fcfcfb"
TILE_FILL = "#f0efec"
BORDER = "#c3c2b7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
ACCENT = "#2a78d6"


def flip(xy):
    return xy[0], -xy[1]


def main(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 11), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for t, corners in enumerate(TILE_CORNERS):
        ax.add_patch(
            Polygon(
                [flip(CORNER_XY[c]) for c in corners],
                closed=True,
                facecolor=TILE_FILL,
                edgecolor=BORDER,
                linewidth=1.2,
                zorder=1,
            )
        )
        x, y = flip(TILE_XY[t])
        ax.text(x, y, f"T{t}", ha="center", va="center", fontsize=15, weight="bold", color=INK)

    for slot, e in enumerate(PORT_EDGES):
        (ax1, ay1), (bx1, by1) = (flip(CORNER_XY[c]) for c in EDGE_CORNERS[e])
        ax.plot([ax1, bx1], [ay1, by1], color=ACCENT, linewidth=5, solid_capstyle="round", zorder=2)
        mx, my = flip(EDGE_XY[e])
        cx, cy = mx * 1.2, my * 1.2  # push the label outwards from the board centre
        ax.text(cx, cy, f"port {slot}", ha="center", va="center", fontsize=9, color=INK_2)

    for e, xy in enumerate(EDGE_XY):
        x, y = flip(xy)
        ax.text(
            x,
            y,
            str(e),
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            color=MUTED,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": SURFACE, "edgecolor": "none"},
            zorder=4,
        )

    for c, xy in enumerate(CORNER_XY):
        x, y = flip(xy)
        ax.add_patch(
            Circle((x, y), 0.17, facecolor=SURFACE, edgecolor=INK_2, linewidth=1, zorder=3)
        )
        ax.text(x, y, str(c), ha="center", va="center", fontsize=7.5, color=INK, zorder=4)

    ax.text(
        0,
        -5.0,
        "T = tile id   ○ = corner id   italic = edge id   blue = port slot (types shuffled per game)",
        ha="center",
        fontsize=10,
        color=INK_2,
    )
    ax.set_xlim(-5.6 * sqrt(3) / 1.6, 5.6 * sqrt(3) / 1.6)
    ax.set_ylim(-5.5, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/img/board.png"))
