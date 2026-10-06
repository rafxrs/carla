"""Pure geometry: rectangles, the window layout, and the board's world-to-screen transform with
hit-testing.

World coordinates are `carla.engine.topology`'s unit-hex `*_XY` (hex size 1, y grows downwards);
screen coordinates are Arcade's (logical pixels, y grows upwards).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, sqrt

from carla.engine.topology import (
    CORNER_XY,
    EDGE_CORNERS,
    EDGE_XY,
    NUM_CORNERS,
    NUM_EDGES,
    NUM_TILES,
    PORT_EDGES,
    TILE_CORNERS,
    TILE_EDGES,
    TILE_XY,
)
from carla.gui import theme

Point = tuple[float, float]

SQRT3 = sqrt(3)
CORNER_PICK = 0.3
"""Click radius around a corner, in hex sizes (neighbouring corners are 1 apart)."""
EDGE_PICK = 0.28
"""Click distance from an edge's segment, in hex sizes."""
PORT_OFFSET = 0.62
"""Distance of a port marker beyond the middle of its coastal edge, in hex sizes."""
PORT_RADIUS = 0.27


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle: left `x`, bottom `y`, width `w`, height `h`."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.top

    def inset(self, dx: float, dy: float | None = None) -> "Rect":
        dy = dx if dy is None else dy
        return Rect(self.x + dx, self.y + dy, max(0.0, self.w - 2 * dx), max(0.0, self.h - 2 * dy))

    def take_top(self, h: float) -> tuple["Rect", "Rect"]:
        """(strip of height `h` along the top, the rest below it)."""
        h = min(h, self.h)
        return Rect(self.x, self.top - h, self.w, h), Rect(self.x, self.y, self.w, self.h - h)

    def take_left(self, w: float) -> tuple["Rect", "Rect"]:
        """(strip of width `w` along the left, the rest to its right)."""
        w = min(w, self.w)
        return Rect(self.x, self.y, w, self.h), Rect(self.x + w, self.y, self.w - w, self.h)

    def columns(self, n: int, gap: float = 0.0) -> list["Rect"]:
        """`n` equal side-by-side cells separated by `gap`."""
        w = (self.w - gap * (n - 1)) / n
        return [Rect(self.x + i * (w + gap), self.y, w, self.h) for i in range(n)]


@dataclass(frozen=True)
class Layout:
    """Window areas: the board column on the left, the side panel and the log on the right."""

    width: float
    height: float
    board: Rect
    """Whole left column (sea background)."""
    status: Rect
    """Top strip: mode badge, prompt or notification, help hint."""
    playback: Rect
    """Strip under the status: play/pause, steps, the seek slider and the speed."""
    viewport: Rect
    """Where the board itself is fitted."""
    hand: Rect
    """Bottom strip: the cards of a hand and, in Play mode, the action buttons."""
    side: Rect
    log: Rect


def compute_layout(width: float, height: float) -> Layout:
    lo, hi = theme.SIDE_WIDTH
    side_w = min(max(width * 0.3, lo), hi, width / 2)
    board = Rect(0, 0, width - side_w, height)
    status, rest = board.take_top(theme.STATUS_HEIGHT)
    playback, rest = rest.take_top(theme.PLAYBACK_HEIGHT)
    lo_h, hi_h = theme.HAND_HEIGHT
    hand = Rect(0, 0, board.w, min(max(height * 0.13, lo_h), hi_h, rest.h / 2))
    viewport = Rect(theme.PAD, hand.top, board.w - 2 * theme.PAD, rest.h - hand.h)
    side_h = min(theme.SIDE_HEIGHT, max(height - theme.LOG_MIN_HEIGHT, height * 0.55))
    side = Rect(board.w, height - side_h, side_w, side_h)
    log = Rect(board.w, 0, side_w, height - side_h)
    return Layout(width, height, board, status, playback, viewport, hand, side, log)


def _port_positions() -> tuple[Point, ...]:
    tile_of = {e: t for t, edges in enumerate(TILE_EDGES) for e in edges}
    out = []
    for e in PORT_EDGES:
        (ex, ey), (tx, ty) = EDGE_XY[e], TILE_XY[tile_of[e]]
        n = hypot(ex - tx, ey - ty)
        out.append((ex + (ex - tx) / n * PORT_OFFSET, ey + (ey - ty) / n * PORT_OFFSET))
    return tuple(out)


PORT_XY: tuple[Point, ...] = _port_positions()
"""World position of each port slot's marker, just off its coastal edge."""

_MARGIN = PORT_RADIUS + 0.12
EXTENT_X = max(abs(x) for x, _ in (*CORNER_XY, *PORT_XY)) + _MARGIN
EXTENT_Y = max(abs(y) for _, y in (*CORNER_XY, *PORT_XY)) + _MARGIN
"""Half-size of the world area the transform fits into the viewport (board plus ports)."""


def segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from `p` to the segment `a`-`b`."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    t = max(0.0, min(1.0, t))
    return hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def in_hex(p: Point, center: Point) -> bool:
    """`p` lies in the pointy-top hexagon of size 1 around `center` (world units)."""
    dx, dy = abs(p[0] - center[0]), abs(p[1] - center[1])
    return dx <= SQRT3 / 2 + 1e-9 and dy <= 1 - dx / SQRT3 + 1e-9


class BoardTransform:
    """Fits the board into `viewport`: world (hex size 1, y down) to screen (pixels, y up)."""

    def __init__(self, viewport: Rect) -> None:
        self.viewport = viewport
        self.scale = max(1e-6, min(viewport.w / (2 * EXTENT_X), viewport.h / (2 * EXTENT_Y)))
        """Pixels per hex size."""
        self.cx, self.cy = viewport.cx, viewport.cy
        self.corners = tuple(self.to_screen(*p) for p in CORNER_XY)
        self.edges = tuple(self.to_screen(*p) for p in EDGE_XY)
        self.tiles = tuple(self.to_screen(*p) for p in TILE_XY)
        self.ports = tuple(self.to_screen(*p) for p in PORT_XY)

    def to_screen(self, wx: float, wy: float) -> Point:
        return self.cx + wx * self.scale, self.cy - wy * self.scale

    def to_world(self, sx: float, sy: float) -> Point:
        return (sx - self.cx) / self.scale, (self.cy - sy) / self.scale

    def offset(self, origin: Point, dx: float, dy: float) -> Point:
        """Screen point `(dx, dy)` hex sizes away from screen point `origin` (dy up)."""
        return origin[0] + dx * self.scale, origin[1] + dy * self.scale

    def hex_points(self, tile: int, inset: float = 1.0) -> list[Point]:
        """Screen corners of `tile`, clockwise on screen from the top, shrunk by `inset`."""
        cx, cy = self.tiles[tile]
        return [
            (cx + (x - cx) * inset, cy + (y - cy) * inset)
            for x, y in (self.corners[c] for c in TILE_CORNERS[tile])
        ]

    def edge_points(self, edge: int, trim: float = 0.0) -> tuple[Point, Point]:
        """Screen end points of `edge`, each pulled `trim` (fraction of the length) inwards."""
        a, b = (self.corners[c] for c in EDGE_CORNERS[edge])
        dx, dy = b[0] - a[0], b[1] - a[1]
        return (a[0] + dx * trim, a[1] + dy * trim), (b[0] - dx * trim, b[1] - dy * trim)

    # ------------------------------------------------------------------ hit-testing

    def corner_at(
        self, sx: float, sy: float, candidates: Iterable[int] | None = None
    ) -> int | None:
        """Nearest corner (among `candidates`, default all) within the click radius."""
        p = self.to_world(sx, sy)
        best, best_d = None, CORNER_PICK
        for c in range(NUM_CORNERS) if candidates is None else candidates:
            d = hypot(p[0] - CORNER_XY[c][0], p[1] - CORNER_XY[c][1])
            if d <= best_d:
                best, best_d = c, d
        return best

    def edge_at(self, sx: float, sy: float, candidates: Iterable[int] | None = None) -> int | None:
        """Nearest edge (among `candidates`, default all) within the click distance."""
        p = self.to_world(sx, sy)
        best, best_d = None, EDGE_PICK
        for e in range(NUM_EDGES) if candidates is None else candidates:
            a, b = EDGE_CORNERS[e]
            d = segment_distance(p, CORNER_XY[a], CORNER_XY[b])
            if d <= best_d:
                best, best_d = e, d
        return best

    def tile_at(self, sx: float, sy: float, candidates: Iterable[int] | None = None) -> int | None:
        """Tile (among `candidates`, default all) whose hexagon contains the point."""
        p = self.to_world(sx, sy)
        for t in range(NUM_TILES) if candidates is None else candidates:
            if in_hex(p, TILE_XY[t]):
                return t
        return None
