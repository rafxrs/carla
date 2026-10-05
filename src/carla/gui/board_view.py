"""Board drawing: sea, tiles with small procedural motifs, number tokens, ports, pieces, the
robber and the highlights of legal targets, all as flat shapes in the theme colours.

The static part (everything that only depends on the board and the window size) is built once
into a keyed layer; pieces and highlights are rebuilt every frame.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import sin, tau

from carla.engine.board import Board
from carla.engine.constants import DESERT, GENERIC_PORT, PIPS, Building, Resource
from carla.engine.state import GameState
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, NUM_TILES, PORT_CORNERS
from carla.gui import theme
from carla.gui.geometry import PORT_RADIUS, BoardTransform, Point, Rect
from carla.gui.render import Layer, scaled
from carla.gui.theme import Color


@dataclass
class BoardMarks:
    """What to emphasise on top of the position."""

    corners: dict[int, bool] = field(default_factory=dict)
    """Clickable corners, True where the click upgrades a settlement to a city."""
    edges: frozenset[int] = frozenset()
    tiles: frozenset[int] = frozenset()
    hover: tuple[str, int] | None = None
    """("corner" | "edge" | "tile", id) of the target under the mouse."""
    player: int = -1
    """Seat whose ghost piece previews the hovered target."""
    flash: tuple[str, int] | None = None
    """Piece placed (or robber moved) by the last step, glowing for a moment."""
    flash_strength: float = 0.0
    rolled: int = 0
    """Number just rolled: its tokens glow."""


Motif = Callable[[Layer, Point, float, int], None]


class BoardView:
    def __init__(self) -> None:
        self.transform = BoardTransform(Rect(0, 0, 800, 600))
        self._static: Layer | None = None

    def set_viewport(self, viewport: Rect) -> None:
        if viewport != self.transform.viewport:
            self.transform = BoardTransform(viewport)

    def static_layer(self, board: Board, sea: Rect) -> Layer:
        """Sea, island, tiles, tokens and ports; rebuilt only when the board or size changes."""
        key = (board, self.transform.viewport, sea)
        if self._static is None or self._static.key != key:
            self._static = Layer("board", key)
            self._draw_static(self._static, board, sea)
        return self._static

    def robber_point(self, board: Board, tile: int) -> Point:
        """Where the robber stands on `tile`: beside the number token, or centred on the desert."""
        x, y = self.transform.tiles[tile]
        offset = 0.0 if board.tile_resource[tile] == DESERT else -0.52 * self.transform.scale
        return x + offset, y

    # ------------------------------------------------------------------ static part

    def _draw_static(self, layer: Layer, board: Board, sea: Rect) -> None:
        tr, s = self.transform, self.transform.scale
        layer.rect(sea.x, sea.y, sea.w, sea.h, theme.SEA)
        for x, y in tr.tiles:
            layer.circle(x, y, 1.62 * s, theme.SHALLOWS)
        for x, y in tr.tiles:
            layer.circle(x, y, 1.1 * s, theme.BEACH)
        for slot in range(len(board.port_types)):
            mx, my = tr.ports[slot]
            for c in PORT_CORNERS[slot]:
                layer.line(mx, my, *tr.corners[c], max(2.0, 0.07 * s), theme.PIER, True)
        for t in range(NUM_TILES):
            terrain = board.tile_resource[t]
            fill = theme.TERRAIN[terrain]
            hexagon = tr.hex_points(t, 0.975)
            layer.polygon(hexagon, fill)
            edge = theme.mix(fill, theme.BLACK, 0.22)
            layer.outline(hexagon, tr.tiles[t], max(1.5, 0.035 * s), edge)
            MOTIFS[terrain](layer, tr.tiles[t], s, -1 if t % 2 else 1)
        for t in range(NUM_TILES):
            if board.tile_number[t]:
                _token(layer, tr.tiles[t], board.tile_number[t], s)
        for slot, kind in enumerate(board.port_types):
            _port(layer, tr.ports[slot], kind, s)

    # ------------------------------------------------------------------ dynamic part

    def pieces_layer(self, state: GameState, marks: BoardMarks, t: float) -> Layer:
        """Roads, buildings, the robber, highlights of legal targets and hover previews."""
        layer = Layer("pieces")
        tr, s = self.transform, self.transform.scale
        pulse = 0.5 + 0.5 * sin(t * tau * 0.9)
        target = theme.with_alpha(theme.TARGET, 120 + 120 * pulse)
        board = state.board
        if marks.rolled:
            for tile in board.tiles_by_number[marks.rolled]:
                if tile != state.robber_tile:
                    x, y = tr.tiles[tile]
                    layer.ring(
                        x, y, 0.37 * s, max(2.0, 0.05 * s), theme.with_alpha(theme.ACCENT, 230)
                    )
        for tile in marks.tiles:
            layer.outline(tr.hex_points(tile, 0.9), tr.tiles[tile], 3, target)
        if marks.flash and marks.flash_strength > 0:
            self._flash(layer, board, marks.flash, marks.flash_strength)
        for e in range(NUM_EDGES):
            owner = state.edge_owner[e]
            if owner >= 0:
                self._road(layer, e, theme.PLAYER[owner])
        for e in marks.edges:
            (ax, ay), (bx, by) = tr.edge_points(e, 0.22)
            w = max(3.0, 0.065 * s)
            layer.line(ax, ay, bx, by, w + 3, theme.with_alpha(theme.PIECE_EDGE, 90))
            layer.line(ax, ay, bx, by, w, theme.with_alpha(theme.TARGET, 110 + 130 * pulse))
        for c in range(NUM_CORNERS):
            owner = state.corner_owner[c]
            if owner >= 0:
                city = state.corner_building[c] == Building.CITY
                self._building(layer, tr.corners[c], theme.PLAYER[owner], city)
        for c, city in marks.corners.items():
            x, y = tr.corners[c]
            if not city:
                layer.circle(x, y, 0.08 * s, theme.with_alpha(theme.TARGET, 50 + 60 * pulse))
            layer.ring(x, y, (0.27 if city else 0.15) * s + 0.03 * s * pulse, 2.5, target)
        robber = state.robber_tile
        if 0 <= robber < NUM_TILES:
            _robber(layer, self.robber_point(board, robber), s)
        if marks.hover is not None:
            self._hover(layer, board, marks)
        return layer

    def _road(self, layer: Layer, edge: int, color: Color, border: bool = True) -> None:
        (ax, ay), (bx, by) = self.transform.edge_points(edge, 0.17)
        w = max(4.0, 0.12 * self.transform.scale)
        if border:
            layer.line(ax, ay, bx, by, w + 3, theme.PIECE_EDGE, True)
        layer.line(ax, ay, bx, by, w, color, True)

    def _building(self, layer: Layer, at: Point, color: Color, city: bool) -> None:
        x, y = at
        s = self.transform.scale
        if city:
            tower = [(-0.2, -0.14), (0.02, -0.14), (0.02, 0.08), (-0.09, 0.21), (-0.2, 0.08)]
            block = [(0.02, -0.14), (0.22, -0.14), (0.22, 0.03), (0.02, 0.03)]
            parts = [[(x + dx * s, y + dy * s) for dx, dy in part] for part in (tower, block)]
            _bordered(layer, parts, (x, y - 0.02 * s), color, 2.0)
            door = theme.mix(color, theme.BLACK, 0.35)
            layer.rect(x + 0.09 * s, y - 0.14 * s, 0.06 * s, 0.09 * s, door)
        else:
            house = [(-0.13, -0.12), (0.13, -0.12), (0.13, 0.03), (0.0, 0.17), (-0.13, 0.03)]
            points = [(x + dx * s, y + dy * s) for dx, dy in house]
            _bordered(layer, [points], (x, y - 0.01 * s), color, 2.0)

    def _flash(self, layer: Layer, board: Board, flash: tuple[str, int], strength: float) -> None:
        tr, s = self.transform, self.transform.scale
        glow = theme.with_alpha(theme.ACCENT, 200 * strength)
        kind, i = flash
        if kind == "corner":
            layer.circle(*tr.corners[i], 0.32 * s, glow)
        elif kind == "edge":
            (ax, ay), (bx, by) = tr.edge_points(i, 0.05)
            layer.line(ax, ay, bx, by, 0.3 * s, glow, True)
        elif kind == "tile":
            layer.outline(tr.hex_points(i, 0.96), tr.tiles[i], 0.12 * s, glow)

    def _hover(self, layer: Layer, board: Board, marks: BoardMarks) -> None:
        assert marks.hover is not None
        tr, s = self.transform, self.transform.scale
        base = theme.PLAYER[marks.player] if marks.player >= 0 else theme.WHITE
        ghost = theme.mix(base, theme.WHITE, 0.4)
        kind, i = marks.hover
        if kind == "corner" and i in marks.corners:
            self._building(layer, tr.corners[i], ghost, marks.corners[i])
            layer.ring(*tr.corners[i], 0.33 * s, 2.5, theme.ACCENT)
        elif kind == "edge" and i in marks.edges:
            self._road(layer, i, ghost, border=False)
            (ax, ay), (bx, by) = tr.edge_points(i, 0.1)
            layer.line(ax, ay, bx, by, 2.0, theme.ACCENT)
        elif kind == "tile" and i in marks.tiles:
            layer.outline(tr.hex_points(i, 0.9), tr.tiles[i], 4, theme.ACCENT)
            _robber(layer, self.robber_point(board, i), s, ghost=True)


# ------------------------------------------------------------------ shapes


def _bordered(
    layer: Layer, parts: Sequence[Sequence[Point]], center: Point, color: Color, border: float
) -> None:
    """Convex `parts` filled with `color` over a dark border about `border` pixels wide."""
    size = max(max(abs(x - center[0]), abs(y - center[1])) for part in parts for x, y in part)
    grow = 1 + border / size if size > 0 else 1
    for part in parts:
        layer.polygon(scaled(part, center, grow), theme.PIECE_EDGE)
    for part in parts:
        layer.polygon(part, color)


def _robber(layer: Layer, at: Point, s: float, ghost: bool = False) -> None:
    x, y = at
    body = theme.mix(theme.ROBBER, theme.WHITE, 0.45) if ghost else theme.ROBBER
    edge = theme.mix(theme.ROBBER_EDGE, theme.WHITE, 0.3) if ghost else theme.ROBBER_EDGE
    if not ghost:
        layer.sector(x, y - 0.17 * s, 0.16 * s, 0, 360, theme.with_alpha(theme.BLACK, 70), 0.05 * s)
    base = [(x - 0.13 * s, y - 0.17 * s), (x + 0.13 * s, y - 0.17 * s)]
    base += [(x + 0.085 * s, y + 0.05 * s), (x - 0.085 * s, y + 0.05 * s)]
    layer.polygon(scaled(base, (x, y - 0.06 * s), 1.16), edge)
    layer.circle(x, y + 0.05 * s, 0.085 * s + 2, edge)
    layer.circle(x, y + 0.16 * s, 0.075 * s + 2, edge)
    layer.polygon(base, body)
    layer.circle(x, y + 0.05 * s, 0.085 * s, body)
    layer.circle(x, y + 0.16 * s, 0.075 * s, body)
    layer.circle(x - 0.025 * s, y + 0.185 * s, 0.022 * s, theme.ROBBER_SHINE)


def _token(layer: Layer, center: Point, number: int, s: float) -> None:
    x, y = center
    r = 0.3 * s
    hot = number in (6, 8)
    ink = theme.HOT if hot else theme.INK
    layer.circle(x + 0.02 * s, y - 0.035 * s, r, theme.with_alpha(theme.BLACK, 55))
    layer.circle(x, y, r, theme.TOKEN)
    layer.ring(x, y, r, max(1.2, 0.022 * s), theme.TOKEN_EDGE)
    size = 0.2 * s * (1.08 if hot else 1.0)
    layer.text(str(number), x, y + 0.05 * s, size, ink, hot, "center", "center")
    pips, gap = PIPS[number], 0.068 * s
    for k in range(pips):
        layer.circle(x + (k - (pips - 1) / 2) * gap, y - 0.155 * s, max(1.3, 0.025 * s), ink, 8)


def _port(layer: Layer, at: Point, kind: int, s: float) -> None:
    x, y = at
    r = PORT_RADIUS * s
    generic = kind == GENERIC_PORT
    fill = theme.PORT if generic else theme.TERRAIN[kind]
    layer.circle(x + 0.02 * s, y - 0.03 * s, r, theme.with_alpha(theme.BLACK, 60))
    layer.circle(x, y, r, fill)
    layer.ring(x, y, r, max(1.5, 0.03 * s), theme.PORT_EDGE)
    if generic:
        layer.text("3:1", x, y, 0.15 * s, theme.INK, True, "center", "center")
        return
    ink = theme.ink_on(kind)
    layer.text(
        theme.RESOURCE_INITIALS[kind], x, y + 0.07 * s, 0.16 * s, ink, True, "center", "center"
    )
    layer.text("2:1", x, y - 0.11 * s, 0.1 * s, ink, False, "center", "center")


# ------------------------------------------------------------------ terrain motifs


def _trees(layer: Layer, at: Point, s: float, flip: int) -> None:
    x, y = at
    color = theme.TERRAIN_DETAIL[Resource.LUMBER]
    for dx, dy in (
        (-0.5, 0.3),
        (-0.2, 0.56),
        (0.22, 0.54),
        (0.5, 0.28),
        (-0.42, -0.5),
        (0.44, -0.5),
    ):
        tx, ty = x + dx * s * flip, y + dy * s
        layer.rect(tx - 0.018 * s, ty - 0.1 * s, 0.036 * s, 0.06 * s, theme.TRUNK)
        layer.triangle(
            (tx - 0.1 * s, ty - 0.05 * s), (tx + 0.1 * s, ty - 0.05 * s), (tx, ty + 0.17 * s), color
        )


def _mounds(layer: Layer, at: Point, s: float, flip: int) -> None:
    x, y = at
    color = theme.TERRAIN_DETAIL[Resource.BRICK]
    for dx, dy, r in (
        (-0.48, 0.26, 0.17),
        (0.1, 0.5, 0.15),
        (0.5, 0.24, 0.13),
        (-0.3, -0.56, 0.15),
        (0.4, -0.5, 0.17),
    ):
        layer.sector(x + dx * s * flip, y + dy * s, r * s, 0, 180, color, r * 0.7 * s)


def _sheep(layer: Layer, at: Point, s: float, flip: int) -> None:
    x, y = at
    wool = theme.TERRAIN_DETAIL[Resource.WOOL]
    for dx, dy in ((-0.48, 0.32), (0.4, 0.42), (-0.46, -0.42), (0.48, -0.36)):
        sx, sy = x + dx * s * flip, y + dy * s
        layer.sector(sx, sy, 0.1 * s, 0, 360, wool, 0.068 * s)
        layer.circle(sx + 0.095 * s * flip, sy + 0.025 * s, 0.04 * s, theme.SHEEP_HEAD)


def _furrows(layer: Layer, at: Point, s: float, flip: int) -> None:
    x, y = at
    color = theme.TERRAIN_DETAIL[Resource.GRAIN]
    width = max(1.5, 0.035 * s)
    for row in (0.5, -0.5):
        for k in range(-2, 3):
            x0 = x + (k * 0.16 - 0.04 * flip) * s
            layer.line(
                x0, y + (row - 0.1) * s, x0 + 0.08 * s * flip, y + (row + 0.1) * s, width, color
            )


def _peaks(layer: Layer, at: Point, s: float, flip: int) -> None:
    x, y = at
    rock = theme.TERRAIN_DETAIL[Resource.ORE]
    for dx, dy, w, h in (
        (-0.42, 0.28, 0.42, 0.36),
        (0.32, 0.36, 0.34, 0.3),
        (0.05, -0.64, 0.3, 0.22),
    ):
        px, py = x + dx * s * flip, y + dy * s
        apex = (px, py + h * s)
        layer.triangle((px - w / 2 * s, py), (px + w / 2 * s, py), apex, rock)
        cap = w / 2 * s * 0.32
        layer.triangle(
            (px - cap, py + h * s * 0.68), (px + cap, py + h * s * 0.68), apex, theme.SNOW
        )


def _dunes(layer: Layer, at: Point, s: float, flip: int) -> None:
    x, y = at
    color = theme.TERRAIN_DETAIL[DESERT]
    for dx, dy, r in (
        (-0.38, 0.38, 0.3),
        (0.32, 0.5, 0.22),
        (0.25, -0.45, 0.32),
        (-0.45, -0.32, 0.2),
    ):
        layer.sector(x + dx * s * flip, y + dy * s, r * s, 0, 180, color, r * 0.35 * s)


MOTIFS: dict[int, Motif] = {
    Resource.LUMBER: _trees,
    Resource.BRICK: _mounds,
    Resource.WOOL: _sheep,
    Resource.GRAIN: _furrows,
    Resource.ORE: _peaks,
    DESERT: _dunes,
}
