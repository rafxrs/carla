"""Static board graph shared by every game: 19 tiles, 54 corners, 72 edges, 9 port slots.

Ids are assigned once, top-to-bottom then left-to-right (see docs/img/board.png). Hexes are
pointy-top in axial coordinates (q, r) with radius 2. Geometry (`*_XY`) is only for rendering;
models must use the adjacency tables, which are invariant to board rotation/reflection.
"""

from itertools import pairwise
from math import atan2, sqrt, tau

NUM_TILES = 19
NUM_CORNERS = 54
NUM_EDGES = 72
NUM_PORTS = 9

# Corner offsets from a tile centre on an integer lattice where x is in units of sqrt(3)/2 and
# y in units of 1/2 (y grows downwards). Order: clockwise starting at the top corner.
_CORNER_OFFSETS = ((0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1))

# Number of coastal edges skipped between consecutive ports, walking clockwise around the
# 30-edge coastline. Gives the 3-fold symmetric frame used by the physical game.
_PORT_GAPS = (2, 2, 3, 2, 2, 3, 2, 2, 3)


def _build():
    axial = [(q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2]
    centers = [(2 * q + r, 3 * r) for q, r in axial]

    lattice = sorted(
        {(cx + dx, cy + dy) for cx, cy in centers for dx, dy in _CORNER_OFFSETS},
        key=lambda p: (p[1], p[0]),
    )
    corner_id = {p: i for i, p in enumerate(lattice)}
    tile_corners = [
        tuple(corner_id[(cx + dx, cy + dy)] for dx, dy in _CORNER_OFFSETS) for cx, cy in centers
    ]

    pairs = {tuple(sorted((tc[i], tc[(i + 1) % 6]))) for tc in tile_corners for i in range(6)}

    def midpoint(pair):
        (ax, ay), (bx, by) = lattice[pair[0]], lattice[pair[1]]
        return (ay + by, ax + bx)

    edge_corners = sorted(pairs, key=midpoint)
    edge_id = {pair: i for i, pair in enumerate(edge_corners)}
    tile_edges = [
        tuple(edge_id[tuple(sorted((tc[i], tc[(i + 1) % 6])))] for i in range(6))
        for tc in tile_corners
    ]

    corner_tiles = [[] for _ in lattice]
    for t, tc in enumerate(tile_corners):
        for c in tc:
            corner_tiles[c].append(t)
    corner_edges = [[] for _ in lattice]
    corner_neighbors = [[] for _ in lattice]
    for e, (a, b) in enumerate(edge_corners):
        corner_edges[a].append(e)
        corner_edges[b].append(e)
        corner_neighbors[a].append(b)
        corner_neighbors[b].append(a)
    edge_neighbors = [
        tuple(sorted({f for c in pair for f in corner_edges[c]} - {e}))
        for e, pair in enumerate(edge_corners)
    ]
    tile_neighbors = [
        tuple(u for u in range(len(axial)) if u != t and len(set(tc) & set(tile_corners[u])) == 2)
        for t, tc in enumerate(tile_corners)
    ]

    # Coastline: edges on exactly one tile, walked clockwise from the top-left corner.
    edge_tile_count = [0] * len(edge_corners)
    for te in tile_edges:
        for e in te:
            edge_tile_count[e] += 1
    coastal = {e for e, n in enumerate(edge_tile_count) if n == 1}
    start = min(c for e in coastal for c in edge_corners[e])
    first = max(
        (e for e in corner_edges[start] if e in coastal),
        key=lambda e: lattice[sum(edge_corners[e]) - start][0],
    )
    coast_edges, coast_corners = [first], [start]
    corner = sum(edge_corners[first]) - start
    while corner != start:
        coast_corners.append(corner)
        nxt = next(e for e in corner_edges[corner] if e in coastal and e != coast_edges[-1])
        coast_edges.append(nxt)
        corner = sum(edge_corners[nxt]) - corner

    port_positions, pos = [], 0
    for gap in _PORT_GAPS:
        port_positions.append(pos)
        pos += gap + 1
    port_edges = tuple(coast_edges[i] for i in port_positions)

    return {
        "TILE_AXIAL": tuple(axial),
        "TILE_CORNERS": tuple(tile_corners),
        "TILE_EDGES": tuple(tile_edges),
        "TILE_NEIGHBORS": tuple(tile_neighbors),
        "CORNER_TILES": tuple(tuple(x) for x in corner_tiles),
        "CORNER_EDGES": tuple(tuple(x) for x in corner_edges),
        "CORNER_NEIGHBORS": tuple(tuple(x) for x in corner_neighbors),
        "EDGE_CORNERS": tuple(edge_corners),
        "EDGE_NEIGHBORS": tuple(edge_neighbors),
        "COAST_EDGES": tuple(coast_edges),
        "COAST_CORNERS": tuple(coast_corners),
        "PORT_EDGES": port_edges,
        "LATTICE": tuple(lattice),
    }


_T = _build()

TILE_AXIAL: tuple[tuple[int, int], ...] = _T["TILE_AXIAL"]
TILE_CORNERS: tuple[tuple[int, ...], ...] = _T["TILE_CORNERS"]
"""Six corners of each tile, clockwise from the top."""
TILE_EDGES: tuple[tuple[int, ...], ...] = _T["TILE_EDGES"]
"""Six edges of each tile; TILE_EDGES[t][i] joins TILE_CORNERS[t][i] and [i + 1]."""
TILE_NEIGHBORS: tuple[tuple[int, ...], ...] = _T["TILE_NEIGHBORS"]
CORNER_TILES: tuple[tuple[int, ...], ...] = _T["CORNER_TILES"]
CORNER_EDGES: tuple[tuple[int, ...], ...] = _T["CORNER_EDGES"]
CORNER_NEIGHBORS: tuple[tuple[int, ...], ...] = _T["CORNER_NEIGHBORS"]
"""Adjacent corners; CORNER_NEIGHBORS[c][i] is the far end of CORNER_EDGES[c][i]."""
EDGE_CORNERS: tuple[tuple[int, int], ...] = _T["EDGE_CORNERS"]
EDGE_NEIGHBORS: tuple[tuple[int, ...], ...] = _T["EDGE_NEIGHBORS"]
COAST_EDGES: tuple[int, ...] = _T["COAST_EDGES"]
"""The 30 coastal edges in clockwise order."""
COAST_CORNERS: tuple[int, ...] = _T["COAST_CORNERS"]
PORT_EDGES: tuple[int, ...] = _T["PORT_EDGES"]
"""Coastal edge of each of the 9 port slots."""
PORT_CORNERS: tuple[tuple[int, int], ...] = tuple(EDGE_CORNERS[e] for e in PORT_EDGES)
CORNER_LINKS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple(zip(CORNER_EDGES[c], CORNER_NEIGHBORS[c], strict=True)) for c in range(NUM_CORNERS)
)
"""(edge, far corner) pairs leaving each corner."""

_LATTICE = _T["LATTICE"]
_HALF_SQRT3 = sqrt(3) / 2
CORNER_XY: tuple[tuple[float, float], ...] = tuple((x * _HALF_SQRT3, y / 2) for x, y in _LATTICE)
"""Rendering coordinates (hex size 1, y downwards)."""
TILE_XY: tuple[tuple[float, float], ...] = tuple(
    (sqrt(3) * (q + r / 2), 1.5 * r) for q, r in TILE_AXIAL
)
EDGE_XY: tuple[tuple[float, float], ...] = tuple(
    ((CORNER_XY[a][0] + CORNER_XY[b][0]) / 2, (CORNER_XY[a][1] + CORNER_XY[b][1]) / 2)
    for a, b in EDGE_CORNERS
)


def _spirals() -> tuple[tuple[int, ...], ...]:
    def ring(t: int) -> int:
        q, r = TILE_AXIAL[t]
        return max(abs(q), abs(r), abs(q + r))

    def around(tiles: list[int], start: int) -> list[int]:
        # Counterclockwise as drawn (y up), beginning with `start`.
        def turn(t: int) -> float:
            (x, y), (sx, sy) = TILE_XY[t], TILE_XY[start]
            return (atan2(-y, x) - atan2(-sy, sx)) % tau

        return sorted(tiles, key=turn)

    coast = [t for t in range(NUM_TILES) if ring(t) == 2]
    inner = [t for t in range(NUM_TILES) if ring(t) == 1]
    centre = TILE_AXIAL.index((0, 0))
    spirals = []
    for corner in (t for t in coast if len(TILE_NEIGHBORS[t]) == 3):
        next_to_corner = next(t for t in TILE_NEIGHBORS[corner] if ring(t) == 1)
        spirals.append((*around(coast, corner), *around(inner, next_to_corner), centre))
    return tuple(spirals)


TILE_SPIRALS: tuple[tuple[int, ...], ...] = _spirals()
"""Order in which the official variable set-up lays the number tokens, one spiral per corner
tile of the coast: counterclockwise round the coast, round the inner ring from the tile next to
that corner, then the centre."""

assert (len(TILE_AXIAL), len(CORNER_XY), len(EDGE_CORNERS)) == (19, 54, 72)
assert len(COAST_EDGES) == 30 and len(PORT_EDGES) == NUM_PORTS
assert all(b in TILE_NEIGHBORS[a] for s in TILE_SPIRALS for a, b in pairwise(s))
del _T
