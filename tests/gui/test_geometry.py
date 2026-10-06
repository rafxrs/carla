from math import hypot

import pytest

from carla.engine.topology import (
    CORNER_XY,
    EDGE_CORNERS,
    EDGE_XY,
    NUM_CORNERS,
    NUM_EDGES,
    NUM_TILES,
    TILE_CORNERS,
    TILE_XY,
)
from carla.gui.geometry import (
    EXTENT_X,
    EXTENT_Y,
    PORT_XY,
    BoardTransform,
    Rect,
    compute_layout,
    in_hex,
    segment_distance,
)

VIEWPORTS = [Rect(10, 62, 880, 698), Rect(0, 0, 400, 300), Rect(-50, 20, 2000, 900)]


@pytest.fixture(params=VIEWPORTS, ids=["default", "small", "wide"])
def transform(request):
    return BoardTransform(request.param)


def test_world_screen_roundtrip_and_orientation(transform):
    for wx, wy in (*TILE_XY, *CORNER_XY, *EDGE_XY, (0.37, -2.1)):
        sx, sy = transform.to_screen(wx, wy)
        assert transform.to_world(sx, sy) == pytest.approx((wx, wy))
    top = transform.to_screen(*CORNER_XY[0])  # corner 0 is the topmost (smallest y)
    assert all(top[1] >= transform.corners[c][1] for c in range(NUM_CORNERS))  # y flipped up


def test_board_and_ports_fit_inside_the_viewport(transform):
    v, margin = transform.viewport, 0.25 * transform.scale
    for x, y in (*transform.corners, *transform.ports):
        assert v.x + margin <= x <= v.right - margin and v.y + margin <= y <= v.top - margin
    assert transform.scale == pytest.approx(min(v.w / (2 * EXTENT_X), v.h / (2 * EXTENT_Y)))


def test_every_corner_edge_and_tile_hits_itself(transform):
    for c in range(NUM_CORNERS):
        assert transform.corner_at(*transform.corners[c]) == c
    for e in range(NUM_EDGES):
        assert transform.edge_at(*transform.edges[e]) == e
    for t in range(NUM_TILES):
        assert transform.tile_at(*transform.tiles[t]) == t


def test_hits_tolerate_imprecise_clicks(transform):
    s = transform.scale
    for c in range(NUM_CORNERS):
        x, y = transform.corners[c]
        assert transform.corner_at(x + 0.2 * s, y - 0.1 * s) == c
    for e in range(NUM_EDGES):
        a, b = (transform.corners[c] for c in EDGE_CORNERS[e])
        x, y = a[0] + (b[0] - a[0]) * 0.35, a[1] + (b[1] - a[1]) * 0.35  # off-centre, on the edge
        assert transform.edge_at(x, y) == e
    for t in range(NUM_TILES):
        for corner in TILE_CORNERS[t]:  # 80% of the way to each of its corners: still inside
            (tx, ty), (cx, cy) = transform.tiles[t], transform.corners[corner]
            assert transform.tile_at(tx + 0.8 * (cx - tx), ty + 0.8 * (cy - ty)) == t


def test_misses_and_candidate_filtering(transform):
    tile = transform.tiles[9]  # the centre tile: far from every corner and edge
    assert transform.corner_at(*tile) is None and transform.edge_at(*tile) is None
    far = transform.to_screen(EXTENT_X, EXTENT_Y)
    assert transform.tile_at(*far) is None and transform.corner_at(*far) is None
    corner = transform.corners[20]
    assert transform.corner_at(*corner, candidates=[19, 21]) is None
    assert transform.corner_at(*corner, candidates=[20]) == 20
    edge = transform.edges[30]
    assert transform.edge_at(*edge, candidates=[]) is None
    assert transform.tile_at(*transform.tiles[4], candidates=[3, 5]) is None


def test_hex_and_edge_points(transform):
    points = transform.hex_points(9)
    assert points == [transform.corners[c] for c in TILE_CORNERS[9]]
    inner = transform.hex_points(9, inset=0.5)
    cx, cy = transform.tiles[9]
    assert hypot(inner[0][0] - cx, inner[0][1] - cy) == pytest.approx(0.5 * transform.scale)
    a, b = transform.edge_points(5, trim=0.25)
    assert hypot(b[0] - a[0], b[1] - a[1]) == pytest.approx(0.5 * transform.scale)


def test_pure_helpers():
    assert segment_distance((0, 1), (-1, 0), (1, 0)) == pytest.approx(1)
    assert segment_distance((3, 0), (-1, 0), (1, 0)) == pytest.approx(2)
    assert in_hex((0, 0.99), (0, 0)) and not in_hex((0.8, 0.6), (0, 0))
    assert len(PORT_XY) == 9
    for x, y in PORT_XY:  # every marker sits in the sea, off the island
        assert all(not in_hex((x, y), t) for t in TILE_XY)


@pytest.mark.parametrize("size", [(1280, 800), (1000, 660), (1920, 1080), (700, 500)])
def test_layout_partitions_the_window(size):
    w, h = size
    layout = compute_layout(w, h)
    assert layout.board.w + layout.side.w == pytest.approx(w)
    assert layout.side.h + layout.log.h == pytest.approx(h)
    assert layout.log.h > 0 and layout.viewport.h > 0 and layout.viewport.w > 0
    for r in (layout.status, layout.playback, layout.viewport, layout.hand):
        assert r.x >= 0 and r.right <= layout.board.right + 1e-9
        assert r.y >= 0 and r.top <= h + 1e-9
    assert layout.status.top == pytest.approx(h)  # status on top, playback right under it
    assert layout.playback.top == pytest.approx(layout.status.y)
    assert layout.playback.w == pytest.approx(layout.board.w)
    assert layout.hand.y == 0 and layout.hand.w == pytest.approx(layout.board.w)
    assert 84 <= layout.hand.h <= 108  # cards and action tiles need this much
    assert layout.hand.top <= layout.viewport.y + 1e-9
    assert layout.viewport.top <= layout.playback.y + 1e-9
    assert layout.side.x == layout.log.x == layout.board.right


def test_rect_helpers():
    r = Rect(10, 20, 100, 50)
    assert (r.right, r.top, r.cx, r.cy) == (110, 70, 60, 45)
    assert r.contains(10, 20) and r.contains(110, 70) and not r.contains(111, 30)
    top, rest = r.take_top(20)
    assert top == Rect(10, 50, 100, 20) and rest == Rect(10, 20, 100, 30)
    left, right = r.take_left(30)
    assert left == Rect(10, 20, 30, 50) and right == Rect(40, 20, 70, 50)
    cells = r.columns(4, gap=4)
    assert len(cells) == 4 and cells[0].x == 10 and cells[-1].right == pytest.approx(110)
    assert r.inset(5) == Rect(15, 25, 90, 40)
