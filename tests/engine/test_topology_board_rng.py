from collections import Counter

import pytest

from carla.engine import topology as T
from carla.engine.board import Board
from carla.engine.constants import (
    DESERT,
    GENERIC_PORT,
    NUMBER_TOKENS,
    PORT_TYPES,
    TERRAIN_COUNTS,
)
from carla.engine.rng import Rng, derive_seed


def test_graph_sizes_and_euler():
    assert (len(T.TILE_CORNERS), len(T.CORNER_EDGES), len(T.EDGE_CORNERS)) == (19, 54, 72)
    assert 54 - 72 + (19 + 1) == 2  # V - E + F for the planar board graph (outer face included)
    assert Counter(len(t) for t in T.CORNER_TILES) == {3: 24, 2: 12, 1: 18}
    assert Counter(len(n) for n in T.CORNER_NEIGHBORS) == {3: 36, 2: 18}
    assert Counter(len(n) for n in T.TILE_NEIGHBORS) == {6: 7, 4: 6, 3: 6}


def test_adjacency_tables_are_consistent():
    for c in range(T.NUM_CORNERS):
        for e, far in zip(T.CORNER_EDGES[c], T.CORNER_NEIGHBORS[c], strict=True):
            assert set(T.EDGE_CORNERS[e]) == {c, far}
        for t in T.CORNER_TILES[c]:
            assert c in T.TILE_CORNERS[t]
    for t in range(T.NUM_TILES):
        corners, edges = T.TILE_CORNERS[t], T.TILE_EDGES[t]
        assert len(set(corners)) == len(set(edges)) == 6
        for i in range(6):
            assert set(T.EDGE_CORNERS[edges[i]]) == {corners[i], corners[(i + 1) % 6]}
        for u in T.TILE_NEIGHBORS[t]:
            assert t in T.TILE_NEIGHBORS[u]
    for e in range(T.NUM_EDGES):
        for f in T.EDGE_NEIGHBORS[e]:
            assert set(T.EDGE_CORNERS[e]) & set(T.EDGE_CORNERS[f])


def test_graph_is_connected():
    seen, todo = {0}, [0]
    while todo:
        for n in T.CORNER_NEIGHBORS[todo.pop()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    assert len(seen) == T.NUM_CORNERS


def test_coastline_and_ports():
    assert len(T.COAST_EDGES) == len(set(T.COAST_EDGES)) == 30
    for i, e in enumerate(T.COAST_EDGES):  # consecutive coastal edges share a corner: a cycle
        nxt = T.COAST_EDGES[(i + 1) % 30]
        assert set(T.EDGE_CORNERS[e]) & set(T.EDGE_CORNERS[nxt])
    assert set(T.PORT_EDGES) <= set(T.COAST_EDGES)
    port_corners = [c for pair in T.PORT_CORNERS for c in pair]
    assert len(port_corners) == len(set(port_corners)) == 18


@pytest.mark.parametrize("seed", range(20))
def test_random_board_composition(seed):
    board = Board.random(Rng(seed))
    assert Counter(board.tile_resource) == TERRAIN_COUNTS
    assert sorted(n for n in board.tile_number if n) == sorted(NUMBER_TOKENS)
    assert Counter(board.port_types) == Counter(PORT_TYPES)
    assert board.tile_number[board.desert_tile] == 0
    for t in range(T.NUM_TILES):
        if board.tile_number[t] in (6, 8):
            assert all(board.tile_number[u] not in (6, 8) for u in T.TILE_NEIGHBORS[t])
    for kind, (a, b) in zip(board.port_types, T.PORT_CORNERS, strict=True):
        assert board.corner_port[a] == board.corner_port[b] == kind
    assert sum(p >= 0 for p in board.corner_port) == 18
    assert sum(len(ts) for ts in board.tiles_by_number) == 19


def test_board_determinism_and_roundtrip():
    assert Board.random(Rng(3)) == Board.random(Rng(3))
    assert Board.random(Rng(3)) != Board.random(Rng(4))
    board = Board.random(Rng(3))
    assert Board.from_dict(board.to_dict()) == board


def test_board_validation():
    good = Board.random(Rng(0))
    with pytest.raises(ValueError, match="19 tiles"):
        Board(good.tile_resource[:-1], good.tile_number, good.port_types)
    numbers = list(good.tile_number)
    numbers[good.desert_tile] = 5
    with pytest.raises(ValueError, match="invalid tile"):
        Board(good.tile_resource, tuple(numbers), good.port_types)
    with pytest.raises(ValueError, match="port type"):
        Board(good.tile_resource, good.tile_number, (GENERIC_PORT + 1,) * 9)
    no_desert = tuple(0 if r == DESERT else r for r in good.tile_resource)
    with pytest.raises(ValueError, match="invalid tile"):
        Board(no_desert, good.tile_number, good.port_types)


def test_rng_is_reproducible_and_pinned():
    # Pinned values: if these change, recorded seeds no longer reproduce old games.
    rng = Rng(12345)
    assert [rng.randint(1, 6) for _ in range(10)] == [3, 1, 5, 2, 3, 2, 4, 1, 1, 3]
    assert derive_seed(0, "board") == 8322775604716955075
    assert derive_seed(0, "board") != derive_seed(0, "play") != derive_seed(1, "board")
    a, b = Rng(9), Rng(9)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_rng_helpers():
    rng = Rng(1)
    for _ in range(200):
        assert rng.weighted_index([0, 3, 0, 1]) in (1, 3)
        assert 0 <= rng.randbelow(7) < 7
    items = list(range(10))
    rng.shuffle(items)
    assert sorted(items) == list(range(10))
    state = rng.getstate()
    first = [rng.random() for _ in range(3)]
    rng.setstate(state)
    assert [rng.random() for _ in range(3)] == first
    with pytest.raises(ValueError, match="positive"):
        rng.weighted_index([0, 0])
