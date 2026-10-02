import itertools

import pytest

from carla.engine import actions as A
from carla.engine.config import EngineConfig
from carla.engine.constants import DEV_CARD_COST, SETTLEMENT_COST, DevCard
from carla.engine.rng import Rng
from carla.engine.rules import longest_road_length
from carla.engine.topology import (
    COAST_CORNERS,
    COAST_EDGES,
    CORNER_NEIGHBORS,
    EDGE_CORNERS,
    TILE_CORNERS,
    TILE_EDGES,
)
from tests.engine.scenario import (
    check_invariants,
    edge_between,
    give,
    post_setup_game,
    put_road,
    put_settlement,
)


def coast_path(start: int, length: int) -> tuple[list[int], list[int]]:
    """(corners, edges) of a coastal path; corners[i] and [i + 1] are joined by edges[i]."""
    corners = [COAST_CORNERS[(start + i) % 30] for i in range(length + 1)]
    edges = [COAST_EDGES[(start + i) % 30] for i in range(length)]
    return corners, edges


def brute_force_longest(game, player: int) -> int:
    """Longest trail by enumerating edge sequences (independent of the engine's DFS)."""
    s = game.state
    edges = [e for e, o in enumerate(s.edge_owner) if o == player]
    blocked = {c for c, o in enumerate(s.corner_owner) if o not in (-1, player)}
    best = 0
    for n in range(len(edges), 0, -1):
        for seq in itertools.permutations(edges, n):
            for start in EDGE_CORNERS[seq[0]]:
                at, ok = start, True
                for i, e in enumerate(seq):
                    if at not in EDGE_CORNERS[e] or (i > 0 and at in blocked):
                        ok = False
                        break
                    a, b = EDGE_CORNERS[e]
                    at = b if at == a else a
                if ok:
                    return n
    return best


def test_simple_shapes():
    game = post_setup_game()
    _, edges = coast_path(0, 5)
    for e in edges:
        put_road(game, 0, e)
    assert longest_road_length(game.state, 0) == 5
    ring = post_setup_game()
    for e in TILE_EDGES[9]:
        put_road(ring, 1, e)
    assert longest_road_length(ring.state, 1) == 6
    spoke = next(
        edge_between(c, n)
        for c in TILE_CORNERS[9]
        for n in CORNER_NEIGHBORS[c]
        if n not in TILE_CORNERS[9]
    )
    put_road(ring, 1, spoke)
    assert longest_road_length(ring.state, 1) == 7


@pytest.mark.parametrize(("cut_at", "expected"), [(1, 5), (2, 4), (3, 3)])
def test_opponent_settlement_cuts_the_road(cut_at, expected):
    game = post_setup_game()
    corners, edges = coast_path(0, 6)
    for e in edges:
        put_road(game, 0, e)
    put_settlement(game, 1, corners[cut_at])
    assert longest_road_length(game.state, 0) == expected


@pytest.mark.parametrize("seed", range(40))
def test_random_road_networks_match_brute_force(seed):
    rng = Rng(seed)
    game = post_setup_game()
    edges = {rng.randbelow(72)}
    while len(edges) < 2 + rng.randbelow(6):
        e = rng.choice(sorted(edges))
        corner = rng.choice(EDGE_CORNERS[e])
        edges.add(edge_between(corner, rng.choice(CORNER_NEIGHBORS[corner])))
    for e in sorted(edges):
        put_road(game, 0, e)
    for c in sorted({c for e in edges for c in EDGE_CORNERS[e]}):
        if rng.random() < 0.25:
            put_settlement(game, 1 + rng.randbelow(3), c)
    assert longest_road_length(game.state, 0) == brute_force_longest(game, 0)


def degree3_cut_window() -> int:
    """Coast index i such that coastal corner i + 3 has an inland neighbour."""
    return next(i for i in range(30) if len(CORNER_NEIGHBORS[COAST_CORNERS[(i + 3) % 30]]) == 3)


def cut_position(holder: int, others: dict[int, int], cutter: int, holder_len: int = 6):
    """Holder owns a coastal road of `holder_len` with corner 3 cuttable by `cutter`, who has a
    road reaching that corner; `others` maps seats to disjoint road lengths elsewhere."""
    game = post_setup_game()
    start = degree3_cut_window()
    corners, edges = coast_path(start, holder_len)
    for e in edges:
        put_road(game, holder, e)
    cut = corners[3]
    inland = next(n for n in CORNER_NEIGHBORS[cut] if n not in corners)
    put_road(game, cutter, edge_between(inland, cut))
    offset = start + holder_len + 2
    for seat, length in others.items():
        for e in coast_path(offset, length)[1]:
            put_road(game, seat, e)
        offset += length + 2
    assert offset - start <= 30, "paths must not wrap onto each other"
    s = game.state
    lengths = s.road_lengths
    best = max(lengths)
    s.longest_road_owner = holder if lengths[holder] == best else -1
    s.current_player = s.decision_player = cutter
    game.set_state(s)
    give(game, cutter, SETTLEMENT_COST)
    return game, cut


def test_cut_holder_still_tied_keeps_longest_road():
    game, cut = cut_position(holder=2, others={1: 5}, cutter=0, holder_len=8)
    assert game.state.longest_road_owner == 2
    game.step(A.SETTLEMENT_AT[cut])  # holder left with 5 (and 3): tied with seat 1
    assert game.state.road_lengths[2] == 5 and game.state.longest_road_owner == 2
    check_invariants(game)


def test_cut_gives_longest_road_to_unique_leader():
    game, cut = cut_position(holder=2, others={1: 5}, cutter=0)
    events = game.step(A.SETTLEMENT_AT[cut])
    assert game.state.road_lengths[2] == 3
    assert game.state.longest_road_owner == 1
    change = next(e for e in events if e.kind.name == "LONGEST_ROAD_CHANGED")
    assert (change.player, change.target) == (1, 2)


def test_cut_with_tied_challengers_sets_longest_road_aside():
    game, cut = cut_position(holder=2, others={1: 5, 3: 5}, cutter=0)
    game.step(A.SETTLEMENT_AT[cut])
    assert game.state.longest_road_owner == -1
    check_invariants(game)


def test_points_gained_on_another_turn_win_at_the_start_of_your_turn():
    game, cut = cut_position(holder=2, others={1: 5}, cutter=0)
    s = game.state
    s.config = EngineConfig(vp_to_win=3)
    game.set_state(s)
    put_settlement(game, 1, COAST_CORNERS[(degree3_cut_window() + 24) % 30])
    game.step(A.SETTLEMENT_AT[cut])  # seat 1 now holds 3 VP, but it is seat 0's turn
    assert game.victory_points(1) == 3 and not game.is_over
    game.step(A.END_TURN)
    assert game.is_over and game.winner == 1


def test_building_to_the_target_wins_immediately():
    game = post_setup_game(config=EngineConfig(vp_to_win=3))
    corners, edges = coast_path(0, 4)
    put_settlement(game, 0, corners[0])
    put_settlement(game, 0, corners[2])
    put_road(game, 0, edges[2])
    put_road(game, 0, edges[3])
    give(game, 0, SETTLEMENT_COST)
    events = game.step(A.SETTLEMENT_AT[corners[4]])
    assert game.is_over and game.winner == 0 and events[-1].kind.name == "GAME_OVER"
    check_invariants(game)


def test_hidden_victory_point_card_wins_when_bought():
    game = post_setup_game(config=EngineConfig(vp_to_win=3))
    corners, _ = coast_path(0, 4)
    put_settlement(game, 0, corners[0])
    put_settlement(game, 0, corners[2])
    give(game, 0, DEV_CARD_COST)
    game.step(A.BUY_DEV, outcome=DevCard.VICTORY_POINT)
    assert game.winner == 0
