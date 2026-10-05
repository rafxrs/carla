import pytest

from carla.engine import actions as A
from carla.engine.constants import (
    CITY_COST,
    GENERIC_PORT,
    ROAD_COST,
    SETTLEMENT_COST,
    Building,
    Resource,
)
from carla.engine.rules import trade_ratios
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS, PORT_CORNERS
from tests.engine.scenario import (
    BOARD,
    check_invariants,
    edge_between,
    engine_main_build_actions,
    give,
    post_setup_game,
    put_road,
    put_settlement,
    reference_main_build_actions,
)


def branch_corner() -> tuple[int, int, int]:
    """Corners a - b - x along two edges, all of degree 3 (room for opponents)."""
    for a in range(54):
        for b in CORNER_NEIGHBORS[a]:
            for x in CORNER_NEIGHBORS[b]:
                if (
                    x != a
                    and x not in CORNER_NEIGHBORS[a]
                    and len(CORNER_NEIGHBORS[a]) == len(CORNER_NEIGHBORS[b]) == 3
                    and len(CORNER_NEIGHBORS[x]) == 3
                ):
                    return a, b, x
    raise AssertionError


def test_building_pays_the_bank():
    game = post_setup_game()
    a, b, _ = branch_corner()
    put_settlement(game, 0, a)
    give(game, 0, ROAD_COST)
    road = A.ROAD_AT[edge_between(a, b)]
    assert road in game.legal_actions()
    game.step(road)
    assert game.state.hands[0] == [0] * 5 and game.state.bank == [19] * 5
    assert game.state.edge_owner[road.edge] == 0 and game.state.roads_left[0] == 14
    check_invariants(game)


def test_roads_and_settlements_follow_connection_and_distance_rules():
    game = post_setup_game()
    a, b, x = branch_corner()
    put_settlement(game, 0, a)
    put_road(game, 0, edge_between(a, b))
    put_road(game, 0, edge_between(b, x))
    give(game, 0, (2, 2, 1, 1, 0))
    legal = game.legal_actions()
    settlements = {act.corner for act in legal if act.type == A.ActionType.BUILD_SETTLEMENT}
    assert settlements == {x}  # b is next to a
    roads = {act.edge for act in legal if act.type == A.ActionType.BUILD_ROAD}
    own = {e for e, o in enumerate(game.state.edge_owner) if o == 0}
    assert roads == {e for c in (a, b, x) for e in CORNER_EDGES[c]} - own
    assert engine_main_build_actions(game) == reference_main_build_actions(game)

    put_settlement(game, 1, x)  # opponent settles at the end of our road: no road through x
    roads = {act.edge for act in game.legal_actions() if act.type == A.ActionType.BUILD_ROAD}
    assert not roads & (set(CORNER_EDGES[x]) - {edge_between(b, x)})
    assert engine_main_build_actions(game) == reference_main_build_actions(game)


def test_city_upgrade():
    game = post_setup_game()
    a, _, _ = branch_corner()
    put_settlement(game, 0, a)
    give(game, 0, CITY_COST)
    assert A.CITY_AT[a] in game.legal_actions()
    game.step(A.CITY_AT[a])
    s = game.state
    assert s.corner_building[a] == Building.CITY
    assert (s.settlements_left[0], s.cities_left[0]) == (5, 3)
    assert game.victory_points(0) == 2
    check_invariants(game)


def test_piece_supply_limits_building():
    game = post_setup_game()
    a, b, x = branch_corner()
    put_settlement(game, 0, a)
    put_road(game, 0, edge_between(a, b))
    put_road(game, 0, edge_between(b, x))
    give(game, 0, SETTLEMENT_COST)
    assert A.SETTLEMENT_AT[x] in game.legal_actions()
    s = game.state
    s.settlements_left[0] = 0
    game.set_state(s)
    assert A.SETTLEMENT_AT[x] not in game.legal_actions()


def maritime_options(game) -> set[tuple[int, int]]:
    return {
        (a.resource, a.resource2)
        for a in game.legal_actions()
        if a.type == A.ActionType.MARITIME_TRADE
    }


def test_maritime_trade_default_ratio_is_four():
    game = post_setup_game()
    give(game, 0, (3, 0, 0, 0, 0))
    assert maritime_options(game) == set()
    give(game, 0, (1, 0, 0, 0, 0))
    assert maritime_options(game) == {(0, r) for r in range(1, 5)}
    trade = A.MARITIME[0][4]
    assert trade is not None
    game.step(trade)
    assert game.state.hands[0] == [0, 0, 0, 0, 1]
    assert game.state.bank == [19, 19, 19, 19, 18]


@pytest.mark.parametrize("slot", range(9))
def test_ports_lower_the_ratio(slot):
    game = post_setup_game()
    kind = BOARD.port_types[slot]
    put_settlement(game, 0, PORT_CORNERS[slot][0])
    ratios = trade_ratios(game.state, 0)
    if kind == GENERIC_PORT:
        assert ratios == [3] * 5
    else:
        assert ratios[kind] == 2 and all(ratios[r] == 4 for r in range(5) if r != kind)
    give(game, 0, (3, 3, 3, 3, 3))
    assert engine_main_build_actions(game) == reference_main_build_actions(game)


def test_maritime_trade_needs_the_bank_to_have_the_card():
    game = post_setup_game()
    give(game, 0, (4, 0, 0, 0, 0))
    give(game, 1, (0, 0, 0, 0, 19))
    assert (Resource.LUMBER, Resource.ORE) not in maritime_options(game)
