import pytest

from carla.engine import actions as A
from carla.engine.config import EngineConfig
from carla.engine.constants import DESERT
from carla.engine.game import Game, IllegalActionError
from carla.engine.state import Phase
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS, CORNER_TILES
from tests.engine.scenario import check_invariants, finish_setup


def test_snake_order_and_starting_resources():
    game = Game(seed=1)
    actors, second_settlements = [], {}
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        actors.append(game.decision_player)
        action = game.legal_actions()[-1]
        if action.type == A.ActionType.BUILD_SETTLEMENT and game.state.setup_step >= 4:
            second_settlements[game.decision_player] = action.corner
        game.step(action)
    assert actors == [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0]
    assert game.phase == Phase.PRE_ROLL and game.current_player == 0 and game.turn == 0
    s = game.state
    for p in range(4):
        assert s.settlements_left[p] == 3 and s.roads_left[p] == 13
        expected = [0] * 5
        for t in CORNER_TILES[second_settlements[p]]:
            if s.board.tile_resource[t] != DESERT:
                expected[s.board.tile_resource[t]] += 1
        assert s.hands[p] == expected
        assert game.victory_points(p) == 2
    check_invariants(game)


def test_setup_road_touches_new_settlement():
    game = Game(seed=2)
    settlement = game.legal_actions()[5]
    game.step(settlement)
    assert game.phase == Phase.SETUP_ROAD
    assert {a.edge for a in game.legal_actions()} == set(CORNER_EDGES[settlement.corner])


def test_setup_distance_rule():
    game = Game(seed=2)
    first = game.legal_actions()[0]
    game.step(first)
    game.step(game.legal_actions()[0])
    corners = {a.corner for a in game.legal_actions()}
    blocked = {first.corner, *CORNER_NEIGHBORS[first.corner]}
    assert not corners & blocked
    assert len(corners) == 54 - len(blocked)


def test_illegal_actions_are_rejected():
    game = Game(seed=0)
    with pytest.raises(IllegalActionError):
        game.step(A.ROLL)
    finish_setup(game)
    with pytest.raises(IllegalActionError):
        game.step(A.END_TURN)  # must roll first
    game.step(A.ROLL, outcome=(1, 2))
    with pytest.raises(ValueError, match="no chance outcome"):
        game.step(A.END_TURN, outcome=(1, 1))


def test_turns_rotate():
    game = Game(seed=0)
    finish_setup(game)
    for expected in (0, 1, 2, 3, 0):
        assert game.current_player == game.decision_player == expected
        game.step(A.ROLL, outcome=(2, 3))
        assert game.phase == Phase.MAIN
        game.step(A.END_TURN)
    assert game.turn == 5 and game.phase == Phase.PRE_ROLL


def test_max_turns_truncates_without_winner():
    game = Game(EngineConfig(max_turns=6), seed=0)
    finish_setup(game)
    events = []
    for _ in range(6):
        game.step(A.ROLL, outcome=(2, 3))
        events = game.step(A.END_TURN)
    assert game.is_over and game.is_truncated and game.winner is None
    assert events[-1].kind.name == "GAME_OVER" and events[-1].player == -1
    assert game.legal_actions() == ()
    with pytest.raises(IllegalActionError):
        game.step(A.ROLL)


def test_domestic_trade_can_be_enabled():
    game = Game(EngineConfig(allow_domestic_trade=True))
    assert game.config.max_trade_proposals_per_turn == 3
    assert game.config.max_domestic_trades_per_turn == 3
    assert game.config.max_trade_cards_per_side == 4
