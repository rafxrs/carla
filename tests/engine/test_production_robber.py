import pytest

from carla.engine import actions as A
from carla.engine.constants import DevCard, Resource
from carla.engine.rules import ChanceOutcomeError
from carla.engine.state import Phase
from carla.engine.topology import TILE_CORNERS
from tests.engine.scenario import (
    BOARD,
    check_invariants,
    give,
    give_dev,
    post_setup_game,
    put_settlement,
)


def dice(total: int) -> tuple[int, int]:
    first = max(1, total - 6)
    return first, total - first


def unique_tile(number: int) -> int:
    """The only tile carrying `number` (2 and 12 appear once)."""
    return BOARD.tile_number.index(number)


def test_settlement_collects_one_city_two():
    t = unique_tile(12)
    r = BOARD.tile_resource[t]
    game = post_setup_game(rolled=False)
    put_settlement(game, 1, TILE_CORNERS[t][0])
    put_settlement(game, 2, TILE_CORNERS[t][3], city=True)
    events = game.step(A.ROLL, outcome=(6, 6))
    assert game.state.hands[1][r] == 1 and game.state.hands[2][r] == 2
    assert sum(map(sum, game.state.hands)) == 3
    produced = {e.player: e.counts for e in events if e.kind.name == "RESOURCES_PRODUCED"}
    assert produced[1][r] == 1 and produced[2][r] == 2
    assert game.phase == Phase.MAIN
    check_invariants(game)


def test_robber_blocks_production():
    t = unique_tile(2)
    game = post_setup_game(rolled=False)
    put_settlement(game, 1, TILE_CORNERS[t][1])
    s = game.state
    s.robber_tile = t
    game.set_state(s)
    game.step(A.ROLL, outcome=(1, 1))
    assert sum(game.state.hands[1]) == 0


def test_bank_shortage_single_receiver_gets_the_rest():
    t = unique_tile(12)
    r = BOARD.tile_resource[t]
    game = post_setup_game(rolled=False)
    put_settlement(game, 1, TILE_CORNERS[t][0], city=True)
    counts = [0] * 5
    counts[r] = 18
    give(game, 3, counts)  # bank keeps a single card of r
    game.step(A.ROLL, outcome=(6, 6))
    assert game.state.hands[1][r] == 1 and game.state.bank[r] == 0


def test_bank_shortage_several_receivers_get_nothing():
    t = unique_tile(12)
    r = BOARD.tile_resource[t]
    game = post_setup_game(rolled=False)
    put_settlement(game, 1, TILE_CORNERS[t][0])
    put_settlement(game, 2, TILE_CORNERS[t][3])
    counts = [0] * 5
    counts[r] = 18
    give(game, 3, counts)
    game.step(A.ROLL, outcome=(6, 6))
    assert game.state.hands[1][r] == game.state.hands[2][r] == 0 and game.state.bank[r] == 1


def test_seven_makes_large_hands_discard_half_in_turn_order():
    game = post_setup_game(rolled=False)
    give(game, 0, (2, 2, 2, 2, 0))  # 8 cards -> discard 4
    give(game, 1, (7, 0, 0, 0, 0))  # 7 cards -> safe
    give(game, 2, (0, 3, 3, 3, 0))  # 9 cards -> discard 4
    give(game, 3, (0, 0, 0, 0, 1))
    events = game.step(A.ROLL, outcome=(3, 4))
    assert not any(e.kind.name == "RESOURCES_PRODUCED" for e in events)
    assert game.phase == Phase.DISCARD and game.state.discards_pending == [4, 0, 4, 0]
    order = []
    while game.phase == Phase.DISCARD:
        order.append(game.decision_player)
        assert all(a.type == A.ActionType.DISCARD for a in game.legal_actions())
        game.step(game.legal_actions()[0])
        check_invariants(game)
    assert order == [0] * 4 + [2] * 4
    assert [sum(h) for h in game.state.hands] == [4, 7, 5, 1]
    assert game.phase == Phase.MOVE_ROBBER and game.decision_player == 0


def test_robber_moves_and_steals_from_eligible_victims():
    game = post_setup_game(rolled=False)
    t = next(t for t in range(19) if t != game.state.robber_tile)
    put_settlement(game, 1, TILE_CORNERS[t][0])
    put_settlement(game, 2, TILE_CORNERS[t][3])  # no cards: cannot be robbed
    give(game, 1, (0, 0, 3, 0, 0))
    game.step(A.ROLL, outcome=(1, 6))
    legal = game.legal_actions()
    assert {a.tile for a in legal} == set(range(19)) - {game.state.robber_tile}
    assert [a for a in legal if a.tile == t] == [A.ROBBER_TO[t][1 + 1]]
    with pytest.raises(ChanceOutcomeError):
        game.step(A.ROBBER_TO[t][2], outcome=Resource.ORE)
    game = post_setup_game(rolled=False)
    put_settlement(game, 1, TILE_CORNERS[t][0])
    give(game, 1, (0, 0, 3, 0, 0))
    game.step(A.ROLL, outcome=(1, 6))
    events = game.step(A.ROBBER_TO[t][2])
    stolen = [e for e in events if e.kind.name == "RESOURCE_STOLEN"]
    assert stolen and stolen[0].value == Resource.WOOL
    assert stolen[0].visible_to(3).value == -1 and stolen[0].visible_to(1).value == Resource.WOOL
    assert game.state.hands[0][Resource.WOOL] == 1 and game.state.hands[1][Resource.WOOL] == 2
    assert game.state.robber_tile == t and game.phase == Phase.MAIN


def test_robber_without_victims_offers_one_move_per_tile():
    game = post_setup_game(rolled=False)
    game.step(A.ROLL, outcome=(6, 1))
    legal = game.legal_actions()
    assert len(legal) == 18 and all(a.player == -1 for a in legal)
    with pytest.raises(ChanceOutcomeError):
        game.step(legal[0], outcome=Resource.ORE)


def test_knight_before_roll_returns_to_pre_roll():
    game = post_setup_game(rolled=False)
    give_dev(game, 0, DevCard.KNIGHT)
    assert A.PLAY_KNIGHT in game.legal_actions()
    game.step(A.PLAY_KNIGHT)
    assert game.phase == Phase.MOVE_ROBBER
    game.step(game.legal_actions()[0])
    assert game.phase == Phase.PRE_ROLL
    assert game.legal_actions() == (A.ROLL,)
