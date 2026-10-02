import json

import pytest

from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.state import GameState
from tests.engine.scenario import finish_setup


def play(game: Game, n: int, seed: int = 0) -> None:
    rng = Rng(seed)
    for _ in range(n):
        if game.is_over:
            return
        game.step(rng.choice(game.legal_actions()))


def test_same_seed_same_game_and_board_seed_option():
    a, b = Game(seed=5), Game(seed=5)
    play(a, 400)
    play(b, 400)
    assert a.history == b.history and a.state_hash() == b.state_hash()
    assert Game(seed=6).board != Game(seed=5).board
    fixed = EngineConfig(board_seed=1)
    assert Game(fixed, seed=5).board == Game(fixed, seed=6).board


def test_explicit_board_does_not_change_the_dice():
    a = Game(seed=3)
    b = Game(seed=3, board=Game(seed=4).board)
    finish_setup(a)
    finish_setup(b)
    assert a.step(A.ROLL)[0].counts == b.step(A.ROLL)[0].counts


def test_legal_actions_are_cached_and_immutable():
    game = Game(seed=0)
    first = game.legal_actions()
    assert isinstance(first, tuple) and game.legal_actions() is first
    game.step(first[0])
    assert game.legal_actions() is not first


def test_snapshot_restore_reproduces_the_future():
    game = Game(seed=8)
    play(game, 300, seed=1)
    snap = game.snapshot()
    play(game, 2000, seed=2)
    end_hash, end_history = game.state_hash(), list(game.history)
    game.restore(snap)
    assert game.num_steps == 300
    play(game, 2000, seed=2)
    assert game.state_hash() == end_hash and game.history == end_history
    game.restore(snap)  # snapshots stay valid after being restored
    assert game.num_steps == 300


def test_clone_is_independent():
    game = Game(seed=8)
    play(game, 200)
    clone = game.clone()
    before = game.state_hash()
    play(clone, 500, seed=3)
    assert game.state_hash() == before and clone.state_hash() != before
    assert game.num_steps == 200


def test_state_copy_is_deep():
    game = Game(seed=1)
    play(game, 200)
    s = game.state
    c = s.copy()
    for name in GameState.__slots__:
        value = getattr(c, name)
        if isinstance(value, list):
            assert value is not getattr(s, name)
            if value and isinstance(value[0], list):
                assert value[0] is not getattr(s, name)[0]


def test_position_json_roundtrip_continues_identically():
    game = Game(seed=13)
    play(game, 250)
    data = json.loads(json.dumps(game.to_dict()))
    restored = Game.from_dict(data)
    assert restored.state_hash() == game.state_hash()
    assert restored.legal_actions() == game.legal_actions()
    play(game, 1000, seed=9)
    play(restored, 1000, seed=9)
    assert restored.state_hash() == game.state_hash()


def test_action_json_roundtrip():
    trade = TradeOffer((1, 0, 0, 0, 0), (0, 0, 0, 1, 0))
    for action in (*A.ALL_ACTIONS, Action(ActionType.PROPOSE_TRADE, trade=trade)):
        assert Action.from_json(json.loads(json.dumps(action.to_json()))) == action
    assert A.ROAD_AT[5].to_json() == {"type": "BUILD_ROAD", "edge": 5}


def test_action_catalogue():
    assert len(A.ALL_ACTIONS) == len(set(A.ALL_ACTIONS)) == len(A.ACTION_INDEX)
    assert A.YEAR_OF_PLENTY[4][1] is A.YEAR_OF_PLENTY[1][4]
    assert all(A.MARITIME[r][r] is None for r in range(5))


def test_forced_outcomes_are_validated():
    game = Game(seed=0)
    finish_setup(game)
    with pytest.raises(ValueError, match="dice"):
        game.step(A.ROLL, outcome=(0, 7))
