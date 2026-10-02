"""Whole random games: rule invariants after every step and engine vs reference legality."""

import pytest

from carla.engine.config import EngineConfig
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.simulate import play_random_game
from tests.engine.scenario import (
    check_invariants,
    engine_main_build_actions,
    reference_main_build_actions,
)


def play_checked(seed: int, reference: bool, config: EngineConfig | None = None) -> Game:
    game, rng = Game(config, seed), Rng(seed + 1000)
    while not game.is_over:
        check_invariants(game)
        if reference:
            assert engine_main_build_actions(game) == reference_main_build_actions(game)
        legal = game.legal_actions()
        assert len(set(legal)) == len(legal), "no duplicate actions"
        game.step(rng.choice(legal))
    check_invariants(game)
    return game


@pytest.mark.parametrize("seed", range(10))
def test_random_games_finish_and_keep_invariants(seed):
    game = play_checked(seed, reference=seed < 3)
    assert game.winner is not None and game.victory_points(game.winner) >= 10


def test_every_legal_action_applies_cleanly():
    game, rng = Game(seed=99), Rng(5)
    while not game.is_over:
        if game.num_steps % 37 == 0:
            for action in game.legal_actions():
                trial = game.clone()
                trial.step(action)
                check_invariants(trial)
        game.step(rng.choice(game.legal_actions()))


def test_short_games_with_a_turn_limit():
    for seed in range(5):
        game = play_checked(seed, reference=False, config=EngineConfig(max_turns=20))
        assert game.turn <= 20


@pytest.mark.parametrize("by_type", [True, False])
def test_play_random_game_is_deterministic(by_type):
    a = play_random_game(11, by_type=by_type)
    b = play_random_game(11, by_type=by_type)
    assert a.history == b.history and a.state_hash() == b.state_hash()
    assert play_random_game(12, by_type=by_type).history != a.history


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(100, 300))
def test_many_random_games(seed):
    play_checked(seed, reference=seed % 10 == 0)
