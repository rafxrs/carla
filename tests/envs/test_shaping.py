from typing import Literal

import numpy as np
import pytest

from carla.config.schema import EnvConfig
from carla.engine.config import EngineConfig
from carla.engine.constants import PIPS
from carla.engine.topology import CORNER_TILES
from carla.envs import CatanEnv
from carla.envs.rewards import expected_production, potential
from tests.engine.scenario import post_setup_game, put_settlement


def _play(
    shaping: Literal["none", "own", "margin", "production"], seed: int = 3, gamma: float = 0.999
):
    env = CatanEnv(
        EngineConfig(max_turns=200),
        EnvConfig(learner_seat=0, shaping=shaping, shaping_coef=0.1, shaping_gamma=gamma),
    )
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards, potentials, done = [], [potential(env.game, 0, env.config)], False
    while not done:
        env.action_space.seed(int(rng.integers(2**31)))
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        rewards.append(reward)
        potentials.append(potential(env.game, 0, env.config))
        done = terminated or truncated
    return env, np.asarray(rewards), np.asarray(potentials)


@pytest.mark.parametrize("shaping", ["own", "margin", "production"])
def test_shaping_is_gamma_potential_difference_and_zero_at_the_end(
    shaping: Literal["own", "margin", "production"],
):
    _, shaped, phi = _play(shaping)
    _, sparse, _ = _play("none")
    assert phi[-1] == 0.0 and np.any(phi[:-1] != 0)
    np.testing.assert_allclose(shaped - sparse, 0.999 * phi[1:] - phi[:-1], atol=1e-5)


def test_undiscounted_shaping_cancels_over_an_episode():
    _, shaped, phi = _play("own", gamma=1.0)
    _, sparse, _ = _play("none")
    # Phi(start) is 0 before anyone builds, and Phi(end) is 0 by definition.
    assert phi[0] == 0.0
    assert shaped.sum() == pytest.approx(sparse.sum(), abs=1e-5)


def test_margin_potential_uses_the_top_opponent():
    env, _, _ = _play("margin")
    env.reset(seed=5)
    for _ in range(40):
        env.step(env.action_space.sample())
    game = env.game
    if game.is_over:
        pytest.skip("game ended early")
    top = max(game.victory_points(p, include_hidden=False) for p in (1, 2, 3))
    assert potential(game, 0, env.config) == pytest.approx(
        0.1 * (game.victory_points(0, include_hidden=True) - top)
    )


def test_undiscounted_production_shaping_cancels_over_an_episode():
    _, shaped, phi = _play("production", gamma=1.0)
    _, sparse, _ = _play("none")
    assert phi[-1] == 0.0
    # The learner's first decision may already see opponents' setup settlements, not its own.
    assert shaped.sum() == pytest.approx(sparse.sum() - phi[0], abs=1e-5)


def test_production_counts_cities_twice_and_skips_the_robber():
    game = post_setup_game()
    state = game.state
    assert expected_production(game, 0) == 0.0
    put_settlement(game, 0, 10)
    put_settlement(game, 0, 40)
    put_settlement(game, 1, 20)
    own = [10, 40]

    def manual(robber: int) -> float:
        pips = sum(
            state.corner_building[c] * PIPS.get(state.board.tile_number[t], 0)
            for c in own
            for t in CORNER_TILES[c]
            if t != robber
        )
        return 4 * pips / 36

    assert expected_production(game, 0) == pytest.approx(manual(state.robber_tile))
    settlement = expected_production(game, 0)
    state.corner_building[own[0]] = 2  # Upgrade to a city.
    assert expected_production(game, 0) > settlement
    productive = next(
        t for t in CORNER_TILES[own[0]] if PIPS.get(state.board.tile_number[t], 0) > 0
    )
    state.robber_tile = productive
    assert expected_production(game, 0) == pytest.approx(manual(productive))
    config = EnvConfig(shaping="production", shaping_coef=0.1, production_weight=0.5)
    assert potential(game, 0, config) == pytest.approx(
        0.1 * (game.victory_points(0, include_hidden=True) + 0.5 * manual(productive))
    )
