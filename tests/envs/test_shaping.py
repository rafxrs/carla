from typing import Literal

import numpy as np
import pytest

from carla.config.schema import EnvConfig
from carla.engine.config import EngineConfig
from carla.envs import CatanEnv
from carla.envs.rewards import potential


def _play(shaping: Literal["none", "own", "margin"], seed: int = 3, gamma: float = 0.999):
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


@pytest.mark.parametrize("shaping", ["own", "margin"])
def test_shaping_is_gamma_potential_difference_and_zero_at_the_end(
    shaping: Literal["own", "margin"],
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
