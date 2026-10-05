"""Reward functions (per step, per seat)."""

import numpy as np

from carla.config.schema import EnvConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.game import Game


def public_victory_points(game: Game) -> np.ndarray:
    return np.asarray(
        [game.victory_points(p, include_hidden=False) for p in range(NUM_PLAYERS)], dtype=np.float32
    )


def step_rewards(vp_before: np.ndarray | None, game: Game, config: EnvConfig) -> np.ndarray:
    """Rewards of all seats for the step just applied to `game`.

    Terminal: `win_reward` for the winner and `loss_reward` for the others, or
    `truncation_reward` for everyone at the turn limit. Shaping: `vp_reward` per public victory
    point gained (needs `vp_before`, the public points before the step).
    TODO(rl): try potential-based shaping (e.g. on production pips) and compare in ablations.
    """
    rewards = np.zeros(NUM_PLAYERS, dtype=np.float32)
    if config.vp_reward and vp_before is not None:
        rewards += config.vp_reward * (public_victory_points(game) - vp_before)
    if game.is_over:
        if game.winner is None:
            rewards += config.truncation_reward
        else:
            rewards += config.loss_reward
            rewards[game.winner] += config.win_reward - config.loss_reward
    return rewards
