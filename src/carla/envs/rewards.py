"""Reward functions (per step, per seat)."""

import numpy as np

from carla.config.schema import EnvConfig
from carla.engine.constants import NUM_PLAYERS, PIPS
from carla.engine.game import Game
from carla.engine.topology import CORNER_TILES


def public_victory_points(game: Game) -> np.ndarray:
    return np.asarray(
        [game.victory_points(p, include_hidden=False) for p in range(NUM_PLAYERS)], dtype=np.float32
    )


def potential(game: Game, seat: int, config: EnvConfig) -> float:
    """Shaping potential of `seat` (see `EnvConfig.shaping`); 0 when disabled or game over."""
    if config.shaping == "none" or game.is_over:
        return 0.0
    own = game.victory_points(seat, include_hidden=True)
    if config.shaping == "margin":
        own -= max(
            game.victory_points(p, include_hidden=False) for p in range(NUM_PLAYERS) if p != seat
        )
    elif config.shaping == "production":
        own += config.production_weight * expected_production(game, seat)
    return config.shaping_coef * own


def expected_production(game: Game, seat: int) -> float:
    """Expected resources `seat` collects per round (one roll per player): each settlement
    (city) gets one (two) cards per adjacent tile whose number is rolled; the robber's tile and
    the desert produce nothing."""
    s = game.state
    pips = 0
    for corner, owner in enumerate(s.corner_owner):
        if owner != seat:
            continue
        for tile in CORNER_TILES[corner]:
            if tile != s.robber_tile:
                pips += s.corner_building[corner] * PIPS.get(s.board.tile_number[tile], 0)
    return NUM_PLAYERS * pips / 36


def step_rewards(vp_before: np.ndarray | None, game: Game, config: EnvConfig) -> np.ndarray:
    """Rewards of all seats for the step just applied to `game`.

    Terminal: `win_reward` for the winner and `loss_reward` for the others, or
    `truncation_reward` for everyone at the turn limit. Shaping: `vp_reward` per public victory
    point gained (needs `vp_before`, the public points before the step).
    Potential-based shaping between a learner's decisions lives in `CatanEnv` (`potential`).
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
