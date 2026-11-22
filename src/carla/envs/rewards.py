"""Reward functions (per step, per seat)."""

from collections.abc import Sequence

import numpy as np

from carla.config.schema import EnvConfig
from carla.engine.constants import (
    CITY_COST,
    DESERT,
    DEV_CARD_COST,
    LARGEST_ARMY_MIN,
    LONGEST_ROAD_MIN,
    NUM_PLAYERS,
    PIPS,
    ROAD_COST,
    SETTLEMENT_COST,
    DevCard,
)
from carla.engine.game import Game
from carla.engine.rules import city_corners, road_edges, settlement_corners, victory_points
from carla.engine.state import GameState
from carla.engine.topology import CORNER_TILES


def public_victory_points(game: Game) -> np.ndarray:
    return np.asarray(
        [game.victory_points(p, include_hidden=False) for p in range(NUM_PLAYERS)], dtype=np.float32
    )


def production_strength(state: GameState, seat: int) -> float:
    """Normalized expected cards per roll, using only public buildings and tiles."""
    expected = 0.0
    for corner, owner in enumerate(state.corner_owner):
        if owner != seat:
            continue
        # Building's enum value is its production multiplier: settlement=1, city=2.
        multiplier = state.corner_building[corner]
        for tile in CORNER_TILES[corner]:
            if tile != state.robber_tile and state.board.tile_resource[tile] != DESERT:
                expected += multiplier * PIPS[state.board.tile_number[tile]] / 36.0
    return min(expected / 2.0, 1.0)


def _purchase_readiness(hand: Sequence[int], cost: Sequence[int], available: bool) -> float:
    if not available:
        return 0.0
    deficit = sum(max(required - held, 0) for required, held in zip(cost, hand, strict=True))
    return max(0.0, min(1.0 - deficit / sum(cost), 1.0))


def build_readiness(state: GameState, seat: int) -> float:
    """Readiness from this seat's known hand and cost-independent normal-play geometry.

    Call only for the learner. Opponent hands must never enter the strategic potential.
    The deck's remaining size, unlike its composition, is public.
    """
    hand = state.hands[seat]
    road = _purchase_readiness(
        hand, ROAD_COST, state.roads_left[seat] > 0 and bool(road_edges(state, seat))
    )
    settlement = _purchase_readiness(
        hand,
        SETTLEMENT_COST,
        state.settlements_left[seat] > 0 and bool(settlement_corners(state, seat)),
    )
    city = _purchase_readiness(
        hand, CITY_COST, state.cities_left[seat] > 0 and bool(city_corners(state, seat))
    )
    dev = _purchase_readiness(hand, DEV_CARD_COST, sum(state.dev_deck) > 0)
    return 0.15 * road + 0.35 * settlement + 0.35 * city + 0.15 * dev


def road_progress(state: GameState, seat: int) -> float:
    owner = state.longest_road_owner
    if owner == seat:
        return 1.0
    required = LONGEST_ROAD_MIN
    if owner >= 0:
        required = max(required, state.road_lengths[owner] + 1)
    return min(state.road_lengths[seat] / required, 1.0)


def army_progress(state: GameState, seat: int) -> float:
    owner = state.largest_army_owner
    if owner == seat:
        return 1.0
    required = LARGEST_ARMY_MIN
    if owner >= 0:
        required = max(required, state.dev_played[owner][DevCard.KNIGHT] + 1)
    return min(state.dev_played[seat][DevCard.KNIGHT] / required, 1.0)


def award_progress(state: GameState, seat: int) -> float:
    """Public progress toward taking (or retaining) Longest Road and Largest Army."""
    return (road_progress(state, seat) + army_progress(state, seat)) / 2.0


def player_utility(state: GameState, seat: int) -> float:
    """Learner utility; own hidden VP and resource cards are legitimately known."""
    victory = min(victory_points(state, seat, include_hidden=True) / 10.0, 1.0)
    return (
        0.50 * victory
        + 0.25 * production_strength(state, seat)
        + 0.10 * build_readiness(state, seat)
        + 0.15 * award_progress(state, seat)
    )


def public_opponent_strength(state: GameState, seat: int) -> float:
    victory = min(victory_points(state, seat, include_hidden=False) / 10.0, 1.0)
    return (
        0.56 * victory
        + 0.28 * production_strength(state, seat)
        + 0.16 * award_progress(state, seat)
    )


def opponent_weights(state: GameState, seat: int) -> np.ndarray:
    """Stable softmax of raw public VP / 2, in seat order excluding the learner."""
    logits = np.asarray(
        [
            victory_points(state, opponent, include_hidden=False) / 2.0
            for opponent in range(NUM_PLAYERS)
            if opponent != seat
        ],
        dtype=np.float64,
    )
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def opponent_threat(state: GameState, seat: int) -> float:
    opponents = (opponent for opponent in range(NUM_PLAYERS) if opponent != seat)
    return float(
        sum(
            weight * public_opponent_strength(state, opponent)
            for opponent, weight in zip(opponents, opponent_weights(state, seat), strict=True)
        )
    )


def strategic_potential(game: Game, seat: int, config: EnvConfig) -> float:
    """Stationary potential; no learned beliefs or hidden opponent information."""
    if game.is_over:
        return 0.0
    return config.shaping_coef * (
        player_utility(game.state, seat) - 0.35 * opponent_threat(game.state, seat)
    )


def potential(game: Game, seat: int, config: EnvConfig) -> float:
    """Shaping potential of `seat` (see `EnvConfig.shaping`); 0 when disabled or game over."""
    if config.shaping == "none" or game.is_over:
        return 0.0
    if config.shaping == "strategic":
        return strategic_potential(game, seat, config)
    own = game.victory_points(seat, include_hidden=True)
    if config.shaping == "margin":
        own -= max(
            game.victory_points(p, include_hidden=False) for p in range(NUM_PLAYERS) if p != seat
        )
    return config.shaping_coef * own


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
