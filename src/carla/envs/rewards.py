"""Reward functions (per step, per seat)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from carla.config.schema import EnvConfig
from carla.engine.constants import (
    CITY_COST,
    DESERT,
    DEV_CARD_COST,
    LARGEST_ARMY_MIN,
    LONGEST_ROAD_MIN,
    NUM_PLAYERS,
    NUM_RESOURCES,
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


def expected_production(state: GameState, seat: int, *, robber: bool = True) -> tuple[float, ...]:
    """Raw expected cards per roll by resource, using public board information only."""
    expected = [0.0] * NUM_RESOURCES
    for corner, owner in enumerate(state.corner_owner):
        if owner != seat:
            continue
        # Building's enum value is its production multiplier: settlement=1, city=2.
        multiplier = state.corner_building[corner]
        for tile in CORNER_TILES[corner]:
            resource = state.board.tile_resource[tile]
            if (not robber or tile != state.robber_tile) and resource != DESERT:
                expected[resource] += multiplier * PIPS[state.board.tile_number[tile]] / 36.0
    return tuple(expected)


_DEFAULT_CONFIG = EnvConfig()


def production_strength(state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG) -> float:
    """Normalized expected cards per roll, using only public buildings and tiles."""
    return min(sum(expected_production(state, seat)) / config.production_normalization, 1.0)


def _purchase_readiness(hand: Sequence[int], cost: Sequence[int], available: bool) -> float:
    if not available:
        return 0.0
    deficit = sum(max(required - held, 0) for required, held in zip(cost, hand, strict=True))
    return max(0.0, min(1.0 - deficit / sum(cost), 1.0))


def build_readiness(state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG) -> float:
    road, settlement, city, dev = build_readiness_components(state, seat)
    return (
        config.readiness_road_weight * road
        + config.readiness_settlement_weight * settlement
        + config.readiness_city_weight * city
        + config.readiness_dev_weight * dev
    )


def build_readiness_components(state: GameState, seat: int) -> tuple[float, float, float, float]:
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
    return road, settlement, city, dev


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


def award_progress(state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG) -> float:
    """Public progress toward taking (or retaining) Longest Road and Largest Army."""
    return config.award_road_weight * road_progress(
        state, seat
    ) + config.award_army_weight * army_progress(state, seat)


def player_utility(state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG) -> float:
    """Learner utility; own hidden VP and resource cards are legitimately known."""
    victory = min(victory_points(state, seat, include_hidden=True) / 10.0, 1.0)
    return (
        config.strategic_vp_weight * victory
        + config.strategic_production_weight * production_strength(state, seat, config)
        + config.strategic_readiness_weight * build_readiness(state, seat, config)
        + config.strategic_award_weight * award_progress(state, seat, config)
    )


def public_opponent_strength(
    state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG
) -> float:
    victory = min(victory_points(state, seat, include_hidden=False) / 10.0, 1.0)
    return (
        config.opponent_vp_weight * victory
        + config.opponent_production_weight * production_strength(state, seat, config)
        + config.opponent_award_weight * award_progress(state, seat, config)
    )


def opponent_weights(
    state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG
) -> np.ndarray:
    """Stable softmax of raw public VP / 2, in seat order excluding the learner."""
    logits = np.asarray(
        [
            victory_points(state, opponent, include_hidden=False)
            / config.opponent_softmax_temperature
            for opponent in range(NUM_PLAYERS)
            if opponent != seat
        ],
        dtype=np.float64,
    )
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def opponent_threat(state: GameState, seat: int, config: EnvConfig = _DEFAULT_CONFIG) -> float:
    opponents = (opponent for opponent in range(NUM_PLAYERS) if opponent != seat)
    return float(
        sum(
            weight * public_opponent_strength(state, opponent, config)
            for opponent, weight in zip(
                opponents, opponent_weights(state, seat, config), strict=True
            )
        )
    )


def strategic_potential(game: Game, seat: int, config: EnvConfig) -> float:
    """Stationary potential; no learned beliefs or hidden opponent information."""
    if game.is_over:
        return 0.0
    return config.shaping_coef * (
        player_utility(game.state, seat, config)
        - config.opponent_threat_coef * opponent_threat(game.state, seat, config)
    )


@dataclass(frozen=True, slots=True)
class StrategicV2Components:
    victory: float
    production_quantity: float
    production_diversity: float
    production: float
    road_readiness: float
    settlement_readiness: float
    city_readiness: float
    dev_readiness: float
    readiness: float
    resource_conservation: float
    learner_utility: float
    opponent_strengths: tuple[float, ...]
    opponent_weights: tuple[float, ...]
    opponent_threat: float
    potential: float


def production_diversity(capacity: Sequence[float], threshold: float) -> float:
    return sum(min(resource / threshold, 1.0) for resource in capacity) / NUM_RESOURCES


def resource_conservation(hand: Sequence[int], normalization: float) -> float:
    cards = sum(hand)
    return cards / (cards + normalization)


def strategic_v2_components(game: Game, seat: int, config: EnvConfig) -> StrategicV2Components:
    """Evaluate ESP-V2 once. Opponent tuples follow seat order, excluding the learner."""
    state = game.state
    victory = min(victory_points(state, seat, include_hidden=True) / 10.0, 1.0)
    quantity = production_strength(state, seat, config)
    diversity = production_diversity(
        expected_production(state, seat, robber=False), config.v2_resource_coverage_threshold
    )
    production = (
        config.v2_production_quantity_weight * quantity
        + config.v2_production_diversity_weight * diversity
    )
    road, settlement, city, dev = build_readiness_components(state, seat)
    readiness = (
        config.v2_readiness_road_weight * road
        + config.v2_readiness_settlement_weight * settlement
        + config.v2_readiness_city_weight * city
        + config.v2_readiness_dev_weight * dev
    )
    reserve = resource_conservation(state.hands[seat], config.v2_resource_reserve_normalization)
    utility = (
        config.v2_vp_weight * victory
        + config.v2_production_weight * production
        + config.v2_readiness_weight * readiness
        + config.v2_resource_conservation_weight * reserve
    )
    strengths = tuple(
        config.v2_opponent_vp_weight
        * min(victory_points(state, opponent, include_hidden=False) / 10.0, 1.0)
        + config.v2_opponent_production_weight * production_strength(state, opponent, config)
        for opponent in range(NUM_PLAYERS)
        if opponent != seat
    )
    weights = tuple(opponent_weights(state, seat, config).tolist())
    threat = sum(weight * strength for weight, strength in zip(weights, strengths, strict=True))
    phi = (
        0.0
        if game.is_over
        else config.shaping_coef * (utility - config.opponent_threat_coef * threat)
    )
    return StrategicV2Components(
        victory=victory,
        production_quantity=quantity,
        production_diversity=diversity,
        production=production,
        road_readiness=road,
        settlement_readiness=settlement,
        city_readiness=city,
        dev_readiness=dev,
        readiness=readiness,
        resource_conservation=reserve,
        learner_utility=utility,
        opponent_strengths=strengths,
        opponent_weights=weights,
        opponent_threat=threat,
        potential=phi,
    )


def potential(game: Game, seat: int, config: EnvConfig) -> float:
    """Shaping potential of `seat` (see `EnvConfig.shaping`); 0 when disabled or game over."""
    if config.shaping == "none" or game.is_over:
        return 0.0
    if config.shaping == "strategic":
        return strategic_potential(game, seat, config)
    if config.shaping == "strategic_v2":
        return strategic_v2_components(game, seat, config).potential
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
