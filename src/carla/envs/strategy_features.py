"""Planning features (`env.strategic_features`): public information and the observer's cards.

corner_strategy (54, CORNER_STRATEGY_FEATURES): dice probability of each resource around the
    corner (5), free site (distance rule), site the observer can settle now, site one road
    beyond the observer's network.
edge_strategy (72, EDGE_STRATEGY_FEATURES): road the observer can build now, road that ends at a
    free site.
seat_strategy (4, SEAT_STRATEGY_FEATURES) in relative seat order: expected resources per round
    by resource (5), public build statistics (7, `carla.envs.strategy.STAT_NAMES`), port
    access (6), victory points still missing (public points).
own_strategy (OWN_STRATEGY_FEATURES,): per purchase (road, settlement, city, development card)
    whether the observer can afford it, its missing cards and the rounds until expected
    production covers them; then the observer's previous action type (one-hot).

Other seats' hands are never read: their production comes from public buildings, their
statistics from public events.
"""

from collections.abc import Sequence

import numpy as np

from carla.engine.actions import NUM_ACTION_TYPES
from carla.engine.constants import (
    CITY_COST,
    DESERT,
    DEV_CARD_COST,
    NUM_PLAYERS,
    NUM_RESOURCES,
    PIPS,
    ROAD_COST,
    SETTLEMENT_COST,
)
from carla.engine.game import Game
from carla.engine.rules import distance_rule_ok, road_edges, settlement_corners, victory_points
from carla.engine.topology import (
    CORNER_EDGES,
    CORNER_TILES,
    EDGE_CORNERS,
    NUM_CORNERS,
    NUM_EDGES,
)
from carla.envs.strategy import NUM_STATS, StrategyStats

CORNER_STRATEGY_FEATURES = NUM_RESOURCES + 3
EDGE_STRATEGY_FEATURES = 2
SEAT_STRATEGY_FEATURES = NUM_RESOURCES + NUM_STATS + 6 + 1
COSTS = (ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST)
OWN_STRATEGY_FEATURES = 3 * len(COSTS) + NUM_ACTION_TYPES
STRATEGY_FEATURE_KEYS = ("corner_strategy", "edge_strategy", "seat_strategy", "own_strategy")


def production_by_resource(game: Game, seat: int) -> np.ndarray:
    """Expected resources per round (four rolls) of `seat`, by resource; robber tile blocked."""
    s = game.state
    out = np.zeros(NUM_RESOURCES, dtype=np.float32)
    board = s.board
    for corner, owner in enumerate(s.corner_owner):
        if owner != seat:
            continue
        level = s.corner_building[corner]
        for tile in CORNER_TILES[corner]:
            resource = board.tile_resource[tile]
            if resource == DESERT or tile == s.robber_tile:
                continue
            out[resource] += level * PIPS.get(board.tile_number[tile], 0)
    return out * NUM_PLAYERS / 36


def rounds_to_afford(hand: Sequence[int], cost: Sequence[int], rates: np.ndarray) -> float:
    """Rounds until expected production covers the missing cards (0 if affordable)."""
    worst = 0.0
    for r in range(NUM_RESOURCES):
        missing = cost[r] - hand[r]
        if missing > 0:
            worst = max(worst, missing / max(float(rates[r]), 0.05))
    return worst


class StrategyFeatureEncoder:
    """Computes the planning arrays; static per-board parts are cached."""

    def __init__(self) -> None:
        self._board = None
        self._corner_pips: np.ndarray | None = None

    def _static(self, game: Game) -> np.ndarray:
        board = game.board
        if board is not self._board:
            pips = np.zeros((NUM_CORNERS, NUM_RESOURCES), dtype=np.float32)
            for c in range(NUM_CORNERS):
                for t in CORNER_TILES[c]:
                    r = board.tile_resource[t]
                    if r != DESERT:
                        pips[c, r] += PIPS.get(board.tile_number[t], 0) / 36
            self._board, self._corner_pips = board, pips
        assert self._corner_pips is not None
        return self._corner_pips

    def encode(
        self,
        game: Game,
        observer: int,
        stats: Sequence[StrategyStats],
        last_action_type: int = -1,
    ) -> dict[str, np.ndarray]:
        s = game.state
        corners = np.zeros((NUM_CORNERS, CORNER_STRATEGY_FEATURES), dtype=np.float32)
        corners[:, :NUM_RESOURCES] = self._static(game) * 2  # at most ~0.42 per tile
        free = np.asarray(
            [s.corner_owner[c] < 0 and distance_rule_ok(s, c) for c in range(NUM_CORNERS)],
            dtype=bool,
        )
        corners[:, NUM_RESOURCES] = free
        setup = s.setup_step < 8
        if not setup:
            for c in settlement_corners(s, observer):
                corners[c, NUM_RESOURCES + 1] = 1.0
        network = np.zeros(NUM_CORNERS, dtype=bool)
        for e, owner in enumerate(s.edge_owner):
            if owner == observer:
                network[list(EDGE_CORNERS[e])] = True
        network |= np.asarray(s.corner_owner) == observer
        edges = np.zeros((NUM_EDGES, EDGE_STRATEGY_FEATURES), dtype=np.float32)
        buildable = road_edges(s, observer) if not setup and s.roads_left[observer] else []
        for e in buildable:
            edges[e, 0] = 1.0
        for e in range(NUM_EDGES):
            if s.edge_owner[e] >= 0:
                continue
            a, b = EDGE_CORNERS[e]
            ends_free = (network[a] and free[b] and not network[b]) or (
                network[b] and free[a] and not network[a]
            )
            if ends_free:
                edges[e, 1] = 1.0
                far = b if network[a] else a
                corners[far, NUM_RESOURCES + 2] = 1.0
        # A free corner already on the network needs no road; mark sites beyond as well.
        for c in np.flatnonzero(network):
            for e in CORNER_EDGES[c]:
                if s.edge_owner[e] >= 0:
                    continue
                other = EDGE_CORNERS[e][0] if EDGE_CORNERS[e][1] == c else EDGE_CORNERS[e][1]
                if free[other]:
                    corners[other, NUM_RESOURCES + 2] = 1.0

        seats = np.zeros((NUM_PLAYERS, SEAT_STRATEGY_FEATURES), dtype=np.float32)
        rates = []
        for p in range(NUM_PLAYERS):
            rel = (p - observer) % NUM_PLAYERS
            rate = production_by_resource(game, p)
            rates.append(rate)
            row = seats[rel]
            row[:NUM_RESOURCES] = rate / 3
            row[NUM_RESOURCES : NUM_RESOURCES + NUM_STATS] = stats[p].normalized()
            i = NUM_RESOURCES + NUM_STATS
            for c, owner in enumerate(s.corner_owner):
                if owner == p and (port := s.board.corner_port[c]) >= 0:
                    row[i + port] = 1.0
            missing = s.config.vp_to_win - victory_points(s, p, include_hidden=False)
            row[i + 6] = max(0, missing) / s.config.vp_to_win

        own = np.zeros(OWN_STRATEGY_FEATURES, dtype=np.float32)
        hand = s.hands[observer]
        for k, cost in enumerate(COSTS):
            missing = sum(max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES))
            own[3 * k] = float(missing == 0)
            own[3 * k + 1] = missing / 5
            own[3 * k + 2] = min(rounds_to_afford(hand, cost, rates[observer]), 10.0) / 10
        if last_action_type >= 0:
            own[3 * len(COSTS) + last_action_type] = 1.0
        return {
            "corner_strategy": corners,
            "edge_strategy": edges,
            "seat_strategy": seats,
            "own_strategy": own,
        }
