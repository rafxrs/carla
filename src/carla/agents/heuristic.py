"""Greedy rule-based bot (v0 baseline for evaluation and imitation-learning data).

TODO(eval): tune weights, add a lookahead "value-function" variant, smarter robber/monopoly use.
"""

from collections.abc import Sequence

from carla.agents.base import Agent
from carla.engine.actions import Action, ActionType
from carla.engine.board import Board
from carla.engine.constants import (
    CITY_COST,
    DESERT,
    DEV_CARD_COST,
    GENERIC_PORT,
    NUM_PLAYERS,
    NUM_RESOURCES,
    PIPS,
    ROAD_COST,
    SETTLEMENT_COST,
    DevCard,
)
from carla.engine.game import Game
from carla.engine.rules import (
    city_corners,
    distance_rule_ok,
    road_edges,
    settlement_corners,
    trade_ratios,
    victory_points,
)
from carla.engine.state import GameState, Phase
from carla.engine.topology import CORNER_NEIGHBORS, CORNER_TILES, EDGE_CORNERS, TILE_CORNERS

RESOURCE_WEIGHTS = (1.0, 1.0, 0.8, 1.1, 1.1)  # lumber, brick, wool, grain, ore
_T = ActionType


def corner_value(board: Board, corner: int, robber_tile: int = -1) -> float:
    """Production value of a corner: weighted pips, resource diversity and port bonus."""
    value, kinds = 0.0, set()
    for t in CORNER_TILES[corner]:
        r = board.tile_resource[t]
        if r == DESERT:
            continue
        pips = PIPS[board.tile_number[t]] * (0.5 if t == robber_tile else 1.0)
        value += pips * RESOURCE_WEIGHTS[r]
        kinds.add(r)
    port = board.corner_port[corner]
    return value + 1.5 * len(kinds) + (0.0 if port < 0 else 1.0 if port == GENERIC_PORT else 1.5)


class HeuristicAgent(Agent):
    """Builds whenever possible (city > settlement > road toward a site > development card),
    trades with the bank toward its next build, robs the leader. Uses only public information
    and its own cards."""

    name = "heuristic"

    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        s = game.state
        by_type: dict[int, list[Action]] = {}
        for a in legal:
            by_type.setdefault(a.type, []).append(a)

        if s.phase in (Phase.SETUP_SETTLEMENT, Phase.MAIN) and _T.BUILD_SETTLEMENT in by_type:
            return max(by_type[_T.BUILD_SETTLEMENT], key=lambda a: corner_value(s.board, a.corner))
        if s.phase in (Phase.SETUP_ROAD, Phase.ROAD_BUILDING):
            return self._best_road(s, seat, by_type[_T.BUILD_ROAD])
        if s.phase == Phase.DISCARD:
            hand = s.hands[seat]
            return max(by_type[_T.DISCARD], key=lambda a: hand[a.resource])
        if s.phase == Phase.MOVE_ROBBER:
            return self._robber(s, seat, by_type[_T.MOVE_ROBBER])
        if s.phase == Phase.PRE_ROLL:
            knight = self._knight_if_blocked(s, seat, by_type)
            return knight or by_type[_T.ROLL_DICE][0]
        return self._main(s, seat, by_type)

    # ------------------------------------------------------------------ main phase

    def _main(self, s: GameState, seat: int, by_type: dict[int, list[Action]]) -> Action:
        if _T.BUILD_CITY in by_type:
            return max(by_type[_T.BUILD_CITY], key=lambda a: corner_value(s.board, a.corner))
        plays = by_type.get(_T.PLAY_DEV_CARD, [])
        knight = self._knight_if_blocked(s, seat, by_type)
        if knight:
            return knight
        target = self._target_cost(s, seat)
        for a in plays:
            if a.dev == DevCard.ROAD_BUILDING and not settlement_corners(s, seat):
                return a
            if a.dev == DevCard.YEAR_OF_PLENTY and self._fills_deficit(s, seat, a, target):
                return a
        if (
            _T.BUILD_ROAD in by_type
            and not settlement_corners(s, seat)
            and s.settlements_left[seat]
        ):
            road = self._best_road(s, seat, by_type[_T.BUILD_ROAD])
            if self._road_score(s, seat, road.edge) > 0:
                return road
        if _T.BUY_DEV_CARD in by_type and target in (DEV_CARD_COST, ROAD_COST):
            return by_type[_T.BUY_DEV_CARD][0]
        trade = self._trade_toward(s, seat, by_type.get(_T.MARITIME_TRADE, []), target)
        return trade or by_type[_T.END_TURN][0]

    def _target_cost(self, s: GameState, seat: int) -> tuple[int, ...]:
        if s.cities_left[seat] and city_corners(s, seat):
            return CITY_COST
        if s.settlements_left[seat] and settlement_corners(s, seat):
            return SETTLEMENT_COST
        if s.roads_left[seat] and s.settlements_left[seat] and road_edges(s, seat):
            return ROAD_COST
        return DEV_CARD_COST

    def _fills_deficit(self, s: GameState, seat: int, action: Action, cost) -> bool:
        hand = s.hands[seat]
        deficit = [max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES)]
        return deficit[action.resource] > 0 and deficit[action.resource2] > (
            1 if action.resource == action.resource2 else 0
        )

    def _trade_toward(self, s: GameState, seat: int, trades: list[Action], cost) -> Action | None:
        hand, ratios = s.hands[seat], trade_ratios(s, seat)
        useful = [
            a
            for a in trades
            if hand[a.resource2] < cost[a.resource2]
            and hand[a.resource] - ratios[a.resource] >= cost[a.resource]
        ]
        return max(useful, key=lambda a: hand[a.resource] - cost[a.resource], default=None)

    # ------------------------------------------------------------------ placement

    def _road_score(self, s: GameState, seat: int, edge: int) -> float:
        """Value of the best free settlement site one or two corners beyond `edge`."""
        best = 0.0
        for corner in EDGE_CORNERS[edge]:
            for site in (corner, *CORNER_NEIGHBORS[corner]):
                if distance_rule_ok(s, site):
                    best = max(best, corner_value(s.board, site) / (1 if site == corner else 2))
        return best

    def _best_road(self, s: GameState, seat: int, roads: list[Action]) -> Action:
        return max(roads, key=lambda a: self._road_score(s, seat, a.edge))

    # ------------------------------------------------------------------ robber

    def _knight_if_blocked(self, s: GameState, seat: int, by_type) -> Action | None:
        blocked = any(s.corner_owner[c] == seat for c in TILE_CORNERS[s.robber_tile])
        for a in by_type.get(_T.PLAY_DEV_CARD, []):
            if a.dev == DevCard.KNIGHT and blocked:
                return a
        return None

    def _robber(self, s: GameState, seat: int, moves: list[Action]) -> Action:
        public_vp = [victory_points(s, p, include_hidden=False) for p in range(NUM_PLAYERS)]

        def harm(a: Action) -> float:
            score = 0.0
            for c in TILE_CORNERS[a.tile]:
                owner = s.corner_owner[c]
                if owner == seat:
                    score -= 10.0
                elif owner >= 0:
                    score += s.corner_building[c] * (1 + public_vp[owner])
            pips = PIPS.get(s.board.tile_number[a.tile], 0)
            victim_bonus = public_vp[a.player] + sum(s.hands[a.player]) / 10 if a.player >= 0 else 0
            return score * pips + victim_bonus

        return max(moves, key=harm)
