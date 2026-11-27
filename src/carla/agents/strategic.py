"""Strategy-styled heuristic bots: teachers for strategy-conditioned imitation and league members.

Each `StrategicHeuristic` follows one coherent plan at a time. A plan is a goal purchase (city,
settlement, road toward a site, development card) chosen by a style-specific value divided by
the rounds until it is affordable, so a goal that would take too long yields to a cheaper one
(fallback). The bot keeps its goal unless another one scores `SWITCH_MARGIN` times higher
(coherence), reserves the cards the goal needs (it does not spend, discard, or trade them away),
trades only surplus toward the goal, and makes other purchases only when they leave the
reserved cards untouched.

Styles:
- city: grain and ore, upgrades to cities first, development cards as the fallback.
- expansion: lumber and brick, roads and settlements, contests longest road.
- devcard: wool, grain and ore, buys development cards, plays knights for largest army.

Like every agent, it reads only public information and its own cards. Decisions are
deterministic.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from carla.agents.base import Agent
from carla.engine.actions import Action, ActionType, TradeOffer
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

_T = ActionType
GOAL_COSTS = {
    "city": CITY_COST,
    "settlement": SETTLEMENT_COST,
    "road": ROAD_COST,
    "dev": DEV_CARD_COST,
}
SWITCH_MARGIN = 1.5


@dataclass(frozen=True)
class Style:
    name: str
    resource_weights: tuple[float, ...]
    """Value of one pip of each resource (lumber, brick, wool, grain, ore)."""
    goal_values: dict
    """Base value of each goal before dividing by the rounds until it is affordable."""
    knight_eagerness: float
    """0: knights only against a robber on own tiles; 1: play every knight early."""
    longest_road: bool
    """Build extra roads to take or keep longest road."""


STYLES = {
    "city": Style(
        "city",
        (0.7, 0.7, 0.6, 1.35, 1.45),
        {"city": 3.2, "settlement": 2.0, "road": 0.9, "dev": 1.1},
        0.2,
        False,
    ),
    "expansion": Style(
        "expansion",
        (1.4, 1.4, 0.9, 1.0, 0.55),
        {"city": 1.5, "settlement": 3.2, "road": 1.7, "dev": 0.5},
        0.2,
        True,
    ),
    "devcard": Style(
        "devcard",
        (0.65, 0.65, 1.25, 1.2, 1.25),
        {"city": 2.4, "settlement": 2.6, "road": 1.0, "dev": 2.0},
        1.0,
        False,
    ),
}


def _production(s: GameState, seat: int) -> list[float]:
    """Expected cards per round by resource (public buildings; robber tile blocked)."""
    out = [0.0] * NUM_RESOURCES
    for corner, owner in enumerate(s.corner_owner):
        if owner != seat:
            continue
        for tile in CORNER_TILES[corner]:
            r = s.board.tile_resource[tile]
            if r != DESERT and tile != s.robber_tile:
                out[r] += s.corner_building[corner] * PIPS.get(s.board.tile_number[tile], 0) / 9
    return out


class StrategicHeuristic(Agent):
    """A heuristic bot with a coherent, style-specific plan (see the module docstring)."""

    def __init__(self, style: str) -> None:
        if style not in STYLES:
            raise ValueError(f"unknown heuristic style {style!r}; known: {sorted(STYLES)}")
        self.style = STYLES[style]
        self.name = f"heuristic_{style}"
        self.goal: str | None = None
        self.goal_changes = 0
        self.multi_card_offers = True
        """Offer two cards for one when well stocked (off for one-for-one policy heads)."""

    def reset(self, game: Game, seat: int, seed: int) -> None:
        self.goal = None
        self.goal_changes = 0

    # ------------------------------------------------------------------ valuation

    def corner_value(self, s: GameState, corner: int, owned: Sequence[float] = ()) -> float:
        """Style-weighted production of a corner, favoring resources the seat lacks."""
        weights = self.style.resource_weights
        value, kinds = 0.0, set()
        for t in CORNER_TILES[corner]:
            r = s.board.tile_resource[t]
            if r == DESERT:
                continue
            pips = PIPS[s.board.tile_number[t]] * (0.5 if t == s.robber_tile else 1.0)
            scarcity = 1.0 + (0.35 if owned and owned[r] == 0 else 0.0)
            value += pips * weights[r] * scarcity
            kinds.add(r)
        port = s.board.corner_port[corner]
        if port == GENERIC_PORT:
            value += 1.0
        elif port >= 0:
            value += 2.5 * weights[port] - 1.0
        return value + 1.2 * len(kinds)

    def _road_score(self, s: GameState, seat: int, edge: int) -> float:
        owned = _production(s, seat)
        best = 0.0
        for corner in EDGE_CORNERS[edge]:
            for site in (corner, *CORNER_NEIGHBORS[corner]):
                if s.corner_owner[site] < 0 and distance_rule_ok(s, site):
                    near = site == corner
                    best = max(best, self.corner_value(s, site, owned) / (1 if near else 2))
        return best

    def _best_road(self, s: GameState, seat: int, roads: list[Action]) -> Action:
        return max(roads, key=lambda a: (self._road_score(s, seat, a.edge), -a.edge))

    # ------------------------------------------------------------------ plan

    def _feasible(self, s: GameState, seat: int) -> dict[str, float]:
        """Goals that make progress now, with their style values."""
        values = dict(self.style.goal_values)
        out = {}
        if s.cities_left[seat] and city_corners(s, seat):
            out["city"] = values["city"]
        if s.settlements_left[seat] and settlement_corners(s, seat):
            out["settlement"] = values["settlement"]
        if s.roads_left[seat] and s.settlements_left[seat] and road_edges(s, seat):
            roads = road_edges(s, seat)
            if not settlement_corners(s, seat) and any(
                self._road_score(s, seat, e) > 0 for e in roads
            ):
                out["road"] = values["road"] + 1.0  # roads unlock the next settlement
            elif self.style.longest_road and self._road_race(s, seat):
                out["road"] = values["road"]
        if any(s.dev_deck):
            out["dev"] = values["dev"]
        return out

    def _road_race(self, s: GameState, seat: int) -> bool:
        """Whether one or two roads would take or defend longest road."""
        holder = s.longest_road_owner
        mine = s.road_lengths[seat]
        if holder == seat:
            rival = max(s.road_lengths[p] for p in range(NUM_PLAYERS) if p != seat)
            return rival >= mine - 1
        target = max(5, s.road_lengths[holder] + 1 if holder >= 0 else 5)
        return target - mine <= 2

    def _rounds(self, s: GameState, seat: int, cost: tuple[int, ...]) -> float:
        hand, rates = s.hands[seat], _production(s, seat)
        ratios = trade_ratios(s, seat)
        missing = [max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES)]
        # Surplus that maritime trade can turn into missing cards.
        trades = sum(
            (hand[r] - cost[r]) // ratios[r] for r in range(NUM_RESOURCES) if hand[r] > cost[r]
        )
        short = max(0, sum(missing) - trades)
        if short == 0:
            return 0.0
        worst = 0.0
        for r in range(NUM_RESOURCES):
            if missing[r]:
                worst = max(worst, missing[r] / max(rates[r], 0.15))
        return min(worst, 12.0) * short / max(1, sum(missing))

    def plan(self, s: GameState, seat: int) -> str:
        """The current goal: keep it unless infeasible or clearly beaten."""
        feasible = self._feasible(s, seat)
        if not feasible:
            self.goal = None
            return "dev"
        missing_vp = s.config.vp_to_win - victory_points(s, seat)
        scores = {}
        for goal, value in feasible.items():
            if missing_vp <= 2 and goal in ("city", "settlement"):
                value *= 1.5  # Close to winning: points first.
            scores[goal] = value / (1.0 + self._rounds(s, seat, GOAL_COSTS[goal]))
        best = max(scores, key=lambda g: (scores[g], g))
        if self.goal in scores and scores[best] < SWITCH_MARGIN * scores[self.goal]:
            return self.goal
        if best != self.goal:
            self.goal_changes += 1
        self.goal = best
        return best

    def reserved(self, s: GameState, seat: int, goal: str | None = None) -> list[int]:
        goal = goal or self.goal or "dev"
        cost = GOAL_COSTS[goal]
        hand = s.hands[seat]
        return [min(hand[r], cost[r]) for r in range(NUM_RESOURCES)]

    def _keeps_reserve(self, s: GameState, seat: int, cost: tuple[int, ...]) -> bool:
        hand, keep = s.hands[seat], self.reserved(s, seat)
        return all(hand[r] - cost[r] >= keep[r] for r in range(NUM_RESOURCES))

    # ------------------------------------------------------------------ decisions

    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        s = game.state
        by_type: dict[int, list[Action]] = {}
        for a in legal:
            by_type.setdefault(a.type, []).append(a)
        if s.phase == Phase.SETUP_SETTLEMENT:
            owned = _production(s, seat)
            return max(
                by_type[_T.BUILD_SETTLEMENT],
                key=lambda a: (self.corner_value(s, a.corner, owned), -a.corner),
            )
        if s.phase in (Phase.SETUP_ROAD, Phase.ROAD_BUILDING):
            return self._best_road(s, seat, by_type[_T.BUILD_ROAD])
        if s.phase == Phase.DISCARD:
            return self._discard(s, seat, by_type[_T.DISCARD])
        if s.phase == Phase.MOVE_ROBBER:
            return self._robber(s, seat, by_type[_T.MOVE_ROBBER])
        if s.phase == Phase.PRE_ROLL:
            knight = self._knight(s, seat, by_type, before_roll=True)
            return knight or by_type[_T.ROLL_DICE][0]
        if s.phase == Phase.TRADE_RESPONSE:
            return self._trade_response(s, seat, by_type)
        if s.phase == Phase.TRADE_CONFIRM:
            return self._trade_confirm(s, seat, by_type)
        return self._main(s, seat, by_type)

    def _main(self, s: GameState, seat: int, by_type: dict[int, list[Action]]) -> Action:
        goal = self.plan(s, seat)
        owned = _production(s, seat)
        order = sorted(self.style.goal_values, key=lambda g: -self.style.goal_values[g])
        # Buy the goal, or anything ranked above it; lower-ranked purchases only from surplus.
        for candidate in order:
            action = self._purchase(s, seat, by_type, candidate, owned)
            if action is None:
                continue
            ranked_higher = self.style.goal_values[candidate] >= self.style.goal_values[goal]
            if (
                candidate == goal
                or ranked_higher
                or self._keeps_reserve(s, seat, GOAL_COSTS[candidate])
            ):
                return action
        card = self._dev_play(s, seat, by_type, goal)
        if card is not None:
            return card
        offer = self._domestic_offer(s, seat, by_type, goal)
        if offer is not None:
            return offer
        trade = self._maritime(s, seat, by_type.get(_T.MARITIME_TRADE, []), goal)
        return trade or by_type[_T.END_TURN][0]

    def _road_useful(self, s: GameState, seat: int) -> bool:
        return (not settlement_corners(s, seat) and s.settlements_left[seat] > 0) or (
            self.style.longest_road and self._road_race(s, seat)
        )

    def _purchase(self, s, seat, by_type, goal: str, owned) -> Action | None:
        if goal == "city" and _T.BUILD_CITY in by_type:
            return max(
                by_type[_T.BUILD_CITY], key=lambda a: (self.corner_value(s, a.corner), a.corner)
            )
        if goal == "settlement" and _T.BUILD_SETTLEMENT in by_type:
            return max(
                by_type[_T.BUILD_SETTLEMENT],
                key=lambda a: (self.corner_value(s, a.corner, owned), -a.corner),
            )
        if goal == "road" and _T.BUILD_ROAD in by_type and self._road_useful(s, seat):
            road = self._best_road(s, seat, by_type[_T.BUILD_ROAD])
            if self._road_score(s, seat, road.edge) > 0 or self.style.longest_road:
                return road
        if goal == "dev" and _T.BUY_DEV_CARD in by_type:
            return by_type[_T.BUY_DEV_CARD][0]
        return None

    # ------------------------------------------------------------------ cards

    def _dev_play(self, s, seat, by_type, goal: str) -> Action | None:
        knight = self._knight(s, seat, by_type, before_roll=False)
        if knight is not None:
            return knight
        cost = GOAL_COSTS[goal]
        hand = s.hands[seat]
        deficit = [max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES)]
        for a in by_type.get(_T.PLAY_DEV_CARD, []):
            if (
                a.dev == DevCard.ROAD_BUILDING
                and (goal in ("road", "settlement") or self.style.longest_road)
                and (not settlement_corners(s, seat) or self.style.longest_road)
            ):
                return a
        plenty = [
            a
            for a in by_type.get(_T.PLAY_DEV_CARD, [])
            if a.dev == DevCard.YEAR_OF_PLENTY
            and deficit[a.resource] > 0
            and deficit[a.resource2] > (1 if a.resource == a.resource2 else 0)
        ]
        if plenty:
            return max(
                plenty, key=lambda a: (deficit[a.resource] + deficit[a.resource2], -a.resource)
            )
        monopolies = [a for a in by_type.get(_T.PLAY_DEV_CARD, []) if a.dev == DevCard.MONOPOLY]
        if monopolies:
            best, take = None, 0.0
            for a in monopolies:
                expected = self._expected_holdings(s, seat, a.resource)
                useful = deficit[a.resource] > 0 or trade_ratios(s, seat)[a.resource] <= 2
                if useful and expected > take:
                    best, take = a, expected
            if best is not None and take >= 3.0:
                return best
        return None

    def _expected_holdings(self, s: GameState, seat: int, resource: int) -> float:
        """Public estimate of the opponents' cards of `resource`: hand size split by their
        production shares (hands themselves are hidden)."""
        total = 0.0
        for p in range(NUM_PLAYERS):
            if p == seat:
                continue
            rates = _production(s, p)
            share = rates[resource] / sum(rates) if sum(rates) > 0 else 0.2
            total += s.hand_size(p) * share
        return total

    def _knight(self, s, seat, by_type, before_roll: bool) -> Action | None:
        knights = [a for a in by_type.get(_T.PLAY_DEV_CARD, []) if a.dev == DevCard.KNIGHT]
        if not knights:
            return None
        blocked = any(s.corner_owner[c] == seat for c in TILE_CORNERS[s.robber_tile])
        if blocked:
            return knights[0]
        played = s.dev_played[seat][DevCard.KNIGHT]
        holder = s.largest_army_owner
        rival = max(
            (s.dev_played[p][DevCard.KNIGHT] for p in range(NUM_PLAYERS) if p != seat), default=0
        )
        racing = holder != seat and played + 1 >= max(3, rival + (1 if holder >= 0 else 0))
        if racing and self.style.knight_eagerness >= 0.2:
            return knights[0]
        if self.style.knight_eagerness >= 1.0 and before_roll:
            return knights[0]
        return None

    # ------------------------------------------------------------------ trade

    def _maritime(self, s, seat, trades: list[Action], goal: str) -> Action | None:
        hand, ratios, cost = s.hands[seat], trade_ratios(s, seat), GOAL_COSTS[goal]
        useful = [
            a
            for a in trades
            if hand[a.resource2] < cost[a.resource2]
            and hand[a.resource] - ratios[a.resource] >= cost[a.resource]
        ]
        return max(
            useful,
            key=lambda a: (
                hand[a.resource] - cost[a.resource],
                cost[a.resource2] - hand[a.resource2],
                -a.resource,
            ),
            default=None,
        )

    def _domestic_offer(self, s, seat, by_type, goal: str) -> Action | None:
        if _T.PROPOSE_TRADE not in by_type or (
            s.phase == Phase.MAIN and s.trade_proposals_this_turn
        ):
            return None
        hand, cost = s.hands[seat], GOAL_COSTS[goal]
        needs = [r for r in range(NUM_RESOURCES) if hand[r] < cost[r]]
        spare = sorted(
            (r for r in range(NUM_RESOURCES) if hand[r] > cost[r]),
            key=lambda r: (cost[r] - hand[r], r),
        )
        if not needs or not spare:
            return None
        give, receive = [0] * NUM_RESOURCES, [0] * NUM_RESOURCES
        give[spare[0]] = 1
        if (
            self.multi_card_offers
            and hand[spare[0]] - cost[spare[0]] >= 3
            and s.config.max_trade_cards_per_side != 1
        ):
            give[spare[0]] = 2  # Generous offers when well stocked.
        receive[needs[0]] = 1
        return Action(_T.PROPOSE_TRADE, trade=TradeOffer(tuple(give), tuple(receive)))

    def _leader_threat(self, s: GameState, other: int) -> bool:
        return victory_points(s, other, include_hidden=False) >= s.config.vp_to_win - 2

    def _trade_response(self, s, seat, by_type) -> Action:
        goal = self.plan(s, seat) if s.phase == Phase.MAIN else (self.goal or "dev")
        hand, cost = s.hands[seat], GOAL_COSTS[goal]
        if _T.ACCEPT_TRADE in by_type and not self._leader_threat(s, s.current_player):
            offer = s.pending_trade
            assert offer is not None
            keep = self.reserved(s, seat, goal)
            gives_reserved = any(hand[r] - offer.receive[r] < keep[r] for r in range(NUM_RESOURCES))
            before = sum(max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES))
            after = sum(
                max(0, cost[r] - hand[r] - offer.give[r] + offer.receive[r])
                for r in range(NUM_RESOURCES)
            )
            if after < before and not gives_reserved:
                return by_type[_T.ACCEPT_TRADE][0]
        counter = None
        if not self._leader_threat(s, s.current_player):
            counter = self._domestic_offer(s, seat, by_type, goal)
        return counter or by_type[_T.REJECT_TRADE][0]

    def _trade_confirm(self, s, seat, by_type) -> Action:
        goal = self.goal or "dev"
        hand, cost = s.hands[seat], GOAL_COSTS[goal]
        before = sum(max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES))
        best, best_gain = by_type[_T.CANCEL_TRADE][0], 0
        for action in by_type.get(_T.CONFIRM_TRADE, ()):
            if self._leader_threat(s, action.player):
                continue
            counter = s.trade_counteroffers[action.player]
            if counter is None:
                assert s.pending_trade is not None
                give, receive = s.pending_trade
            else:
                give, receive = counter.receive, counter.give
            after = sum(
                max(0, cost[r] - hand[r] + give[r] - receive[r]) for r in range(NUM_RESOURCES)
            )
            if before - after > best_gain:
                best, best_gain = action, before - after
        return best

    # ------------------------------------------------------------------ robber and discards

    def _discard(self, s, seat, discards: list[Action]) -> Action:
        hand = s.hands[seat]
        keep = self.reserved(s, seat)
        weights = self.style.resource_weights

        def key(a: Action) -> tuple:
            surplus = hand[a.resource] - keep[a.resource]
            return (surplus > 0, surplus, -weights[a.resource], -a.resource)

        return max(discards, key=key)

    def _robber(self, s, seat, moves: list[Action]) -> Action:
        public_vp = [victory_points(s, p, include_hidden=False) for p in range(NUM_PLAYERS)]

        def harm(a: Action) -> tuple:
            score = 0.0
            for c in TILE_CORNERS[a.tile]:
                owner = s.corner_owner[c]
                if owner == seat:
                    score -= 10.0
                elif owner >= 0:
                    score += s.corner_building[c] * (1 + public_vp[owner])
            pips = PIPS.get(s.board.tile_number[a.tile], 0)
            victim = public_vp[a.player] + s.hand_size(a.player) / 8 if a.player >= 0 else 0.0
            return (score * pips + victim, -a.tile, a.player)

        return max(moves, key=harm)


class CityHeuristic(StrategicHeuristic):
    def __init__(self) -> None:
        super().__init__("city")


class ExpansionHeuristic(StrategicHeuristic):
    def __init__(self) -> None:
        super().__init__("expansion")


class DevCardHeuristic(StrategicHeuristic):
    def __init__(self) -> None:
        super().__init__("devcard")
