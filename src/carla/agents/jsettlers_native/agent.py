"""Native base-game SMART/FAST orchestration with declared CARLA adaptations.

Translated decision components derive from JSettlers 2.6.10, GPL-3.0-or-later.
Whole-policy equivalence is a separate, currently experimental release gate.
"""

from collections.abc import Sequence

from carla.agents.api import AgentCapabilities, AgentView
from carla.agents.base import Agent
from carla.engine.actions import Action, ActionType
from carla.engine.constants import DevCard
from carla.engine.events import EventKind
from carla.engine.rules import legal_trade_offer
from carla.engine.state import Phase
from carla.engine.topology import CORNER_TILES

from .coordinates import LAND_HEX_ORDER, hashtable_order, initial_targets, potential_order
from .estimator import ZERO, estimates, periods_for_corners, player_corners, ratios_for_corners
from .fast import fast_plan
from .knowledge import Knowledge
from .negotiation import NegotiationMemory, Negotiator, Response
from .opening import Opening, legal_site
from .planner import projected_win_times, smart_plan
from .random import JavaRandom
from .strategies import bank_trade, discard, free_resources, knight_for_army, monopoly
from .trackers import PieceType, Tracker, update_conflicts
from .view import query_state

T = ActionType


class NativeJSettlersAgent(Agent):
    capabilities = AgentCapabilities(victory_points=(10,), domestic_trading="adapted")
    conformance = "experimental: component agreement is not whole-policy equivalence"

    def __init__(self, strategy="smart", *, compatibility="native"):
        if strategy not in ("smart", "fast"):
            raise ValueError("strategy must be smart or fast")
        if compatibility not in ("native", "java-adapter"):
            raise ValueError("compatibility must be native or java-adapter")
        self.strategy = strategy
        self.compatibility = compatibility
        self.name = f"jsettlers-{strategy}-native"

    def reset(self, game, seat, seed):
        self.capabilities.validate(game)
        view = AgentView.from_game(game, seat)
        self.seat, self.rng = seat, JavaRandom(seed)
        self.knowledge = Knowledge(view.state, seat, view.step)
        self.state = query_state(view, self.knowledge)
        self.remaining = view.state["development_remaining"]
        self.trackers = [Tracker(p) for p in range(4)]
        self.memory, self.opening = NegotiationMemory(seat=seat), Opening()
        self.memory.reset_turn(self.knowledge.hands)
        self.plan = []
        self.wanted_road = None
        self.turn = self.state.turn
        # SOCRobotBrain.doneTrading: set when makeOffer finds no offer, reset at each turn start
        # (resetFieldsAndBuildingPlan), so a robot stops proposing for the rest of that turn.
        self.done_trading = False
        self.diagnostics = {}
        if self.state.phase > Phase.SETUP_ROAD:
            self._refresh()

    def _refresh(self):
        for tracker in self.trackers:
            tracker.refresh(self.state)
        update_conflicts(self.trackers, self.state)

    def _start_turn(self, turn):
        """resetFieldsAndBuildingPlan, which the original runs on every TURN message."""
        self.plan.clear()
        self.memory.reset_turn(self.knowledge.hands)
        self.turn = turn
        self.done_trading = False

    def observe(self, events: Sequence, step: int):
        if not self.knowledge.observe(events, step):
            return
        construction = produced = False
        for event in events:
            self.memory.observe(event)
            produced |= (
                event.kind == EventKind.RESOURCES_PRODUCED
                and event.player == self.seat
                and sum(event.counts) > 0
            )
            p, location = event.player, event.location
            kind = event.kind
            if kind == EventKind.TURN_ENDED:
                # Reset now, before the next player's first offer arrives: resetting lazily at
                # our next decision would also forget offers made earlier in that turn.
                self._start_turn(self.turn + 1)
            elif kind == EventKind.ROAD_BUILT:
                self.state.edge_owner[location] = p
                self.state.roads_left[p] -= 1
                construction = True
                if p == self.seat and self.wanted_road == location:
                    self.wanted_road = None
            elif kind == EventKind.SETTLEMENT_BUILT:
                self.state.corner_owner[location] = p
                self.state.corner_building[location] = 1
                self.state.settlements_left[p] -= 1
                construction = True
            elif kind == EventKind.CITY_BUILT:
                self.state.corner_building[location] = 2
                self.state.cities_left[p] -= 1
                self.state.settlements_left[p] += 1
                construction = True
            elif kind == EventKind.LONGEST_ROAD_CHANGED:
                self.state.longest_road_owner = p
            elif kind == EventKind.LARGEST_ARMY_CHANGED:
                self.state.largest_army_owner = p
            if self.plan and p == self.seat:
                piece = self.plan[0]
                matches = (
                    (kind == EventKind.ROAD_BUILT and piece.type == PieceType.ROAD)
                    or (kind == EventKind.SETTLEMENT_BUILT and piece.type == PieceType.SETTLEMENT)
                    or (kind == EventKind.CITY_BUILT and piece.type == PieceType.CITY)
                ) and piece.location == location
                if matches or (kind == EventKind.DEV_CARD_BOUGHT and piece.type == PieceType.CARD):
                    self.plan.pop(0)
            for tracker in self.trackers:
                tracker.observe((event,))
        if produced:
            # The server sends a gaining player its hand after a roll, still in ROLL_OR_CARD,
            # and handlePLAYERELEMENT then resets every player's selling flags.
            self.memory.reset_is_selling(self.knowledge.hands)
        # Trackers start once the complete setup board exists; resource-only
        # transitions do not change their candidate dependency graph.
        if construction and self.knowledge.setup_roads >= 8:
            self._refresh()

    def _plan_for(self, player):
        if self.strategy == "fast":
            return fast_plan(self.state, player, self.trackers[player], self.remaining)
        return smart_plan(self.state, player, self.trackers, self.remaining)

    def _negotiator(self):
        return Negotiator(
            self.seat,
            self.memory,
            self.knowledge,
            self._plan_for,
            lambda player: self.trackers[player].win_eta,
        )

    def _robber(self, legal):
        s, seat = self.state, self.seat
        # selectPlayerToThwart only calls recalcWinGameETA: award estimates stay as cached.
        etas = projected_win_times(s, self.trackers, self.remaining)
        victim = min((p for p in range(4) if p != seat), key=lambda p: etas[p])
        own_tiles = {
            t
            for c, owner in enumerate(s.corner_owner)
            if owner == seat
            for t in CORNER_TILES[c]
            if s.board.tile_resource[t] >= 0
        }
        corners = player_corners(s, victim)
        ratios = ratios_for_corners(s.board, corners)
        permitted = {a.tile for a in legal}
        tiles = [tile for tile in LAND_HEX_ORDER if tile in permitted]
        best, worst = None, 0
        for tile in tiles:
            if tile in own_tiles:
                continue
            speed = sum(
                estimates(
                    ZERO, periods_for_corners(s.board, corners, tile), ratios, include_ship=True
                )
            )
            if speed > worst:
                best, worst = tile, speed
        if best is None:
            # Original strategy's random recovery prefers an own-producing
            # tile for its first 30 attempts; preserve that unusual condition.
            best, attempts = s.robber_tile, 0
            while best not in permitted or (attempts < 30 and best not in own_tiles):
                best = LAND_HEX_ORDER[self.rng.index(19)]
                attempts += 1
        choices = [a for a in legal if a.tile == best]
        return min(choices, key=lambda a: etas[a.player] if a.player >= 0 else 10000)

    def act(self, game, seat, legal):
        if seat != self.seat:
            raise ValueError("agent seat changed without reset")
        view = AgentView.from_game(game, seat)
        self.knowledge.reconcile(view.state)
        self.state, self.remaining = (
            query_state(view, self.knowledge),
            view.state["development_remaining"],
        )
        s = self.state
        if s.turn != self.turn:  # Fallback; turns normally start in `observe`.
            self._start_turn(s.turn)
        by_type = {}
        for action in legal:
            by_type.setdefault(action.type, []).append(action)
        if s.phase == Phase.SETUP_SETTLEMENT:
            candidates = potential_order(a.corner for a in legal)
            corner = (
                self.opening.first_settlement(s.board, candidates)
                if s.setup_step < 4
                else self.opening.second_settlement(s.board, candidates)
            )
            return Action(T.BUILD_SETTLEMENT, corner=corner)
        if s.phase == Phase.SETUP_ROAD:
            candidates = potential_order(c for c in range(54) if legal_site(s, c))
            targets = hashtable_order(c for c in initial_targets(s.setup_corner) if c in candidates)
            edge = self.opening.initial_road(
                s, seat, s.setup_corner, candidates, hashtable_order(candidates), targets
            )
            return Action(T.BUILD_ROAD, edge=edge)
        self._refresh()
        if s.phase == Phase.MOVE_ROBBER:
            return self._robber(legal)
        if s.phase == Phase.PRE_ROLL:
            blocked = any(s.robber_tile in CORNER_TILES[c] for c in player_corners(s, seat))
            knight = next(
                (a for a in by_type.get(T.PLAY_DEV_CARD, ()) if a.dev == DevCard.KNIGHT), None
            )
            return (
                knight
                if knight is not None and blocked and s.board.tile_resource[s.robber_tile] >= 0
                else by_type[T.ROLL_DICE][0]
            )
        corners = player_corners(s, seat)
        periods, ratios = periods_for_corners(s.board, corners), ratios_for_corners(
            s.board, corners
        )
        if s.phase == Phase.MAIN and not self.plan and sum(s.hands[seat]) <= 1:
            knight = next(
                (a for a in by_type.get(T.PLAY_DEV_CARD, ()) if a.dev == DevCard.KNIGHT), None
            )
            return (
                knight
                if knight is not None and knight_for_army(s, seat)
                else by_type[T.END_TURN][0]
            )
        if s.phase == Phase.MAIN:
            # planAndDoActionForPLAY1 plays a Largest Army knight before planning; the pending
            # robber placement then skips planning (and its tracker ETA refresh) entirely.
            knight = next(
                (a for a in by_type.get(T.PLAY_DEV_CARD, ()) if a.dev == DevCard.KNIGHT), None
            )
            if knight is not None and knight_for_army(s, seat):
                return knight
        if s.phase in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
            # handleMAKEOFFER only negotiates; it does not plan our build (which would refresh
            # tracker ETAs). considerOffer2 plans into a scratch stack when it needs a target.
            return self._trade(s, by_type)
        if not self.plan:
            self.plan = list(self._plan_for(seat))
        if s.phase == Phase.DISCARD:
            target = self.plan[0].cost if self.plan else ZERO
            chosen = discard(s.hands[seat], target, periods, 1)
            return next(a for a in legal if chosen[a.resource])
        if s.phase == Phase.ROAD_BUILDING:
            if self.compatibility == "java-adapter":
                wanted = (
                    Action(T.BUILD_ROAD, edge=self.wanted_road)
                    if self.wanted_road is not None
                    else None
                )
                if wanted in legal:
                    return wanted
                # The Java sidecar invokes planStuff on the retained stack,
                # then selects the highest-scoring legal road independently.
                self.plan = [*self._plan_for(seat), *self.plan]
                return max(legal, key=lambda a: self.trackers[seat].roads[a.edge].score)
            for piece in self.plan:
                if piece.type == PieceType.ROAD and piece.action() in legal:
                    return piece.action()
            # The original adapter revalidates the tracker road candidates.
            return max(legal, key=lambda a: self.trackers[seat].roads[a.edge].score)
        plays = by_type.get(T.PLAY_DEV_CARD, ())
        if not self.plan:
            return by_type[T.END_TURN][0]
        self.memory.targets[seat] = self.plan[0]
        target = self.plan[0].cost
        if len(self.plan) > 1 and all(p.type == PieceType.ROAD for p in self.plan[:2]):
            action = next((a for a in plays if a.dev == DevCard.ROAD_BUILDING), None)
            if action is not None:
                self.wanted_road = self.plan[0].location
                return action
        deficit = sum(max(0, t - h) for t, h in zip(target, s.hands[seat], strict=True))
        if deficit == 2:
            resources = free_resources(s.hands[seat], target, periods)
            for action in plays:
                if action.dev == DevCard.YEAR_OF_PLENTY:
                    selected = [0] * 5
                    selected[action.resource] += 1
                    selected[action.resource2] += 1
                    if tuple(selected) == resources:
                        return action
        resource = monopoly([hand.known for hand in self.knowledge.hands], seat, ratios)
        for action in plays:
            if action.dev == DevCard.MONOPOLY and action.resource == resource:
                return action
        if deficit and T.PROPOSE_TRADE in by_type and not self.done_trading:
            negotiator = self._negotiator()
            offer = negotiator.make_offer(s)
            self.memory.wants_another = [[False] * 5 for _ in range(4)]
            action = Action(T.PROPOSE_TRADE, trade=offer)
            if offer is not None and legal_trade_offer(s, action):
                self.memory.offers_made.append(offer)
                return action
            if offer is None:
                self.done_trading = True
        trade = bank_trade(s.hands[seat], target, periods, ratios)
        if trade in legal:
            return trade
        action = self.plan[0].action()
        if action in legal:
            return action
        return by_type[T.END_TURN][0]

    def _trade(self, state, by_type):
        if state.phase == Phase.TRADE_CONFIRM and self.compatibility == "java-adapter":
            return by_type.get(T.CONFIRM_TRADE, by_type.get(T.CANCEL_TRADE))[0]
        negotiator = self._negotiator()
        if state.phase == Phase.TRADE_RESPONSE:
            offer = state.pending_trade
            if offer is not None:
                response = negotiator.consider(state, state.current_player, self.seat, offer)
                if response == Response.ACCEPT and T.ACCEPT_TRADE in by_type:
                    return by_type[T.ACCEPT_TRADE][0]
                # handleMAKEOFFER only counteroffers on COUNTER_OFFER.
                if response != Response.COUNTER:
                    return by_type[T.REJECT_TRADE][0]
            counter = negotiator.make_offer(state, original=offer)
            action = Action(T.PROPOSE_TRADE, trade=counter)
            return (
                action
                if counter is not None and legal_trade_offer(state, action)
                else by_type[T.REJECT_TRADE][0]
            )
        candidates = []
        for action in by_type.get(T.CONFIRM_TRADE, ()):
            offer = state.trade_counteroffers[action.player]
            if offer is not None:
                if negotiator.consider(state, action.player, self.seat, offer) == Response.ACCEPT:
                    candidates.append(action)
            elif state.pending_trade is not None:
                candidates.append(action)
        # CARLA confirmation is an explicit adaptation: use seat order among
        # acceptable offers. The Java adapter currently takes the first legal
        # offer without this counteroffer-value check; traces declare this gap.
        return candidates[0] if candidates else by_type[T.CANCEL_TRADE][0]
