"""Visible resource accounting, following JSettlers' SOCResourceSet semantics.

Derived from JSettlers 2.6.10, Robert S. Thomas, Jeremy D. Monin et al.,
GPL-3.0-or-later. Resource indices at this boundary are CARLA indices.
"""

from dataclasses import dataclass, field

from carla.engine.constants import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST, DevCard
from carla.engine.events import Event, EventKind


@dataclass
class ResourceKnowledge:
    """Known cards plus an unknown count; never guesses a hidden card type."""

    known: list[int] = field(default_factory=lambda: [0] * 5)
    unknown: int = 0

    def change(self, resource: int, amount: int) -> None:
        if resource < 0:
            if amount < 0:
                # A hidden loss invalidates all known resource types, as in
                # SOCResourceSet.subtract(amt, UNKNOWN, true).
                self.unknown += sum(self.known)
                self.known = [0] * 5
            self.unknown += amount
        elif amount >= 0:
            self.known[resource] += amount
        else:
            used = min(self.known[resource], -amount)
            self.known[resource] -= used
            self.unknown -= -amount - used

    def reconcile(self, count: int, exact=None) -> None:
        if exact is not None:
            self.known = list(exact)
            self.unknown = 0
        elif self.unknown < 0 or sum(self.known) > count:
            self.known = [0] * 5
            self.unknown = count
        else:
            self.unknown = count - sum(self.known)


class Knowledge:
    def __init__(self, state: dict, seat: int, step: int = 0):
        self.seat = seat
        self.last_step = step
        self.hands = [ResourceKnowledge() for _ in range(4)]
        self.setup_buildings = sum(owner >= 0 for owner in state["corner_owner"])
        self.setup_roads = sum(owner >= 0 for owner in state["edge_owner"])
        self.free_roads = [0] * 4
        self.free_roads[state["current_player"]] = state["free_roads"]
        self.reconcile(state)

    def reconcile(self, state: dict) -> None:
        for player, hand in enumerate(self.hands):
            hand.reconcile(state["hand_sizes"][player], state["hands"][player])

    def observe(self, events: list[Event], step: int) -> bool:
        if step == self.last_step:
            return False
        if step != self.last_step + 1:
            raise ValueError(f"resource event sequence: expected {self.last_step + 1}, got {step}")
        for event in events:
            self._event(event)
        self.last_step = step
        return True

    def _event(self, event: Event) -> None:
        kind, player = event.kind, event.player
        hand = self.hands[player]
        if kind in (
            EventKind.RESOURCES_PRODUCED,
            EventKind.STARTING_RESOURCES,
            EventKind.MARITIME_TRADED,
            EventKind.YEAR_OF_PLENTY_TAKEN,
        ):
            for resource, amount in enumerate(event.counts):
                hand.change(resource, amount)
        costs = {
            EventKind.SETTLEMENT_BUILT: SETTLEMENT_COST,
            EventKind.CITY_BUILT: CITY_COST,
            EventKind.ROAD_BUILT: ROAD_COST,
            EventKind.DEV_CARD_BOUGHT: DEV_CARD_COST,
        }
        if kind in costs:
            paid = True
            if kind == EventKind.SETTLEMENT_BUILT:
                paid = self.setup_buildings >= 8
                self.setup_buildings += 1
            elif kind == EventKind.ROAD_BUILT:
                paid = self.setup_roads >= 8
                self.setup_roads += 1
                if paid and self.free_roads[player]:
                    self.free_roads[player] -= 1
                    paid = False
            if paid:
                for resource, amount in enumerate(costs[kind]):
                    hand.change(resource, -amount)
        if kind == EventKind.DEV_CARD_PLAYED and event.value == DevCard.ROAD_BUILDING:
            self.free_roads[player] = 2
        elif kind == EventKind.RESOURCE_STOLEN:
            self.hands[event.target].change(event.value, -1)
            hand.change(event.value, 1)
        elif kind == EventKind.DISCARDED:
            hand.change(event.value, -1)
        elif kind == EventKind.MONOPOLY_TAKEN:
            for other, amount in enumerate(event.counts):
                if other != player:
                    self.hands[other].change(event.value, -amount)
                    hand.change(event.value, amount)
        elif kind == EventKind.TRADE_EXECUTED:
            for resource in range(5):
                delta = event.receive[resource] - event.give[resource]
                hand.change(resource, delta)
                self.hands[event.target].change(resource, -delta)
        elif kind == EventKind.TURN_ENDED:
            self.free_roads[player] = 0

    def snapshot(self) -> list[dict]:
        return [{"known": hand.known[:], "unknown": hand.unknown} for hand in self.hands]
