"""Negotiation memory and acceptance kernel from JSettlers 2.6.10.

Derived from SOCRobotNegotiator, Robert S. Thomas, Jeremy D. Monin et al.,
GPL-3.0-or-later. Offer/counteroffer search and target-plan prediction are
separate dependencies, not replaced with CARLA's heuristic.
"""

from dataclasses import dataclass, field
from enum import IntEnum

from carla.engine.actions import TradeOffer
from carla.engine.events import EventKind
from carla.planning.jsettlers_eta import ORDER

from .estimator import periods_for_corners, player_corners, ratios_for_corners, rolls
from .knowledge import ResourceKnowledge
from .strategies import bank_trade
from .trackers import PieceType


class Response(IntEnum):
    REJECT = 0
    ACCEPT = 1
    COUNTER = 2


@dataclass
class NegotiationMemory:
    seat: int | None = None
    selling: list[list[bool]] = field(default_factory=lambda: [[False] * 5 for _ in range(4)])
    wants_another: list[list[bool]] = field(default_factory=lambda: [[False] * 5 for _ in range(4)])
    offers_made: list = field(default_factory=list)
    targets: list = field(default_factory=lambda: [None] * 4)
    offers: dict = field(default_factory=dict)

    def reset_is_selling(self, hands):
        """resetIsSelling: TRUE for known holdings, without clearing old entries."""
        for player, hand in enumerate(hands):
            for resource in range(5):
                if hand.known[resource] > 0:
                    self.selling[player][resource] = True

    def reset_turn(self, hands):
        """resetFieldsAndBuildingPlan's negotiator resets. wantsAnotherOffer survives turns: only
        our own makeOffer (and the constructor) resets it."""
        self.reset_is_selling(hands)
        self.offers_made.clear()
        self.targets = [None] * 4
        self.offers.clear()

    def observe(self, event):
        player = event.player
        if event.kind in (EventKind.TRADE_PROPOSED, EventKind.TRADE_COUNTERED):
            self.offers[player] = (event.give, event.receive)
            if player == self.seat:
                return  # Original handleMAKEOFFER ignores its own server echo.
            for resource in range(5):
                if event.give[resource]:
                    self.wants_another[player][resource] = True
                if event.receive[resource]:
                    self.selling[player][resource] = False
        elif event.kind == EventKind.TRADE_RESPONDED and event.value == 0:
            offer = self.offers.get(event.target)
            if offer is not None:
                for resource, amount in enumerate(offer[1]):
                    if amount and not self.wants_another[player][resource]:
                        self.selling[player][resource] = False
        # Gains never mark a resource as selling (markAsSelling has no caller). The flags are
        # reset from holdings at turn boundaries and after our own production instead.
        elif event.kind in (EventKind.TRADE_EXECUTED, EventKind.TRADE_CANCELLED):
            self.offers.clear()


def consider_offer(
    hand: ResourceKnowledge,
    target,
    give,
    receive,
    periods,
    ratios,
    *,
    sender_win_eta,
    in_race=False,
):
    """Original considerOffer2 after target-piece prediction and race detection.

    give/receive are from the RECEIVER's perspective. No hidden types are
    inferred when the receiver resource set has unknown cards.
    """
    if not hand.unknown and any(h < g for h, g in zip(hand.known, give, strict=True)):
        return Response.REJECT
    if target is None or sender_win_eta <= 25 or in_race:
        return Response.REJECT
    before = rolls(hand.known, target, periods, ratios, cutoff=1000)
    after = ResourceKnowledge(hand.known[:], hand.unknown)
    for resource in range(5):
        after.change(resource, -give[resource])
        after.change(resource, receive[resource])
    changed = rolls(after.known, target, periods, ratios, cutoff=1000)
    return Response.ACCEPT if changed < before else Response.COUNTER


class Negotiator:
    """Original offer-search stages with explicit plan/ETA dependencies."""

    def __init__(self, seat, memory, knowledge, plan_for, win_times):
        self.seat, self.memory, self.knowledge = seat, memory, knowledge
        self.plan_for, self.win_times = plan_for, win_times
        self.recipients = ()

    def eta(self, player):
        return self.win_times(player) if callable(self.win_times) else self.win_times[player]

    def target(self, player):
        if self.memory.targets[player] is None:
            plan = self.plan_for(player)
            self.memory.targets[player] = plan[0] if plan else None
        return self.memory.targets[player]

    def consider(self, state, sender, receiver, offer):
        # considerOffer2's order matters: predicting a target plans (refreshing the real
        # trackers' ETA caches), so the resource check and the receiver's target come first.
        hand = self.knowledge.hands[receiver]
        if not hand.unknown and any(h < g for h, g in zip(hand.known, offer.receive, strict=True)):
            return Response.REJECT
        target = self.target(receiver)
        if target is None:
            return Response.REJECT
        sender_target = self.target(sender)
        if sender_target is None:
            return Response.REJECT
        race = False
        if target.type in (PieceType.ROAD, PieceType.SETTLEMENT):
            # considerOffer2 matches a threat or conflict by piece type and location only, not
            # by whose piece it is.
            race = any(
                kind == int(sender_target.type) and location == sender_target.location
                for _, kind, location in target.threats
            )
            if target.type == PieceType.SETTLEMENT and sender_target.type == PieceType.SETTLEMENT:
                race = race or any(
                    location == sender_target.location for _, location in target.conflicts
                )
        corners = player_corners(state, receiver)
        return consider_offer(
            self.knowledge.hands[receiver],
            target.cost,
            offer.receive,
            offer.give,
            periods_for_corners(state.board, corners),
            ratios_for_corners(state.board, corners),
            sender_win_eta=self.eta(sender),
            in_race=race,
        )

    def _aux(self, state, give, receive, resource):
        offer = TradeOffer(tuple(give), tuple(receive))
        if offer in self.memory.offers_made or any(
            other != self.seat
            and tuple(outside[0]) == offer.receive
            and tuple(outside[1]) == offer.give
            for other, outside in self.memory.offers.items()
        ):
            return None
        allowed = range(4) if state.current_player == self.seat else (state.current_player,)
        recipients = tuple(
            p
            for p in allowed
            if p != self.seat
            and self.memory.selling[p][resource]
            and sum(self.knowledge.hands[p].known) + self.knowledge.hands[p].unknown >= sum(receive)
            and self.eta(p) >= 25
        )
        if any(self.consider(state, self.seat, p, offer) == Response.ACCEPT for p in recipients):
            self.recipients = recipients
            return offer
        return None

    def make_offer(self, state, *, original=None):
        target_piece = self.target(self.seat)
        if target_piece is None:
            return None
        target, hand = target_piece.cost, self.knowledge.hands[self.seat]
        if hand.unknown or all(h >= t for h, t in zip(hand.known, target, strict=True)):
            return None
        corners = player_corners(state, self.seat)
        periods, ratios = periods_for_corners(state.board, corners), ratios_for_corners(
            state.board, corners
        )
        needed = sorted((r for r in ORDER if target[r]), key=lambda r: periods[r])
        unneeded = sorted((r for r in ORDER if not target[r]), key=lambda r: periods[r])
        bank = bank_trade(hand.known, target, periods, ratios)
        batna_hand = hand.known[:]
        if bank is not None:
            batna_hand[bank.resource] -= ratios[bank.resource]
            batna_hand[bank.resource2] += 1
        batna = rolls(batna_hand, target, periods, ratios, 1000)
        leftovers = [max(0, h - t) for h, t in zip(hand.known, target, strict=True)]
        selling = [
            any(self.memory.selling[p][r] for p in range(4) if p != self.seat) for r in range(5)
        ]

        def available(r):
            return original.give[r] > 0 if original is not None else selling[r]

        def trial(give_resources, get, count=1, *, condition=True, strict=False):
            give, receive = [0] * 5, [0] * 5
            for resource in give_resources:
                give[resource] += 1
            receive[get] = count
            if any(g > h for g, h in zip(give, hand.known, strict=True)):
                return None
            changed = [h - g + r for h, g, r in zip(hand.known, give, receive, strict=True)]
            eta = rolls(changed, target, periods, ratios, 1000)
            better = eta < batna or (
                not strict
                and bank is not None
                and eta == batna
                and sum(give) < ratios[bank.resource]
            )
            return self._aux(state, give, receive, get) if not condition or better else None

        missing = [r for r in reversed(needed) if hand.known[r] < target[r] and available(r)]
        if missing:
            get = missing[0]
            for give in unneeded:
                if hand.known[give]:
                    offer = trial((give,), get, condition=False)
                    if offer is not None:
                        return offer
            for give in needed:
                if give != get and hand.known[give] > target[give]:
                    offer = trial((give,), get)
                    if offer is not None:
                        return offer
            # Retain the reference's reused second-loop index between outer
            # iterations; changing it alters its candidate ordering.
            second = 0
            for first in unneeded:
                if not hand.known[first]:
                    continue
                for idx in range(second, len(unneeded)):
                    offer = trial((first, unneeded[idx]), get)
                    if offer is not None:
                        return offer
                for second_resource in needed:
                    if (
                        second_resource != get
                        and leftovers[first]
                        and leftovers[second_resource]
                        and (first != second_resource or leftovers[first] >= 2)
                    ):
                        offer = trial((first, second_resource), get)
                        if offer is not None:
                            return offer
                second = len(needed)
            second = 0
            for first in needed:
                if first == get or not leftovers[first]:
                    continue
                for idx in range(second, len(unneeded)):
                    other = unneeded[idx]
                    if leftovers[other]:
                        offer = trial((first, other), get)
                        if offer is not None:
                            return offer
                for other in needed:
                    if other != get and leftovers[other] >= (2 if first == other else 1):
                        offer = trial((first, other), get)
                        if offer is not None:
                            return offer
                second = len(needed)
        # Last stages request an otherwise unneeded resource that enables a
        # bank trade. Counteroffers additionally try requests of 2 and 3 cards.
        for count in ((1,) if original is None else (1, 2, 3)):
            start = len(unneeded) - 1
            while start >= 0:
                # makeOffer in the pinned source indexes neededRsrc here,
                # including its default-zero (unused type) padded entries.
                resource = (
                    unneeded[start]
                    if original is not None
                    else needed[start] if start < len(needed) else None
                )
                if resource is not None and available(resource):
                    break
                start -= 1
            for idx in range(start, -1, -1):
                get = unneeded[idx]
                for give in (*unneeded, *needed):
                    if give == get or not leftovers[give]:
                        continue
                    current = leftovers[:]
                    current[get] += count
                    current[give] -= 1
                    if bank_trade(current, target, periods, ratios) is not None:
                        offer = trial((give,), get, count, strict=True)
                        if offer is not None:
                            return offer
        return None
