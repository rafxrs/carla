"""Translations of the independent JSettlers base-game resource strategies.

Derived from DiscardStrategy, MonopolyStrategy, SOCRobotDM and
SOCRobotNegotiator, JSettlers 2.6.10, GPL-3.0-or-later.
"""

from carla.engine.actions import Action, ActionType
from carla.planning.jsettlers_eta import ORDER


def discard(hand, target, periods, count):
    """Translate the original ordering, including its rare-resource-first sort."""
    if not 0 <= count <= sum(hand):
        raise ValueError("invalid discard count")
    leftovers = [max(0, h - t) for h, t in zip(hand, target, strict=True)]
    needed = [h - left for h, left in zip(hand, leftovers, strict=True)]
    result = [0] * 5
    # The Java estimator factory initializes its stable sort in resource order.
    # Despite DiscardStrategy's "easiest to hardest" comment, the called
    # getRollsForResourcesSorted actually sorts DESCENDING in release 2.6.10.
    order = sorted(ORDER, key=lambda r: -periods[r])
    for source in (leftovers, needed):
        for resource in order:
            amount = min(source[resource], count - sum(result))
            result[resource] += amount
    return tuple(result)


def monopoly(known_hands, seat, ratios):
    best_count, choice = 0, -1
    for resource in ORDER:
        count = sum(hand[resource] for p, hand in enumerate(known_hands) if p != seat)
        count //= ratios[resource]
        if count > best_count:
            best_count, choice = count, resource
    return choice if best_count > 2 else None


def free_resources(hand, target, periods, count=2):
    """SOCRobotDM.chooseFreeResources; returns a possibly incomplete choice."""
    result, current = [0] * 5, list(hand)
    for _ in range(count):
        needed = [r for r in ORDER if current[r] < target[r]]
        if not needed:
            break
        resource = max(needed, key=lambda r: periods[r])
        result[resource] += 1
        current[resource] += 1
    return tuple(result)


def bank_trade(hand, target, periods, ratios):
    """SOCRobotNegotiator.getOfferToBank, preserving stable ordering."""
    if all(h >= t for h, t in zip(hand, target, strict=True)):
        return None
    needed = sorted((r for r in ORDER if target[r]), key=lambda r: periods[r])
    unneeded = sorted((r for r in ORDER if not target[r]), key=lambda r: periods[r])
    receive = next(r for r in reversed(needed) if hand[r] < target[r])
    for give in unneeded:
        if hand[give] >= ratios[give]:
            return Action(ActionType.MARITIME_TRADE, resource=give, resource2=receive)
    for give in needed:
        available = hand[give]
        if periods[give] >= periods[receive]:
            available -= target[give]
        if available >= ratios[give]:
            return Action(ActionType.MARITIME_TRADE, resource=give, resource2=receive)
    return None


def knight_for_army(state, seat):
    holder = state.largest_army_owner
    if holder == seat:
        return False
    target = 3 if holder < 0 else state.dev_played[holder][0] + 1
    return state.dev_played[seat][0] + state.dev_hands[seat][0] >= target
