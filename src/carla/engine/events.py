"""Events: what every player observes when an action is applied (the public event stream)."""

from enum import IntEnum
from typing import NamedTuple


class EventKind(IntEnum):
    """Field usage per kind (unused fields keep their defaults).

    DICE_ROLLED            player, value=sum, counts=(die1, die2)
    RESOURCES_PRODUCED     player, counts=gain per resource
    STARTING_RESOURCES     player, counts=gain per resource (second setup settlement)
    SETTLEMENT_BUILT       player, location=corner
    CITY_BUILT             player, location=corner
    ROAD_BUILT             player, location=edge
    DEV_CARD_BOUGHT        player, value=DevCard (private to player)
    DEV_CARD_PLAYED        player, value=DevCard
    ROBBER_MOVED           player, location=tile
    RESOURCE_STOLEN        player=thief, target=victim, value=resource (private to both)
    DISCARDED              player, value=resource
    MARITIME_TRADED        player, counts=hand change per resource (negative = given)
    YEAR_OF_PLENTY_TAKEN   player, counts=gain per resource
    MONOPOLY_TAKEN         player, value=resource, counts=cards taken from each seat
    LONGEST_ROAD_CHANGED   player=new holder or -1, target=previous holder or -1
    LARGEST_ARMY_CHANGED   player=new holder, target=previous holder or -1
    TURN_ENDED             player
    GAME_OVER              player=winner, or -1 when truncated
    TRADE_*                reserved for domestic trade (not implemented yet)
    """

    DICE_ROLLED = 0
    RESOURCES_PRODUCED = 1
    STARTING_RESOURCES = 2
    SETTLEMENT_BUILT = 3
    CITY_BUILT = 4
    ROAD_BUILT = 5
    DEV_CARD_BOUGHT = 6
    DEV_CARD_PLAYED = 7
    ROBBER_MOVED = 8
    RESOURCE_STOLEN = 9
    DISCARDED = 10
    MARITIME_TRADED = 11
    YEAR_OF_PLENTY_TAKEN = 12
    MONOPOLY_TAKEN = 13
    LONGEST_ROAD_CHANGED = 14
    LARGEST_ARMY_CHANGED = 15
    TURN_ENDED = 16
    GAME_OVER = 17
    TRADE_PROPOSED = 18
    TRADE_RESPONDED = 19
    TRADE_EXECUTED = 20
    TRADE_CANCELLED = 21


NUM_EVENT_KINDS = len(EventKind)

PRIVATE_KINDS = frozenset({EventKind.DEV_CARD_BOUGHT, EventKind.RESOURCE_STOLEN})
"""Kinds whose `value` only the involved players (player, target) may see."""


class Event(NamedTuple):
    kind: EventKind
    player: int = -1
    target: int = -1
    value: int = -1
    location: int = -1
    counts: tuple[int, ...] = ()

    def visible_to(self, viewer: int) -> "Event":
        """The event as seen by seat `viewer` (hidden values replaced by -1)."""
        if self.kind in PRIVATE_KINDS and viewer != self.player and viewer != self.target:
            return self._replace(value=-1)
        return self
