"""Actions: one complete decision of the deciding player, laid out as factored policy heads.

All enumerable actions are pre-built singletons (`ROAD_AT[e]`, `ROBBER_TO[t][v + 1]`, ...), so
legal-action generation never allocates and `ACTION_INDEX` gives a stable flat index.
"""

from enum import IntEnum
from typing import Any, NamedTuple

from carla.engine.constants import NUM_PLAYERS, NUM_RESOURCES, DevCard
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, NUM_TILES


class ActionType(IntEnum):
    ROLL_DICE = 0
    END_TURN = 1
    BUILD_SETTLEMENT = 2
    BUILD_CITY = 3
    BUILD_ROAD = 4
    BUY_DEV_CARD = 5
    PLAY_DEV_CARD = 6
    MOVE_ROBBER = 7
    DISCARD = 8
    MARITIME_TRADE = 9
    # Domestic trade (interface only, see rules.py): propose -> others respond -> confirm/cancel.
    PROPOSE_TRADE = 10
    ACCEPT_TRADE = 11
    REJECT_TRADE = 12
    CONFIRM_TRADE = 13
    CANCEL_TRADE = 14


NUM_ACTION_TYPES = len(ActionType)


class TradeOffer(NamedTuple):
    """Domestic trade proposal: per-resource counts the proposer gives and wants."""

    give: tuple[int, ...]
    receive: tuple[int, ...]


class Action(NamedTuple):
    """A complete decision. Fields mirror the policy heads; unused fields stay -1/None.

    ================  =====================================================================
    type              fields used
    ================  =====================================================================
    BUILD_SETTLEMENT  corner
    BUILD_CITY        corner
    BUILD_ROAD        edge (also the free roads of setup and Road Building)
    PLAY_DEV_CARD     dev; Year of Plenty: resource <= resource2; Monopoly: resource
    MOVE_ROBBER       tile, player = victim to steal from (-1 when nobody can be robbed)
    DISCARD           resource (one card per decision)
    MARITIME_TRADE    resource = given type, resource2 = received type (ratio from ports)
    PROPOSE_TRADE     trade
    CONFIRM_TRADE     player = accepting partner
    ================  =====================================================================

    `player` is an absolute seat; environments convert it to a seat relative to the actor.
    """

    type: ActionType
    corner: int = -1
    edge: int = -1
    tile: int = -1
    player: int = -1
    dev: int = -1
    resource: int = -1
    resource2: int = -1
    trade: TradeOffer | None = None

    def to_json(self) -> dict[str, Any]:
        """Compact JSON form: the type name plus the fields in use."""
        out: dict[str, Any] = {"type": self.type.name}
        for name, value in zip(self._fields[1:], self[1:], strict=True):
            if name == "trade":
                if value is not None:
                    out["trade"] = {"give": list(value.give), "receive": list(value.receive)}
            elif value != -1:
                out[name] = value
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Action":
        kwargs = dict(data)
        kind = ActionType[kwargs.pop("type")]
        if "trade" in kwargs:
            t = kwargs["trade"]
            kwargs["trade"] = TradeOffer(tuple(t["give"]), tuple(t["receive"]))
        return cls(kind, **kwargs)


ROLL = Action(ActionType.ROLL_DICE)
END_TURN = Action(ActionType.END_TURN)
BUY_DEV = Action(ActionType.BUY_DEV_CARD)
SETTLEMENT_AT = tuple(Action(ActionType.BUILD_SETTLEMENT, corner=c) for c in range(NUM_CORNERS))
CITY_AT = tuple(Action(ActionType.BUILD_CITY, corner=c) for c in range(NUM_CORNERS))
ROAD_AT = tuple(Action(ActionType.BUILD_ROAD, edge=e) for e in range(NUM_EDGES))
ROBBER_TO = tuple(
    tuple(Action(ActionType.MOVE_ROBBER, tile=t, player=v) for v in range(-1, NUM_PLAYERS))
    for t in range(NUM_TILES)
)
"""ROBBER_TO[tile][victim + 1]; index 0 means no steal."""
DISCARD_OF = tuple(Action(ActionType.DISCARD, resource=r) for r in range(NUM_RESOURCES))
MARITIME = tuple(
    tuple(
        Action(ActionType.MARITIME_TRADE, resource=g, resource2=r) if g != r else None
        for r in range(NUM_RESOURCES)
    )
    for g in range(NUM_RESOURCES)
)
"""MARITIME[give][receive]; None on the diagonal."""
PLAY_KNIGHT = Action(ActionType.PLAY_DEV_CARD, dev=DevCard.KNIGHT)
PLAY_ROAD_BUILDING = Action(ActionType.PLAY_DEV_CARD, dev=DevCard.ROAD_BUILDING)
_YOP = {
    (a, b): Action(ActionType.PLAY_DEV_CARD, dev=DevCard.YEAR_OF_PLENTY, resource=a, resource2=b)
    for a in range(NUM_RESOURCES)
    for b in range(a, NUM_RESOURCES)
}
YEAR_OF_PLENTY = tuple(
    tuple(_YOP[min(a, b), max(a, b)] for b in range(NUM_RESOURCES)) for a in range(NUM_RESOURCES)
)
"""YEAR_OF_PLENTY[a][b], symmetric: both orders give the same (min, max) action object."""
MONOPOLY_ON = tuple(
    Action(ActionType.PLAY_DEV_CARD, dev=DevCard.MONOPOLY, resource=r) for r in range(NUM_RESOURCES)
)
ACCEPT_TRADE = Action(ActionType.ACCEPT_TRADE)
REJECT_TRADE = Action(ActionType.REJECT_TRADE)
CANCEL_TRADE = Action(ActionType.CANCEL_TRADE)
CONFIRM_TRADE_WITH = tuple(Action(ActionType.CONFIRM_TRADE, player=p) for p in range(NUM_PLAYERS))

ALL_ACTIONS: tuple[Action, ...] = (
    ROLL,
    END_TURN,
    BUY_DEV,
    *SETTLEMENT_AT,
    *CITY_AT,
    *ROAD_AT,
    *(a for row in ROBBER_TO for a in row),
    *DISCARD_OF,
    *(a for row in MARITIME for a in row if a is not None),
    PLAY_KNIGHT,
    PLAY_ROAD_BUILDING,
    *(YEAR_OF_PLENTY[a][b] for a in range(NUM_RESOURCES) for b in range(a, NUM_RESOURCES)),
    *MONOPOLY_ON,
    ACCEPT_TRADE,
    REJECT_TRADE,
    CANCEL_TRADE,
    *CONFIRM_TRADE_WITH,
)
"""Every action except PROPOSE_TRADE (whose offers are combinatorial)."""
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ALL_ACTIONS)}
