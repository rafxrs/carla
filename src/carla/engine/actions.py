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
    # Domestic trade (protocol in `rules`): open, answer, conclude.
    PROPOSE_TRADE = 10
    ACCEPT_TRADE = 11
    REJECT_TRADE = 12
    CONFIRM_TRADE = 13
    CANCEL_TRADE = 14
    INVITE_TRADE = 15


NUM_ACTION_TYPES = len(ActionType)


class TradeOffer(NamedTuple):
    """Domestic trade terms from the side of the seat offering them: cards per resource it
    gives and wants."""

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
    PROPOSE_TRADE     trade (an offer in MAIN, a counteroffer in TRADE_RESPONSE)
    INVITE_TRADE      no fields; asks the other players for offers
    CONFIRM_TRADE     player = partner whose acceptance or counteroffer is taken
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
        for name in self._fields[1:-1]:
            value = getattr(self, name)
            if value != -1:
                out[name] = value
        if self.trade is not None:
            out["trade"] = {"give": list(self.trade.give), "receive": list(self.trade.receive)}
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
INVITE_TRADE = Action(ActionType.INVITE_TRADE)
_ONE_CARD = tuple(tuple(int(r == i) for r in range(NUM_RESOURCES)) for i in range(NUM_RESOURCES))
"""_ONE_CARD[r]: a single card of resource r, as counts per resource."""
ONE_FOR_ONE = tuple(
    tuple(
        (
            Action(ActionType.PROPOSE_TRADE, trade=TradeOffer(_ONE_CARD[g], _ONE_CARD[r]))
            if g != r
            else None
        )
        for r in range(NUM_RESOURCES)
    )
    for g in range(NUM_RESOURCES)
)
"""ONE_FOR_ONE[give][receive]: the offers `legal_actions` enumerates; None on the diagonal. Any
other valid offer is legal too (`rules.legal_trade_offer`)."""
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
    *(a for row in ONE_FOR_ONE for a in row if a is not None),
    ACCEPT_TRADE,
    REJECT_TRADE,
    CANCEL_TRADE,
    INVITE_TRADE,
    *CONFIRM_TRADE_WITH,
)
"""Every fixed action, including the one-for-one subset of combinatorial trade offers."""
ACTION_INDEX: dict[Action, int] = {a: i for i, a in enumerate(ALL_ACTIONS)}
