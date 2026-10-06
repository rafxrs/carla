"""Factored action space: policy heads, validity masks, and Action <-> head-index conversion.

A decision samples the `type` head, then the heads that type uses (`heads_used`). Seat heads are
relative to the actor: 0 is the actor itself, which for the robber means "steal from nobody".
Each mask below applies to the head(s) listed, once the earlier heads are fixed:

==============  ========  ===================  ==========================================
mask            shape     heads                used when
==============  ========  ===================  ==========================================
type            (16,)     type                 always
settlement      (54,)     corner               type == BUILD_SETTLEMENT
city            (54,)     corner               type == BUILD_CITY
road            (72,)     edge                 type == BUILD_ROAD
robber          (19, 4)   tile -> player       type == MOVE_ROBBER
dev             (5,)      dev                  type == PLAY_DEV_CARD
year_of_plenty  (5, 5)    resource -> resource2  dev == YEAR_OF_PLENTY (resource <= resource2)
monopoly        (5,)      resource             dev == MONOPOLY
discard         (5,)      resource             type == DISCARD
maritime        (5, 5)    resource -> resource2  type == MARITIME_TRADE (give -> receive)
partner         (4,)      player               type == CONFIRM_TRADE
trade           (5, 5)    resource -> resource2 type == PROPOSE_TRADE (1 card each)
==============  ========  ===================  ==========================================

The heads offer one-for-one trades. Larger and mixed offers are accepted by the engine and GUI;
the planned trade decoder can extend the policy beyond this baseline action subset.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from carla.engine import actions as A
from carla.engine.actions import NUM_ACTION_TYPES, Action, ActionType
from carla.engine.constants import NUM_DEV_CARD_TYPES, NUM_PLAYERS, NUM_RESOURCES, DevCard
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, NUM_TILES

HEAD_SIZES: dict[str, int] = {
    "type": NUM_ACTION_TYPES,
    "corner": NUM_CORNERS,
    "edge": NUM_EDGES,
    "tile": NUM_TILES,
    "player": NUM_PLAYERS,
    "dev": NUM_DEV_CARD_TYPES,
    "resource": NUM_RESOURCES,
    "resource2": NUM_RESOURCES,
}
HEAD_NAMES: tuple[str, ...] = tuple(HEAD_SIZES)
HEAD_INDEX = {name: i for i, name in enumerate(HEAD_NAMES)}

MASK_SHAPES: dict[str, tuple[int, ...]] = {
    "type": (NUM_ACTION_TYPES,),
    "settlement": (NUM_CORNERS,),
    "city": (NUM_CORNERS,),
    "road": (NUM_EDGES,),
    "robber": (NUM_TILES, NUM_PLAYERS),
    "dev": (NUM_DEV_CARD_TYPES,),
    "year_of_plenty": (NUM_RESOURCES, NUM_RESOURCES),
    "monopoly": (NUM_RESOURCES,),
    "discard": (NUM_RESOURCES,),
    "maritime": (NUM_RESOURCES, NUM_RESOURCES),
    "partner": (NUM_PLAYERS,),
    "trade": (NUM_RESOURCES, NUM_RESOURCES),
}

_T = ActionType


def relative_seat(seat: int, actor: int) -> int:
    return (seat - actor) % NUM_PLAYERS


def absolute_seat(relative: int, actor: int) -> int:
    return (relative + actor) % NUM_PLAYERS


def heads_used(action_type: int, dev: int = -1) -> tuple[str, ...]:
    """Heads (after `type`) that define an action of this type (and development card)."""
    if action_type in (_T.BUILD_SETTLEMENT, _T.BUILD_CITY):
        return ("corner",)
    if action_type == _T.BUILD_ROAD:
        return ("edge",)
    if action_type == _T.MOVE_ROBBER:
        return ("tile", "player")
    if action_type == _T.PLAY_DEV_CARD:
        if dev == DevCard.YEAR_OF_PLENTY:
            return ("dev", "resource", "resource2")
        if dev == DevCard.MONOPOLY:
            return ("dev", "resource")
        return ("dev",)
    if action_type == _T.DISCARD:
        return ("resource",)
    if action_type == _T.MARITIME_TRADE:
        return ("resource", "resource2")
    if action_type == _T.CONFIRM_TRADE:
        return ("player",)
    if action_type == _T.PROPOSE_TRADE:
        return ("resource", "resource2")
    return ()


def empty_masks() -> dict[str, np.ndarray]:
    return {k: np.zeros(shape, dtype=bool) for k, shape in MASK_SHAPES.items()}


def _one_for_one(offer: A.TradeOffer | None) -> tuple[int, int]:
    """Extract a baseline policy offer without silently dropping extra cards."""
    if (
        offer is None
        or len(offer.give) != NUM_RESOURCES
        or len(offer.receive) != NUM_RESOURCES
        or sum(offer.give) != 1
        or sum(offer.receive) != 1
        or any(type(n) is not int or n not in (0, 1) for n in (*offer.give, *offer.receive))
    ):
        raise ValueError("only one-for-one offers can be encoded as baseline policy heads")
    give, receive = offer.give.index(1), offer.receive.index(1)
    if give == receive:
        raise ValueError("a trade must exchange different resources")
    return give, receive


def build_masks(legal: Iterable[Action], actor: int) -> dict[str, np.ndarray]:
    """Masks (see module docstring) allowing exactly the `legal` actions of seat `actor`."""
    m = empty_masks()
    for a in legal:
        t = a.type
        m["type"][t] = True
        if t == _T.BUILD_SETTLEMENT:
            m["settlement"][a.corner] = True
        elif t == _T.BUILD_CITY:
            m["city"][a.corner] = True
        elif t == _T.BUILD_ROAD:
            m["road"][a.edge] = True
        elif t == _T.MOVE_ROBBER:
            m["robber"][a.tile, 0 if a.player < 0 else relative_seat(a.player, actor)] = True
        elif t == _T.PLAY_DEV_CARD:
            m["dev"][a.dev] = True
            if a.dev == DevCard.YEAR_OF_PLENTY:
                m["year_of_plenty"][a.resource, a.resource2] = True
            elif a.dev == DevCard.MONOPOLY:
                m["monopoly"][a.resource] = True
        elif t == _T.DISCARD:
            m["discard"][a.resource] = True
        elif t == _T.MARITIME_TRADE:
            m["maritime"][a.resource, a.resource2] = True
        elif t == _T.CONFIRM_TRADE:
            m["partner"][relative_seat(a.player, actor)] = True
        elif t == _T.PROPOSE_TRADE:
            give, receive = _one_for_one(a.trade)
            m["trade"][give, receive] = True
    return m


def sample_heads(masks: Mapping[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Uniform random legal decision as head indices: the reference semantics of the masks
    (model-side samplers must condition the heads exactly like this)."""
    heads = np.full(len(HEAD_NAMES), -1, dtype=np.int64)

    def pick(head: str, mask: Any) -> int:
        heads[HEAD_INDEX[head]] = value = int(rng.choice(np.flatnonzero(mask)))
        return value

    t = pick("type", masks["type"])
    if t == _T.BUILD_SETTLEMENT:
        pick("corner", masks["settlement"])
    elif t == _T.BUILD_CITY:
        pick("corner", masks["city"])
    elif t == _T.BUILD_ROAD:
        pick("edge", masks["road"])
    elif t == _T.MOVE_ROBBER:
        tile = pick("tile", masks["robber"].any(axis=1))
        pick("player", masks["robber"][tile])
    elif t == _T.PLAY_DEV_CARD:
        dev = pick("dev", masks["dev"])
        if dev == DevCard.YEAR_OF_PLENTY:
            first = pick("resource", masks["year_of_plenty"].any(axis=1))
            pick("resource2", masks["year_of_plenty"][first])
        elif dev == DevCard.MONOPOLY:
            pick("resource", masks["monopoly"])
    elif t == _T.DISCARD:
        pick("resource", masks["discard"])
    elif t == _T.MARITIME_TRADE:
        give = pick("resource", masks["maritime"].any(axis=1))
        pick("resource2", masks["maritime"][give])
    elif t == _T.CONFIRM_TRADE:
        pick("player", masks["partner"])
    elif t == _T.PROPOSE_TRADE:
        give = pick("resource", masks["trade"].any(axis=1))
        pick("resource2", masks["trade"][give])
    return heads


def encode_action(action: Action, actor: int) -> np.ndarray:
    """Head indices of `action` in HEAD_NAMES order; unused heads are -1."""
    heads = np.full(len(HEAD_NAMES), -1, dtype=np.int64)
    heads[0] = action.type
    if action.type == _T.PROPOSE_TRADE:
        give, receive = _one_for_one(action.trade)
        heads[HEAD_INDEX["resource"]] = give
        heads[HEAD_INDEX["resource2"]] = receive
        return heads
    for name in heads_used(action.type, action.dev):
        value = getattr(action, name)
        if name == "player":
            value = 0 if value < 0 else relative_seat(value, actor)
        heads[HEAD_INDEX[name]] = value
    return heads


def decode_action(heads: Sequence[int] | np.ndarray | Mapping[str, int], actor: int) -> Action:
    """Inverse of `encode_action`; heads not used by the chosen type are ignored."""
    if isinstance(heads, Mapping):
        get = heads.__getitem__
    else:
        values = [int(h) for h in heads]
        get = lambda name: values[HEAD_INDEX[name]]  # noqa: E731
    action_type = ActionType(get("type"))
    if action_type == _T.PROPOSE_TRADE:
        give, receive = int(get("resource")), int(get("resource2"))
        if not 0 <= give < NUM_RESOURCES or not 0 <= receive < NUM_RESOURCES or give == receive:
            raise ValueError("invalid one-for-one trade heads")
        offer = A.ONE_FOR_ONE[give][receive]
        assert offer is not None
        return offer
    dev = get("dev") if action_type == _T.PLAY_DEV_CARD else -1
    fields = {}
    for name in heads_used(action_type, dev):
        value = int(get(name))
        if name == "player":
            value = (
                -1 if value == 0 and action_type == _T.MOVE_ROBBER else absolute_seat(value, actor)
            )
        fields[name] = DevCard(value) if name == "dev" else value
    return Action(action_type, **fields)
