"""Canonical offer tokens: sorted give cards, SWITCH, sorted receive cards, STOP.

Both sides are from the offer maker's perspective. PAD belongs only to fixed-width
policy actions; it is never part of a `TradeOffer` or the codec's output.
"""

from collections.abc import Sequence

from carla.engine.actions import TradeOffer
from carla.engine.constants import NUM_RESOURCES

SWITCH = NUM_RESOURCES
STOP = NUM_RESOURCES + 1
PAD = -1


def offer_tokens(offer: TradeOffer, max_cards: int) -> tuple[int, ...]:
    """One sorted sequence per count-vector offer, including both delimiters."""
    if len(offer.give) != NUM_RESOURCES or len(offer.receive) != NUM_RESOURCES:
        raise ValueError("trade vectors must have five resources")
    if any(type(n) is not int or n < 0 for n in (*offer.give, *offer.receive)):
        raise ValueError("trade counts must be nonnegative integers")
    if not 1 <= sum(offer.give) <= max_cards or not 1 <= sum(offer.receive) <= max_cards:
        raise ValueError("both trade sides must contain cards within the decoder cap")
    if any(g and r for g, r in zip(offer.give, offer.receive, strict=True)):
        raise ValueError("the same resource cannot appear on both sides")
    return (
        *(i for i, count in enumerate(offer.give) for _ in range(count)),
        SWITCH,
        *(i for i, count in enumerate(offer.receive) for _ in range(count)),
        STOP,
    )


def tokens_offer(tokens: Sequence[int]) -> TradeOffer:
    """Reject malformed, noncanonical, or trailing token data."""
    if any(type(t) is not int for t in tokens):
        raise ValueError("trade tokens must be integers")
    values = list(tokens)
    if not values or values[-1] != STOP or values.count(SWITCH) != 1 or values.count(STOP) != 1:
        raise ValueError("offer needs one switch and one final stop")
    split = values.index(SWITCH)
    sides = (values[:split], values[split + 1 : -1])
    if any(not side or side != sorted(side) for side in sides):
        raise ValueError("trade sides must be nonempty and sorted")
    if any(t < 0 or t >= NUM_RESOURCES for side in sides for t in side):
        raise ValueError("invalid resource token")
    give = tuple(sides[0].count(i) for i in range(NUM_RESOURCES))
    receive = tuple(sides[1].count(i) for i in range(NUM_RESOURCES))
    if any(a and b for a, b in zip(give, receive, strict=True)):
        raise ValueError("the same resource cannot appear on both sides")
    return TradeOffer(give, receive)
