"""Native port of JSettlers 2.6.10 SOCBuildingSpeedEstimate's fast algorithm.

Derived from JSettlers (Robert S. Thomas, Jeremy D. Monin et al.), GPL-3.0-or-later.
Resource iteration follows Java's clay, ore, sheep, wheat, wood ordering. The
rounded probabilities and float32 arithmetic are intentional reference semantics.
"""

import math
import struct

from carla.engine.constants import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST
from carla.engine.rules import trade_ratios
from carla.engine.topology import CORNER_TILES

ORDER = (1, 4, 2, 3, 0)
PROBABILITIES = (0, 0, 0.03, 0.06, 0.08, 0.11, 0.14, 0.17, 0.14, 0.11, 0.08, 0.06, 0.03)
COSTS = (ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def rolls_per_resource(state, seat, *, robber=False):
    probabilities = [0.0] * 5
    for corner, owner in enumerate(state.corner_owner):
        if owner != seat:
            continue
        for _ in range(state.corner_building[corner]):
            for tile in CORNER_TILES[corner]:
                resource = state.board.tile_resource[tile]
                if resource >= 0 and (not robber or tile != state.robber_tile):
                    probabilities[resource] = f32(
                        probabilities[resource] + f32(PROBABILITIES[state.board.tile_number[tile]])
                    )
    return tuple(math.floor(f32(1 / p) + 0.5) if p else 55555 for p in probabilities)


def estimate(start, target, periods, ratios, cutoff=40):
    """Return (rolls, resources), or raise TimeoutError at the reference cutoff."""
    hand = list(start)

    def satisfied():
        return all(h >= t for h, t in zip(hand, target, strict=True))

    def trade():
        for give in ORDER:
            for _ in range(max(0, (hand[give] - target[give]) // ratios[give])):
                needed = [r for r in ORDER if hand[r] < target[r]]
                if needed:
                    get = max(needed, key=lambda r: periods[r])
                    hand[get] += 1
                    hand[give] -= ratios[give]
                if satisfied():
                    return

    rolls = 0
    if not satisfied():
        trade()
    while not satisfied():
        rolls += 1
        if rolls > cutoff:
            raise TimeoutError("JSettlers ETA cutoff")
        for resource in ORDER:
            if periods[resource] == 0 or rolls % periods[resource] == 0:
                hand[resource] += 1
        if not satisfied():
            trade()
    return rolls, tuple(hand)


def building_estimates(state, seat):
    """Match getEstimatesFromNowFast, including its shared catch/cutoff behaviour."""
    result = [40] * 4
    periods = rolls_per_resource(state, seat)
    ratios = trade_ratios(state, seat)
    try:
        for i, cost in enumerate(COSTS):
            result[i] = estimate(state.hands[seat], cost, periods, ratios)[0]
    except TimeoutError:
        pass
    return tuple(result)
