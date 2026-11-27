"""Production/port helpers around the translated fast building estimator.

Derived from JSettlers 2.6.10, GPL-3.0-or-later. Keep rounded probabilities,
Java float arithmetic, resource ordering and cutoff behavior intentionally.
"""

import math
from functools import lru_cache

from carla.engine.topology import CORNER_TILES
from carla.planning.jsettlers_eta import COSTS, PROBABILITIES, estimate, f32

PERCENT = (0, 0, 3, 6, 8, 11, 14, 17, 14, 11, 8, 6, 3)
ZERO = (0, 0, 0, 0, 0)
ALL_COSTS = (*COSTS, (1, 0, 1, 0, 0))  # Java's speedup includes SHIP on a land board.


def periods_for_corners(board, corners, blocked=-1):
    probabilities = [0.0] * 5
    for corner in corners:
        for tile in CORNER_TILES[corner]:
            resource = board.tile_resource[tile]
            if resource >= 0 and tile != blocked:
                probabilities[resource] = f32(
                    probabilities[resource] + f32(PROBABILITIES[board.tile_number[tile]])
                )
    return tuple(math.floor(f32(1 / p) + 0.5) if p else 55555 for p in probabilities)


def player_corners(state, seat):
    return tuple(
        corner
        for corner, owner in enumerate(state.corner_owner)
        if owner == seat
        for _ in range(state.corner_building[corner])
    )


def ratios_for_corners(board, corners):
    ports = {board.corner_port[corner] for corner in corners}
    return tuple(2 if r in ports else 3 if 5 in ports else 4 for r in range(5))


def estimates(hand, periods, ratios, cutoff=40, *, include_ship=False):
    return _estimates(tuple(hand), tuple(periods), tuple(ratios), cutoff, include_ship)


@lru_cache(maxsize=16384)
def _estimates(hand, periods, ratios, cutoff, include_ship):
    costs = ALL_COSTS if include_ship else COSTS
    result = [cutoff] * len(costs)
    try:
        for i, cost in enumerate(costs):
            result[i] = estimate(hand, cost, periods, ratios, cutoff)[0]
    except TimeoutError:
        pass
    return tuple(result)


def rolls(hand, target, periods, ratios, cutoff=100):
    return _rolls(tuple(hand), tuple(target), tuple(periods), tuple(ratios), cutoff)


@lru_cache(maxsize=16384)
def _rolls(hand, target, periods, ratios, cutoff):
    try:
        return estimate(hand, target, periods, ratios, cutoff)[0]
    except TimeoutError:
        return cutoff


def city_speedup(state, seat, corner):
    corners = player_corners(state, seat)
    old = estimates(
        ZERO,
        periods_for_corners(state.board, corners),
        ratios_for_corners(state.board, corners),
        include_ship=True,
    )
    new_corners = (*corners, corner)
    new = estimates(
        ZERO,
        periods_for_corners(state.board, new_corners),
        ratios_for_corners(state.board, corners),
        include_ship=True,
    )
    return tuple(a - b for a, b in zip(old, new, strict=True))
