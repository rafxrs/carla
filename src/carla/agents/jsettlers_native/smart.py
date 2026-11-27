"""Numerical SMART scoring kernels from SOCRobotDM, JSettlers 2.6.10.

Robert S. Thomas, Jeremy D. Monin et al., GPL-3.0-or-later. These kernels take
win-time estimates as inputs. Projection and hypothetical tracker updates live
in projection.py and planner.py and are validated independently.
"""

import math

from carla.planning.jsettlers_eta import f32

THREAT_MULTIPLIER = 1.0
"""SOCRobotParameters.threatMultiplier of the stock robots (SOCServer ROBOT_PARAMS_*)."""


def eta_bonus(eta: int, bonus: float, factor=0.13):
    return f32(f32(bonus) / f32(math.pow(f32(1 + f32(factor)), eta)))


def win_time_bonus(before, after, seat, *, adversarial=1.0, leader_adversarial=1.0):
    """calcWGETABonusAux, including the original tied-leader loop semantics."""
    difference = [old - new for old, new in zip(before, after, strict=True)]
    best = min(1000, *before)
    leaders = [p for p, eta in enumerate(before) if eta == best]
    bonus = 25.0 if after[seat] == 0 else 0.0
    if before[seat] > 0 and bonus == 0:
        bonus = f32(25 * f32(f32(difference[seat]) / f32(before[seat])))
    # The original applies this loop once PER leader, not once for the union
    # of nonleaders. Preserve that behavior rather than silently improving it.
    for player in range(4):
        for leader in leaders:
            if player in (seat, leader):
                continue
            if before[player] > 0:
                term = f32(-25 * f32(adversarial))
                term = f32(term * f32(f32(difference[player]) / f32(before[player])))
                term = f32(term * f32(f32(best) / f32(before[player])))
                bonus = f32(bonus + term)
            elif difference[player] < 0:
                bonus = f32(bonus + f32(25 * f32(adversarial)))
    for leader in leaders:
        if leader == seat:
            continue
        if before[leader] > 0:
            term = f32(-25 * f32(leader_adversarial))
            term = f32(term * f32(f32(difference[leader]) / f32(before[leader])))
            bonus = f32(bonus + term)
        elif difference[leader] < 0:
            bonus = f32(bonus + f32(25 * f32(leader_adversarial)))
    return bonus
