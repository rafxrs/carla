"""Classic-road path and award search from SOCPlayer and SOCRobotDM.

Derived from JSettlers 2.6.10, GPL-3.0-or-later. Road-node insertion order is
explicit because it affects which equal-length path the Java robot retains.
"""

from dataclasses import dataclass

from carla.engine.constants import ROAD_COST
from carla.engine.topology import EDGE_CORNERS

from .coordinates import NODES
from .estimator import (
    ZERO,
    estimates,
    periods_for_corners,
    player_corners,
    ratios_for_corners,
    rolls,
)
from .fast import AwardPlan
from .trackers import Candidate, PieceType

_NODE_IDS = {coord: i for i, coord in enumerate(NODES)}
_EDGE_IDS = {frozenset(pair): edge for edge, pair in enumerate(EDGE_CORNERS)}


def adjacent_nodes(corner):
    coord = NODES[corner]
    offsets = (-17, 17, 15 if (coord >> 4) % 2 == 0 else -15)
    return tuple(_NODE_IDS[coord + delta] for delta in offsets if coord + delta in _NODE_IDS)


@dataclass(frozen=True)
class RoadPath:
    beginning: int
    end: int
    edges: frozenset[int]

    @property
    def length(self):
        return len(self.edges)


def longest_paths(state, seat, node_order=None):
    if node_order is None:
        node_order = sorted(
            {
                c
                for e, owner in enumerate(state.edge_owner)
                if owner == seat
                for c in EDGE_CORNERS[e]
            },
            key=NODES.__getitem__,
        )
    retained = []
    for start in node_order:
        pending = [(start, frozenset())]
        while pending:
            corner, visited = pending.pop()
            choices = []
            blocked = bool(visited) and state.corner_owner[corner] not in (-1, seat)
            if not blocked:
                for other in reversed(adjacent_nodes(corner)):
                    edge = _EDGE_IDS[frozenset((corner, other))]
                    if state.edge_owner[edge] == seat and edge not in visited:
                        choices.append((other, visited | {edge}))
            if choices:
                pending.extend(choices)
                continue
            add = True
            trash = []
            for old in retained:
                if old.edges & visited:
                    if old.length < len(visited):
                        trash.append(old)
                    else:
                        add = False
            retained = [old for old in retained if old not in trash]
            if add:
                retained.append(RoadPath(start, corner, visited))
    return tuple(retained)


def extension(state, seat, paths, start, path_length, target, depth):
    """SOCRobotDM.recalcLongestRoadETAAux's path-returning branch."""
    longest, num_roads, best = 0, 500, None
    pending = [(start, path_length, ())]
    while pending:
        corner, length, visited = pending.pop()
        end = length > 0 and state.corner_owner[corner] not in (-1, seat)
        if not end:
            for path in paths:
                if start not in (path.beginning, path.end) and corner in (path.beginning, path.end):
                    end = True
                    length += path.length
                    break
        end = end or len(visited) >= depth
        choices = []
        if not end:
            for other in adjacent_nodes(corner):
                edge = _EDGE_IDS[frozenset((corner, other))]
                if state.edge_owner[edge] < 0 and edge not in visited:
                    choices.append((other, path_length + len(visited) + 1, (*visited, edge)))
        if choices:
            pending.extend(choices)
        elif length > longest or (length == longest and path_length + len(visited) < num_roads):
            longest, num_roads, best = length, len(visited), visited
    return best if longest > target and best else None


def longest_road_plan(state, seat, *, node_order=None):
    holder = state.longest_road_owner
    if holder == seat:
        return AwardPlan()
    paths = longest_paths(state, seat, node_order)
    target = max(4, state.road_lengths[seat]) if holder < 0 else state.road_lengths[holder]
    best = None
    for path in paths:
        depth = min(target + 1 - path.length, state.roads_left[seat])
        for start in (path.beginning, path.end):
            candidate = extension(state, seat, paths, start, path.length, target, depth)
            if candidate is not None and (best is None or len(candidate) < len(best)):
                best = candidate
    if best is None:
        return AwardPlan()
    corners = player_corners(state, seat)
    cost = tuple(len(best) * count for count in ROAD_COST)
    eta = rolls(
        state.hands[seat],
        cost,
        periods_for_corners(state.board, corners),
        ratios_for_corners(state.board, corners),
    )
    return AwardPlan(eta, tuple(Candidate(PieceType.ROAD, e) for e in best))


def knights_to_buy(state, seat):
    """Knights still to buy for Largest Army (OLD + NEW knights count, OLD ones are usable)."""
    holder = state.largest_army_owner
    target = 3 if holder < 0 else 0 if holder == seat else state.dev_played[holder][0] + 1
    if state.dev_played[seat][0] + state.dev_hands[seat][0] >= target:
        return 0
    old = state.dev_hands[seat][0] - (
        state.dev_bought_this_turn[0] if state.current_player == seat else 0
    )
    return target - state.dev_played[seat][0] - old


def award_estimates(state, seat, development_remaining, node_order=None):
    """SOCPlayerTracker's from-nothing award estimates and required counts."""
    corners = player_corners(state, seat)
    periods = periods_for_corners(state.board, corners)
    ratios = ratios_for_corners(state.board, corners)
    holder = state.largest_army_owner
    knights = knights_to_buy(state, seat)
    army_eta = (
        0
        if holder == seat
        else (
            (estimates(ZERO, periods, ratios, include_ship=True)[3] + 1) * knights
            if development_remaining >= knights
            else 500
        )
    )
    plan = longest_road_plan(state, seat, node_order=node_order)
    roads = 0 if state.longest_road_owner == seat else len(plan.pieces) if plan.pieces else 500
    road_eta = roads * rolls(ZERO, ROAD_COST, periods, ratios, cutoff=500)
    return army_eta, road_eta, knights, roads
