"""Base-game candidate graph used by the native JSettlers port.

Derived from SOCPlayerTracker/SOCPossiblePiece, JSettlers 2.6.10,
GPL-3.0-or-later. Candidate graphs are rebuilt from public placements; historical
equivalence with Java's incremental graph is checked separately.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from carla.engine.actions import Action, ActionType
from carla.engine.constants import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST
from carla.engine.events import EventKind
from carla.engine.rules import road_edges, settlement_corners
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS, EDGE_CORNERS

from .coordinates import EDGES, NODES, adjacent_edges
from .estimator import city_speedup, periods_for_corners, player_corners, ratios_for_corners, rolls
from .opening import legal_site


class PieceType(IntEnum):
    CARD = -2
    ROAD = 0
    SETTLEMENT = 1
    CITY = 2


COSTS = {
    PieceType.CARD: DEV_CARD_COST,
    PieceType.ROAD: ROAD_COST,
    PieceType.SETTLEMENT: SETTLEMENT_COST,
    PieceType.CITY: CITY_COST,
}


@dataclass
class Candidate:
    type: PieceType
    location: int
    necessary_roads: list[int] = field(default_factory=list)
    speedup: tuple[int, ...] = (0, 0, 0, 0, 0)
    eta: int = 0
    score: float = 0.0
    path: tuple[int, ...] = ()  # execution order, first road first
    conflicts: list[tuple[int, int]] = field(default_factory=list)
    threats: list[tuple[int, int, int]] = field(default_factory=list)
    number_needed: int = -1

    @property
    def cost(self):
        return COSTS[self.type]

    def action(self):
        if self.type == PieceType.CARD:
            return Action(ActionType.BUY_DEV_CARD)
        if self.type == PieceType.ROAD:
            return Action(ActionType.BUILD_ROAD, edge=self.location)
        return Action(
            ActionType.BUILD_CITY if self.type == PieceType.CITY else ActionType.BUILD_SETTLEMENT,
            corner=self.location,
        )


def _insertion_order(necessary, previous, discovered):
    """Prerequisites in SOCPossiblePiece's list order: the original appends a road when
    expansion first links it, so surviving entries keep their place and new ones follow in
    expansion order. The order decides which equal-length road path a robot keeps."""
    kept = [] if previous is None else [e for e in previous.necessary_roads if e in necessary]
    return kept + sorted((e for e in necessary if e not in kept), key=discovered.__getitem__)


def _incremental_count(previous, necessary, roads):
    """A surviving candidate's road count as SOCPlayerTracker's incremental updates leave it.

    Expansion only lowers it through new prerequisites (otherwise the count is kept, even when
    stale). removeDependents, which drops a prerequisite another player took, recomputes it as
    the smallest count over the remaining prerequisites, plus one.
    """
    if set(previous.necessary_roads) - set(necessary):
        return min(40, *(roads[e].number_needed + 1 for e in necessary))
    added = set(necessary) - set(previous.necessary_roads)
    if added:
        return min(previous.number_needed, *(roads[e].number_needed + 1 for e in added))
    return previous.number_needed


@dataclass
class Tracker:
    seat: int
    roads: dict[int, Candidate] = field(default_factory=dict)
    settlements: dict[int, Candidate] = field(default_factory=dict)
    cities: dict[int, Candidate] = field(default_factory=dict)
    road_nodes: list[int] = field(default_factory=list)
    win_eta: int = 0
    # SOCPlayerTracker's cached award estimates, with its constructor defaults. Only
    # `projection.recalc_awards` (recalcLongestRoadETA/recalcLargestArmyETA) changes them, so a
    # projection between refreshes reuses stale values exactly as the original robot does.
    longest_road_eta: int = 500
    roads_to_go: int = 20
    largest_army_eta: int = 500
    _signature: tuple | None = field(default=None, repr=False)

    def adopt_awards(self, other: "Tracker") -> None:
        """Keep `other`'s cached award estimates (a refresh the original made on this tracker)."""
        self.longest_road_eta = other.longest_road_eta
        self.roads_to_go = other.roads_to_go
        self.largest_army_eta = other.largest_army_eta

    def observe(self, events):
        for event in events:
            if event.kind == EventKind.ROAD_BUILT and event.player == self.seat:
                for corner in sorted(EDGE_CORNERS[event.location], key=NODES.__getitem__):
                    if corner not in self.road_nodes:
                        self.road_nodes.append(corner)

    def refresh(self, state):
        signature = (
            tuple(state.corner_owner),
            tuple(state.corner_building),
            tuple(state.edge_owner),
        )
        if self._signature == signature:
            return
        self._signature = signature
        previous_roads, previous_sets = self.roads, self.settlements
        self.roads, self.settlements, self.cities = {}, {}, {}
        # Snapshot initialization has a declared coordinate order. Normal play
        # supplies chronological placements through observe instead.
        for edge, owner in enumerate(state.edge_owner):
            if owner == self.seat:
                for corner in sorted(EDGE_CORNERS[edge], key=NODES.__getitem__):
                    if corner not in self.road_nodes:
                        self.road_nodes.append(corner)
        for corner, owner in enumerate(state.corner_owner):
            if owner == self.seat and state.corner_building[corner] == 1:
                self.cities[corner] = Candidate(
                    PieceType.CITY, corner, speedup=city_speedup(state, self.seat, corner)
                )
        # EXPAND_LEVEL=1: currently possible roads, plus one additional road.
        # Java also emits settlements adjacent to each of those two layers.
        roots = sorted(road_edges(state, self.seat), key=EDGES.__getitem__)
        paths: dict[int, list[tuple[int, ...]]] = {e: [(e,)] for e in roots}
        queue = deque((e, (e,)) for e in roots)
        while queue:
            edge, path = queue.popleft()
            if len(path) >= min(2, state.roads_left[self.seat]):
                continue
            # expandRoadOrShip visits neighbours in getAdjacentEdgesToEdge order.
            for neighbor in adjacent_edges(edge):
                (corner,) = set(EDGE_CORNERS[edge]) & set(EDGE_CORNERS[neighbor])
                if state.corner_owner[corner] not in (-1, self.seat):
                    continue
                if state.edge_owner[neighbor] >= 0 or neighbor in path or neighbor in roots:
                    continue
                next_path = (*path, neighbor)
                if next_path not in paths.setdefault(neighbor, []):
                    paths[neighbor].append(next_path)
                    queue.append((neighbor, next_path))
        # Expansion order: prerequisites first appear in this breadth-first order.
        discovered = {edge: i for i, edge in enumerate(paths)}
        for edge in sorted(paths, key=EDGES.__getitem__):
            alternatives = paths[edge]
            necessary = (
                []
                if edge in roots
                else _insertion_order(
                    dict.fromkeys(path[-2] for path in alternatives),
                    previous_roads.get(edge),
                    discovered,
                )
            )
            self.roads[edge] = Candidate(PieceType.ROAD, edge, necessary_roads=necessary)
        for edge in sorted(paths, key=lambda e: min(map(len, paths[e]))):
            candidate = self.roads[edge]
            previous = previous_roads.get(edge)
            if not candidate.necessary_roads:
                candidate.number_needed = 0
            elif previous is None:
                candidate.number_needed = min(
                    self.roads[e].number_needed + 1 for e in candidate.necessary_roads
                )
            else:
                candidate.number_needed = _incremental_count(
                    previous, candidate.necessary_roads, self.roads
                )
        immediate = set(settlement_corners(state, self.seat))
        for corner in range(54):
            if not legal_site(state, corner):
                continue
            necessary = _insertion_order(
                [e for e in CORNER_EDGES[corner] if e in paths],
                previous_sets.get(corner),
                discovered,
            )
            if corner in immediate:
                necessary = []
            elif not necessary:
                continue
            candidate = Candidate(PieceType.SETTLEMENT, corner, necessary_roads=necessary)
            previous = previous_sets.get(corner)
            if not necessary:
                candidate.number_needed = 0
            elif previous is None:
                candidate.number_needed = min(self.roads[e].number_needed + 1 for e in necessary)
            else:
                candidate.number_needed = _incremental_count(previous, necessary, self.roads)
            self.settlements[corner] = candidate
        self.cities = dict(sorted(self.cities.items(), key=lambda item: NODES[item[0]]))
        self.settlements = dict(sorted(self.settlements.items(), key=lambda item: NODES[item[0]]))

    def settlement_paths(self, state):
        corners = player_corners(state, self.seat)
        periods = periods_for_corners(state.board, corners)
        ratios = ratios_for_corners(state.board, corners)
        hand = state.hands[self.seat]
        for candidate in self.settlements.values():
            candidate.path = ()
            if candidate.necessary_roads:
                queue = deque((e, ()) for e in candidate.necessary_roads)
                for _ in range(50):
                    if not queue:
                        break
                    edge, following = queue.popleft()
                    road = self.roads[edge]
                    if not road.necessary_roads:
                        candidate.path = (edge, *following)
                        queue.clear()
                    elif len(queue) + len(road.necessary_roads) > 40:
                        candidate.eta = 500
                        break
                    else:
                        queue.extend((prior, (edge, *following)) for prior in road.necessary_roads)
                if queue or not candidate.path:
                    candidate.eta = 500
                    continue
            cost = tuple(SETTLEMENT_COST[r] + len(candidate.path) * ROAD_COST[r] for r in range(5))
            candidate.eta = rolls(hand, cost, periods, ratios)


def update_conflicts(trackers, state):
    for tracker in trackers:
        for candidate in tracker.settlements.values():
            candidate.conflicts = [
                (other.seat, corner)
                for other in trackers
                for corner in other.settlements
                if (other.seat, corner) != (tracker.seat, candidate.location)
                and corner in (candidate.location, *CORNER_NEIGHBORS[candidate.location])
            ]
        for road in tracker.roads.values():
            road.threats = [
                (other.seat, int(PieceType.ROAD), road.location)
                for other in trackers
                if other.seat != tracker.seat and road.location in other.roads
            ]
            supporting_nodes = set()
            if not road.necessary_roads:
                supporting_nodes = {
                    corner
                    for corner in EDGE_CORNERS[road.location]
                    if any(state.edge_owner[e] == tracker.seat for e in CORNER_EDGES[corner])
                }
            elif len(road.necessary_roads) == 1:
                supporting_nodes = set(EDGE_CORNERS[road.location]) & set(
                    EDGE_CORNERS[road.necessary_roads[0]]
                )
            road.threats.extend(
                (other.seat, int(PieceType.SETTLEMENT), corner)
                for corner in supporting_nodes
                for other in trackers
                if other.seat != tracker.seat and corner in other.settlements
            )
        # Root-road threats propagate through roads with a single prerequisite.
        for root in tracker.roads.values():
            if root.necessary_roads:
                continue
            pending, visited = [root.location], set()
            while pending:
                edge = pending.pop()
                if edge in visited:
                    continue
                visited.add(edge)
                for road in tracker.roads.values():
                    if edge in road.necessary_roads:
                        if road.necessary_roads == [edge]:
                            for threat in root.threats:
                                if threat not in road.threats:
                                    road.threats.append(threat)
                        pending.append(road.location)
        for settlement in tracker.settlements.values():
            threats = [
                (other.seat, int(PieceType.SETTLEMENT), settlement.location)
                for other in trackers
                if other.seat != tracker.seat and settlement.location in other.settlements
            ]
            if settlement.necessary_roads:
                first, *others = settlement.necessary_roads
                threats.extend(
                    threat
                    for threat in tracker.roads[first].threats
                    if all(threat in tracker.roads[e].threats for e in others)
                )
            settlement.threats = list(dict.fromkeys(threats))
