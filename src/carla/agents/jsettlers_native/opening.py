"""OpeningBuildStrategy translated from JSettlers 2.6.10.

Robert S. Thomas, Jeremy D. Monin et al., GPL-3.0-or-later. Callers supply
candidate order explicitly so collection-order differences can be diagnosed.
"""

from dataclasses import dataclass

from carla.engine.constants import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST
from carla.engine.topology import CORNER_NEIGHBORS, CORNER_TILES, EDGE_CORNERS
from carla.planning.jsettlers_eta import ORDER, estimate

from .coordinates import EDGES, NODES
from .estimator import PERCENT, ZERO, periods_for_corners, player_corners, ratios_for_corners, rolls

OPENING_COSTS = (SETTLEMENT_COST, CITY_COST, DEV_CARD_COST, ROAD_COST)


def probability(board, corners):
    return sum(
        PERCENT[board.tile_number[tile]] for corner in corners for tile in CORNER_TILES[corner]
    )


def pair_speed(board, corners, cutoff):
    periods = periods_for_corners(board, corners)
    ratios = ratios_for_corners(board, corners)
    speed, complete = 0, False
    try:
        for i, cost in enumerate(OPENING_COSTS):
            speed += estimate(ZERO, cost, periods, ratios, cutoff)[0]
            if i == 3:
                complete = True
            elif speed >= cutoff:
                break
    except TimeoutError:
        return cutoff, False
    return speed, complete


def single_speed(board, corner, cutoff):
    periods = periods_for_corners(board, (corner,))
    ratios = ratios_for_corners(board, (corner,))
    return sum(rolls(ZERO, cost, periods, ratios, cutoff) for cost in OPENING_COSTS)


def legal_site(state, corner):
    return state.corner_owner[corner] < 0 and all(
        state.corner_owner[n] < 0 for n in CORNER_NEIGHBORS[corner]
    )


def two_away(corner):
    return {
        target
        for adjacent in CORNER_NEIGHBORS[corner]
        for target in CORNER_NEIGHBORS[adjacent]
        if target != corner and target not in CORNER_NEIGHBORS[corner]
    }


def rarity(board):
    result = [0] * 5
    for resource, number in zip(board.tile_resource, board.tile_number, strict=True):
        if resource >= 0:
            result[resource] += PERCENT[number]
    return tuple(result)


def number_scores(board, candidates, weight, owned_corners=None):
    numbers = (
        None
        if owned_corners is None
        else {board.tile_number[t] for c in owned_corners for t in CORNER_TILES[c]}
    )
    result = {}
    for corner in candidates:
        score = 0
        for tile in CORNER_TILES[corner]:
            number = board.tile_number[tile]
            score += PERCENT[number]
            if number and numbers is not None and number not in numbers:
                score += PERCENT[number]
        result[corner] = (score * 100 // (40 if numbers is None else 80)) * weight
    return result


def port_scores(board, scores, generic_weight, specific_weight, owned_ports=(), *, distant=False):
    resource_rarity = rarity(board)
    for port_type in (5, *ORDER):
        if port_type in owned_ports:
            continue
        if port_type != 5 and resource_rarity[port_type] <= 33:
            continue
        weight = (
            generic_weight if port_type == 5 else resource_rarity[port_type] * specific_weight // 56
        )
        targets = [c for c, port in enumerate(board.corner_port) if port == port_type]
        for corner in scores:
            if distant:
                # Preserve original loop behavior: encountering the candidate
                # itself breaks the scan without clearing an earlier score.
                score = 0
                for target in targets:
                    if corner == target:
                        break
                    if target in two_away(corner):
                        score = 100
            else:
                score = 100 if corner in targets else 0
            scores[corner] += score * weight


@dataclass
class Opening:
    first: int = -1
    second: int = -1
    road_destination: int = -1

    def first_settlement(self, board, candidates):
        candidates = tuple(candidates)
        best_speed, best_probability = 160, 0
        self.first = self.second = -1
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
                if second in CORNER_NEIGHBORS[first]:
                    continue
                speed, complete = pair_speed(board, (first, second), best_speed)
                prob = probability(board, (first, second))
                if speed < best_speed or (
                    speed == best_speed and complete and prob > best_probability
                ):
                    self.first, self.second = first, second
                    best_speed, best_probability = speed, prob
        if self.first < 0:
            raise ValueError("no opening pair in supported board")
        if single_speed(board, self.first, 100) > single_speed(board, self.second, best_speed):
            self.first, self.second = self.second, self.first
        return self.first

    def second_settlement(self, board, candidates):
        if self.first < 0:
            raise ValueError("first settlement is required before second settlement")
        best_speed, best_probability = 160, 0
        self.second = -1
        for corner in candidates:
            if corner == self.first or corner in CORNER_NEIGHBORS[self.first]:
                continue
            speed, _ = pair_speed(board, (self.first, corner), best_speed)
            prob = probability(board, (self.first, corner))
            if (
                speed < best_speed
                or self.second < 0
                or (speed == best_speed and prob > best_probability)
            ):
                self.second = corner
                best_speed, best_probability = speed, prob
        if self.second < 0:
            raise ValueError("no second opening site")
        return self.second

    def initial_road(self, state, seat, corner, candidates, enemy_order=(), target_order=None):
        """Choose a two-away destination, then the first road towards it.

        Hashtable enumeration and initial-settlement iteration are inputs to
        differential fixtures; native callers use their declared stable order.
        """
        owned = player_corners(state, seat)
        targets = [
            c
            for c in (target_order or sorted(two_away(corner)))
            if c in two_away(corner) and legal_site(state, c)
        ]
        scores = number_scores(state.board, targets, 3, owned)
        owned_ports = {state.board.corner_port[c] for c in owned}
        port_scores(state.board, scores, 5, 10, owned_ports)
        allowed = {c for c in candidates if legal_site(state, c)}
        if state.setup_step < 4:
            other_scores = number_scores(state.board, enemy_order or candidates, 100)
            port_scores(state.board, other_scores, 5, 10, distant=True)
            # Original predicts 2 * (remaining players) placements before our
            # second settlement. It removes only the chosen node from scores.
            for _ in range(2 * (3 - seat)):
                if not other_scores:
                    break
                best = max(other_scores, key=other_scores.get)
                if other_scores[best] <= 0:
                    break
                del other_scores[best]
                allowed.difference_update((best, *CORNER_NEIGHBORS[best]))
        best, best_score = -1, 0
        for target in targets:
            if target in allowed and scores[target] > best_score:
                best, best_score = target, scores[target]
        if best < 0:
            # planInitRoad keeps its initial BoardNodeScorePair(0, 0): destination coordinate 0,
            # and getAdjacentEdgeToNode2Away then picks the road by coordinate arithmetic.
            self.road_destination = -1
            edge = adjacent_edge_to_node_2away(NODES[corner], 0)
            road = _EDGE_IDS.get(edge)
            if road is None or road not in roads_from(corner):
                raise ValueError("original opening strategy chose no road on the board")
            return road
        self.road_destination = best
        adjacent = next(n for n in CORNER_NEIGHBORS[corner] if best in CORNER_NEIGHBORS[n])
        return next(e for e, pair in enumerate(EDGE_CORNERS) if set(pair) == {corner, adjacent})


_EDGE_IDS = {coord: e for e, coord in enumerate(EDGES)}


def roads_from(corner):
    return [e for e, pair in enumerate(EDGE_CORNERS) if corner in pair]


def adjacent_edge_to_node_2away(node, node2away):
    """SOCBoard.getAdjacentEdgeToNode2Away (classic board coordinates)."""
    if (node >> 4) % 2 == 0:
        if node2away in (node - 0x02, node + 0x20):
            return node - 0x01  # south
        return node - 0x11 if node2away < node else node  # NW, NE
    if node2away in (node - 0x20, node + 0x02):
        return node - 0x10  # north
    return node if node2away > node else node - 0x11  # SE, SW
