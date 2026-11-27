"""Base-game win-time projection translated from SOCPlayerTracker.

Derived from JSettlers 2.6.10, GPL-3.0-or-later. This is the robot's greedy
projection, not a game simulation or access to an opponent's true cards.
"""

from carla.engine.rules import victory_points
from carla.engine.topology import CORNER_NEIGHBORS

from .coordinates import NODES
from .estimator import ZERO, estimates, periods_for_corners, player_corners
from .roads import award_estimates, knights_to_buy


def recalc_awards(state, tracker, development_remaining):
    """recalcLongestRoadETA + recalcLargestArmyETA: refresh the tracker's cached estimates."""
    army_eta, road_eta, _, roads = award_estimates(
        state, tracker.seat, development_remaining, tracker.road_nodes
    )
    tracker.largest_army_eta, tracker.longest_road_eta, tracker.roads_to_go = (
        army_eta,
        road_eta,
        roads,
    )


def win_time(state, seat, tracker, development_remaining, *, trace=None):
    """recalcWinGameETA. It starts from the tracker's *cached* award estimates and road count
    (see `recalc_awards`); only the knight count is recomputed inside the projection."""
    tracker.settlement_paths(state)
    corners = list(player_corners(state, seat))
    ports = {state.board.corner_port[c] for c in corners}
    settlements = dict(tracker.settlements)
    cities = set(tracker.cities)
    settlement_pieces, city_pieces = state.settlements_left[seat], state.cities_left[seat]
    city_spots = len(cities)
    army_eta, road_eta = tracker.largest_army_eta, tracker.longest_road_eta
    knights, roads = knights_to_buy(state, seat), tracker.roads_to_go
    have_army, have_road = state.largest_army_owner == seat, state.longest_road_owner == seat
    need_army = need_road = False

    def speeds(extra=(), port=None):
        new_ports = ports | ({port} if port is not None else set())
        ratios = tuple(2 if r in new_ports else 3 if 5 in new_ports else 4 for r in range(5))
        return estimates(
            ZERO, periods_for_corners(state.board, (*corners, *extra)), ratios, include_ship=True
        )

    def road_count(candidate):
        return len(candidate.path) if not candidate.necessary_roads or candidate.path else 40

    def choose_set(candidates, current_speed):
        fastest, best_gain, chosen, chosen_speed = 500, 0, None, None
        for corner in sorted(candidates, key=NODES.__getitem__):
            candidate = candidates[corner]
            eta = current_speed[1] + candidate.number_needed * current_speed[0]
            new_speed = speeds((corner,), state.board.corner_port[corner])
            gain = sum(max(0, a - b) for a, b in zip(current_speed, new_speed, strict=True))
            if eta < fastest:
                # Preserve the original accumulation in its strict-ETA branch.
                fastest, chosen, chosen_speed = eta, candidate, new_speed
                best_gain += gain
            elif eta == fastest and gain > best_gain:
                chosen, chosen_speed, best_gain = candidate, new_speed, gain
        return chosen, chosen_speed

    def remove_set(corner):
        settlements.pop(corner, None)
        for neighbor in CORNER_NEIGHBORS[corner]:
            settlements.pop(neighbor, None)

    points, total = victory_points(state, seat), 0
    current = speeds()
    while points < state.config.vp_to_win:
        road, settlement, city, card = current[:4]
        if points == state.config.vp_to_win - 1:
            fastest = 500
            if settlement_pieces > 0 and settlements:
                chosen = min(
                    settlements.values(), key=lambda s: settlement + s.number_needed * road
                )
                if settlement + chosen.number_needed * road < fastest:
                    fastest = settlement + road_count(chosen) * road
            if city_pieces > 0 and city_spots > 0:
                fastest = min(fastest, city)
            if not have_army and not need_army and army_eta < fastest:
                fastest = army_eta
            if not have_road and not need_road and road_eta < fastest:
                fastest = road_eta
            total += fastest
            if trace is not None:
                trace.append({"points": points, "eta": fastest, "kind": "final"})
            break

        fastest = 500
        chosen_city = None
        city_speed = None
        two_cities = 500
        if city_pieces > 1 and city_spots > 1:
            for corner in sorted(cities, key=NODES.__getitem__):
                city_speed = speeds((corner,))
                if city + city_speed[2] < two_cities:
                    chosen_city, two_cities = corner, city + city_speed[2]
            fastest = min(fastest, two_cities)

        chosen_set = second_set = None
        set_speed = None
        two_sets = 0
        can_two_sets = settlement_pieces > 1 and len(settlements) > 1
        if can_two_sets:
            chosen_set, set_speed = choose_set(settlements, current)
            if chosen_set is None:
                can_two_sets = False
            else:
                alternatives = {
                    c: s
                    for c, s in settlements.items()
                    if c not in (chosen_set.location, *CORNER_NEIGHBORS[chosen_set.location])
                }
                second_set, _ = choose_set(alternatives, current)
                can_two_sets = second_set is not None
                if can_two_sets:
                    two_sets = settlement + road_count(chosen_set) * road
                    two_sets += set_speed[1] + road_count(second_set) * set_speed[0]
                    fastest = min(fastest, two_sets)

        one_each, set_before_city, city_before_set = 0, 500, 500
        can_each = (
            city_pieces > 0
            and (
                (settlement_pieces > 0 and city_spots >= 0)
                or (settlement_pieces >= 0 and city_spots > 0)
            )
            and bool(settlements)
        )
        if can_each:
            if chosen_city is None and city_spots > 0:
                best_gain = 0
                for corner in sorted(cities, key=NODES.__getitem__):
                    candidate_speed = speeds((corner,))
                    gain = sum(max(0, a - b) for a, b in zip(current, candidate_speed, strict=True))
                    if gain >= best_gain:
                        best_gain, chosen_city, city_speed = gain, corner, candidate_speed
            if chosen_set is None:
                chosen_set, set_speed = choose_set(settlements, current)
            if chosen_set is not None:
                if city_spots == 0:
                    chosen_city = chosen_set.location
                if settlement_pieces > 0 and city_spots >= 0:
                    set_before_city = set_speed[2] + settlement + road_count(chosen_set) * road
                if settlement_pieces >= 0 and city_spots > 0:
                    city_before_set = city + city_speed[1] + road_count(chosen_set) * city_speed[0]
                one_each = min(set_before_city, city_before_set)
                fastest = min(fastest, one_each)
            else:
                can_each = False

        if not have_army and not need_army and points > 5:
            army_eta = (card + 1) * knights if development_remaining >= knights else 500
            fastest = min(fastest, army_eta)
        if not have_road and not need_road and points > 5:
            road_eta = road * roads
            fastest = min(fastest, road_eta)
        points += 2
        total += fastest
        if can_two_sets and fastest == two_sets:
            if trace is not None:
                trace.append(
                    {
                        "points": points,
                        "eta": fastest,
                        "kind": "two_sets",
                        "corners": (chosen_set.location, second_set.location),
                    }
                )
            for candidate in (chosen_set, second_set):
                remove_set(candidate.location)
                cities.add(candidate.location)
                corners.append(candidate.location)
                ports.add(state.board.corner_port[candidate.location])
            settlement_pieces -= 2
            city_spots += 2
            current = speeds()
        elif can_each and fastest == one_each:
            if trace is not None:
                trace.append(
                    {
                        "points": points,
                        "eta": fastest,
                        "kind": "one_each",
                        "corners": (chosen_set.location, chosen_city),
                    }
                )
            if chosen_city is None:
                break
            remove_set(chosen_set.location)
            if chosen_set.location != chosen_city:
                cities.add(chosen_set.location)
            cities.discard(chosen_city)
            city_pieces -= 1
            corners.extend((chosen_set.location, chosen_city))
            ports.add(state.board.corner_port[chosen_set.location])
            current = speeds()
        elif city_pieces > 1 and city_spots > 1 and fastest == two_cities:
            if trace is not None:
                trace.append(
                    {"points": points, "eta": fastest, "kind": "two_cities", "corner": chosen_city}
                )
            cities.discard(chosen_city)
            corners.append(chosen_city)
            best_gain, second_city = 0, None
            for corner in sorted(cities, key=NODES.__getitem__):
                candidate_speed = speeds((corner,))
                gain = sum(max(0, a - b) for a, b in zip(current, candidate_speed, strict=True))
                if gain >= best_gain:
                    best_gain, second_city = gain, corner
            count = 1
            if second_city is not None:
                cities.discard(second_city)
                corners.append(second_city)
                count = 2
            else:
                points -= 1
            settlement_pieces += count
            city_pieces -= count
            city_spots -= count
            current = speeds()
        elif not have_road and not need_road and points - 2 > 5 and fastest == road_eta:
            if trace is not None:
                trace.append({"points": points, "eta": fastest, "kind": "road"})
            need_road = True
        elif not have_army and not need_army and points - 2 > 5 and fastest == army_eta:
            if trace is not None:
                trace.append({"points": points, "eta": fastest, "kind": "army"})
            need_army = True
    return total
