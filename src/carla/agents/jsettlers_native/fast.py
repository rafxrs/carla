"""SOCRobotDM.dumbFastGameStrategy translated for base-game candidate inputs.

Derived from JSettlers 2.6.10, GPL-3.0-or-later. Award path construction is
provided by roads.py and can also be supplied explicitly in module fixtures.
"""

from dataclasses import dataclass

from carla.engine.constants import DevCard
from carla.engine.rules import victory_points

from .estimator import estimates, periods_for_corners, player_corners, ratios_for_corners, rolls
from .trackers import Candidate, PieceType, Tracker


@dataclass(frozen=True)
class AwardPlan:
    eta: int = 500
    pieces: tuple[Candidate, ...] = ()


def army_plan(state, seat, periods, ratios, development_remaining):
    holder = state.largest_army_owner
    size = 3 if holder < 0 else 0 if holder == seat else state.dev_played[holder][0] + 1
    count = 0
    if state.dev_played[seat][0] + state.dev_hands[seat][DevCard.KNIGHT] < size:
        old = state.dev_hands[seat][DevCard.KNIGHT]
        if state.current_player == seat:
            old -= state.dev_bought_this_turn[DevCard.KNIGHT]
        count = size - state.dev_played[seat][0] - old
    if development_remaining < count:
        return AwardPlan()
    cost = tuple(c * count for c in Candidate(PieceType.CARD, -1).cost)
    eta = rolls(state.hands[seat], cost, periods, ratios)
    return AwardPlan(eta, tuple(Candidate(PieceType.CARD, -1, eta=1) for _ in range(count)))


def fast_plan(state, seat, tracker: Tracker, development_remaining, *, longest_road=None):
    """Return pieces in execution order; never substitute a different bot."""
    if longest_road is None:
        from .roads import longest_road_plan

        longest_road = longest_road_plan(state, seat, node_order=tracker.road_nodes)
    corners = player_corners(state, seat)
    periods = periods_for_corners(state.board, corners)
    ratios = ratios_for_corners(state.board, corners)
    etas = estimates(state.hands[seat], periods, ratios)
    tracker.settlement_paths(state)
    city = settlement = None
    best_eta = 500
    special = state.current_player != seat
    if victory_points(state, seat) < 5:
        if state.cities_left[seat]:
            for candidate in tracker.cities.values():
                if city is None or sum(candidate.speedup) > sum(city.speedup):
                    city, best_eta = candidate, etas[2]
        for candidate in tracker.settlements.values():
            if candidate.eta < best_eta:
                settlement, best_eta = candidate, candidate.eta
            elif candidate.eta == best_eta:
                incumbent = settlement or city
                if incumbent is None or sum(candidate.speedup) > sum(incumbent.speedup):
                    settlement = candidate
        chosen = settlement or city
        if chosen is None:
            return (
                (Candidate(PieceType.CARD, -1, eta=etas[3]),)
                if development_remaining and not special
                else ()
            )
    else:
        chosen = None
        army = army_plan(state, seat, periods, ratios, development_remaining)
        if not special and army.eta < best_eta:
            chosen, best_eta = army, army.eta
        if longest_road.eta < best_eta:
            chosen, best_eta = longest_road, longest_road.eta
        if state.cities_left[seat] and etas[2] <= best_eta:
            for candidate in tracker.cities.values():
                if city is None or sum(candidate.speedup) > sum(city.speedup):
                    city, chosen, best_eta = candidate, candidate, etas[2]
        if state.settlements_left[seat]:
            for candidate in tracker.settlements.values():
                if candidate.eta < best_eta:
                    settlement, chosen, best_eta = candidate, candidate, candidate.eta
                elif candidate.eta == best_eta:
                    incumbent = settlement or city
                    if incumbent is None or sum(candidate.speedup) > sum(incumbent.speedup):
                        settlement, chosen = candidate, candidate
        if isinstance(chosen, AwardPlan):
            return chosen.pieces
        if chosen is None:
            return ()
    return (*(tracker.roads[e] for e in chosen.path), chosen)
