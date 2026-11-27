"""SMART planning against private copies of public board-query state.

Derived from SOCRobotDM, JSettlers 2.6.10, GPL-3.0-or-later. These copies do
not contain the referee's hidden hands, deck order or random generator.
"""

from copy import deepcopy

from carla.engine.events import Event, EventKind
from carla.engine.rules import _update_largest_army, _update_longest_road, longest_road_length
from carla.engine.topology import CORNER_EDGES
from carla.planning.jsettlers_eta import estimate, f32

from .estimator import estimates, periods_for_corners, player_corners, ratios_for_corners
from .projection import recalc_awards, win_time
from .smart import THREAT_MULTIPLIER, eta_bonus, win_time_bonus
from .trackers import Candidate, PieceType


def put_hypothetical(state, seat, piece):
    """SOCGame.putTempPiece: place the piece and update road lengths and Longest Road, as the
    engine's build handlers do (`_update_longest_road` only awards from stored lengths)."""
    if piece.type == PieceType.ROAD:
        state.edge_owner[piece.location] = seat
        state.roads_left[seat] -= 1
        state.road_lengths[seat] = longest_road_length(state, seat)
        event = Event(EventKind.ROAD_BUILT, player=seat, location=piece.location)
    elif piece.type == PieceType.SETTLEMENT:
        state.corner_owner[piece.location] = seat
        state.corner_building[piece.location] = 1
        state.settlements_left[seat] -= 1
        # A settlement can cut opponents' roads.
        for owner in {state.edge_owner[e] for e in CORNER_EDGES[piece.location]} - {-1, seat}:
            state.road_lengths[owner] = longest_road_length(state, owner)
        event = Event(EventKind.SETTLEMENT_BUILT, player=seat, location=piece.location)
    else:
        state.corner_building[piece.location] = 2
        state.cities_left[seat] -= 1
        state.settlements_left[seat] += 1
        event = Event(EventKind.CITY_BUILT, player=seat, location=piece.location)
    _update_longest_road(state, [])
    return event


def win_times(state, trackers, remaining):
    """SOCPlayerTracker.updateWinGameETAs: refresh each tracker's award estimates, then project."""
    for tracker in trackers:
        recalc_awards(state, tracker, remaining)
    return projected_win_times(state, trackers, remaining)


def projected_win_times(state, trackers, remaining):
    """recalcWinGameETA alone on every tracker, reusing cached award estimates (as
    RobberStrategy.selectPlayerToThwart does)."""
    values = tuple(win_time(state, tracker.seat, tracker, remaining) for tracker in trackers)
    for tracker, value in zip(trackers, values, strict=True):
        tracker.win_eta = value
    return values


def smart_plan(state, seat, trackers, remaining):
    own = trackers[seat]
    periods = periods_for_corners(state.board, player_corners(state, seat))
    ratios = ratios_for_corners(state.board, player_corners(state, seat))
    times = estimates(state.hands[seat], periods, ratios, include_ship=True)
    before = win_times(state, trackers, remaining)

    def score(piece, eta):
        hypothetical = state.copy()
        copies = deepcopy(trackers)
        if piece.type == PieceType.ROAD:
            try:
                _, resources = estimate(hypothetical.hands[seat], piece.cost, periods, ratios, 50)
                hypothetical.hands[seat] = [
                    a - b for a, b in zip(resources, piece.cost, strict=True)
                ]
            except TimeoutError:
                pass  # Original retains the existing hand at this cutoff.
        event = put_hypothetical(hypothetical, seat, piece)
        for tracker in copies:
            tracker.observe((event,))
            tracker.refresh(hypothetical)
        bonus = win_time_bonus(before, win_times(hypothetical, copies, remaining), seat)
        if piece.type == PieceType.ROAD and piece.threats:
            # getWinGameETABonusForRoad; settlements and cities have no such multiplier.
            bonus = f32(bonus * f32(THREAT_MULTIPLIER))
        return eta_bonus(eta, bonus)

    settlements = []
    roads = []
    cities = []
    if state.settlements_left[seat]:
        for piece in own.settlements.values():
            piece.score = 0
            if not piece.necessary_roads:
                piece.score = score(piece, times[1])
                settlements.append(piece)
    if state.roads_left[seat]:
        for piece in own.roads.values():
            piece.score = 0
            if not piece.necessary_roads:
                # Pinned smartGameStrategy uses SHIP's ETA for its ordinary
                # "good roads" branch, even on the classic board.
                piece.score = score(piece, times[4])
                roads.append(piece)
    if state.cities_left[seat]:
        for piece in own.cities.values():
            piece.score = score(piece, times[2])
            cities.append(piece)
    settlement = max(settlements, key=lambda c: c.score, default=None)
    road = max(roads, key=lambda c: c.score, default=None)
    city = max(cities, key=lambda c: c.score, default=None)
    pick = None
    if (
        city is not None
        and city.score > 0
        and (
            settlement is None
            or city.score > settlement.score
            or (city.score == settlement.score and times[2] < times[1])
        )
        and (
            road is None
            or city.score > road.score
            or (city.score == road.score and times[2] < times[0])
        )
    ):
        pick = city
    elif (
        road is not None
        and road.score > 0
        and (settlement is None or settlement.score < road.score)
    ):
        pick = road
    elif settlement is not None:
        pick = settlement
    if remaining and state.current_player == seat:
        knight_state, vp_state = state.copy(), state.copy()
        knight_state.dev_played[seat][0] += 1
        _update_largest_army(knight_state, seat, [])
        knight_bonus = win_time_bonus(
            before, win_times(knight_state, deepcopy(trackers), remaining), seat
        )
        vp_state.dev_hands[seat][4] += 1
        vp_trackers = deepcopy(trackers)
        vp_times = win_times(vp_state, vp_trackers, remaining)
        vp_bonus = win_time_bonus(before, vp_times, seat)
        # getDevCardScore runs these updates on the real trackers and removes its temporary VP
        # card without recalculating afterwards, so they keep the VP-card ETAs and awards.
        for tracker, copy, value in zip(trackers, vp_trackers, vp_times, strict=True):
            tracker.win_eta = value
            tracker.adopt_awards(copy)
        bonus = f32(f32(knight_bonus * f32(0.58)) + f32(vp_bonus * f32(0.21)))
        card = Candidate(
            PieceType.CARD, -1, eta=times[3], score=eta_bonus(times[3], f32(bonus + 3))
        )
        if pick is None or card.score > pick.score:
            pick = card
    plan = [] if pick is None else [pick]
    if (
        road is not None
        and state.roads_left[seat] >= 2
        and not state.dev_played_this_turn
        and state.dev_hands[seat][1]
        - (state.dev_bought_this_turn[1] if seat == state.current_player else 0)
        > 0
    ):
        after_road = state.copy()
        put_hypothetical(after_road, seat, road)
        next_roads = [
            piece
            for piece in own.roads.values()
            if road.location in piece.necessary_roads and after_road.edge_owner[piece.location] < 0
        ]
        second = next(iter(next_roads), None)
        if second is None:
            second = next((p for p in roads if p.location != road.location), None)
        if second is not None and plan:
            if plan[0].type == PieceType.ROAD:
                plan.insert(1, second)
            else:
                plan = [road, second, *plan]
    return tuple(plan)
