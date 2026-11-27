"""Native JSettlers component checks, including original-Java differential checks."""

import os

import pytest

from carla.agents.api import AgentView
from carla.agents.heuristic import HeuristicAgent
from carla.agents.jsettlers_native.coordinates import (
    EDGES,
    NODES,
    hashtable_order,
    initial_targets,
    potential_order,
)
from carla.agents.jsettlers_native.estimator import periods_for_corners, player_corners
from carla.agents.jsettlers_native.knowledge import Knowledge, ResourceKnowledge
from carla.agents.jsettlers_native.opening import Opening
from carla.agents.jsettlers_native.reference import JSettlersReference
from carla.agents.jsettlers_native.strategies import bank_trade, discard, free_resources, monopoly
from carla.engine import Game
from carla.engine.actions import ActionType
from carla.engine.constants import SETTLEMENT_COST
from carla.engine.events import Event, EventKind
from carla.engine.rules import trade_ratios

JAVA = pytest.mark.skipif(
    not os.environ.get("JSETTLERS_JAR") or not os.environ.get("CARLA_JSETTLERS_CLASSES"),
    reason="optional pinned JSettlers runtime required",
)


def test_unknown_loss_discards_type_knowledge_but_known_spend_uses_unknown():
    hand = ResourceKnowledge([2, 1, 0, 0, 0], 3)
    hand.change(1, -2)
    assert hand.known == [2, 0, 0, 0, 0] and hand.unknown == 2
    hand.change(-1, -1)
    assert hand.known == [0] * 5 and hand.unknown == 3
    hand.change(3, 2)
    hand.reconcile(5)
    assert hand.known == [0, 0, 0, 2, 0] and hand.unknown == 3


def test_acceptance_does_not_transfer_resources_and_event_delivery_is_idempotent():
    state = AgentView.from_game(Game(seed=12), 0).state
    knowledge = Knowledge(state, 0)
    knowledge.hands[0] = ResourceKnowledge([1, 0, 0, 0, 0])
    knowledge.hands[1] = ResourceKnowledge([0, 1, 0, 0, 0])
    assert knowledge.observe([Event(EventKind.TRADE_RESPONDED, player=1, target=0, value=1)], 1)
    assert knowledge.hands[0].known == [1, 0, 0, 0, 0]
    execution = Event(
        EventKind.TRADE_EXECUTED, player=0, target=1, give=(1, 0, 0, 0, 0), receive=(0, 1, 0, 0, 0)
    )
    assert knowledge.observe([execution], 2)
    assert not knowledge.observe([execution], 2)
    assert knowledge.hands[0].known == [0, 1, 0, 0, 0]
    assert knowledge.hands[1].known == [1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="sequence"):
        knowledge.observe([], 4)


def test_discard_preserves_upstream_descending_sort_despite_its_comment():
    assert discard([1] * 5, [0] * 5, [1, 5, 3, 2, 4], 2) == (0, 1, 0, 0, 1)
    assert discard([2, 1, 1, 1, 0], SETTLEMENT_COST, [1] * 5, 2) == (1, 1, 0, 0, 0)


@JAVA
@pytest.mark.parametrize("seed", [12, 41, 552])
def test_opening_pair_matches_original_with_original_candidate_order(seed):
    game = Game(seed=seed)
    with JSettlersReference() as reference:
        reference.reset(game, 0, 42)
        snapshot = reference.inspect(game)
        assert tuple(snapshot["nodes"]) == NODES
        assert tuple(snapshot["edges"]) == EDGES
        candidates = snapshot["players"][0]["potential_order"]
        assert potential_order(candidates) == candidates
        assert (
            Opening().first_settlement(game.state.board, candidates)
            == reference.inspect(game, "opening")["corner"]
        )


@JAVA
def test_resource_knowledge_and_strategies_against_original():
    game, policy = Game(seed=41), HeuristicAgent()
    seat = 1
    knowledge = Knowledge(AgentView.from_game(game, seat).state, seat)
    comparisons = 0
    with JSettlersReference() as reference:
        reference.reset(game, seat, 42)
        for step in range(240):
            if game.is_over:
                break
            if step % 11 == 0:
                view = AgentView.from_game(game, seat)
                knowledge.reconcile(view.state)
                snapshot = reference.inspect(game)
                assert knowledge.snapshot() == [
                    {"known": entry["known"], "unknown": entry["unknown"]}
                    for entry in snapshot["players"]
                ]
                state = game.state
                periods = periods_for_corners(state.board, player_corners(state, seat))
                ratios = trade_ratios(state, seat)
                for target in (SETTLEMENT_COST, (0, 0, 0, 2, 3), (2, 1, 0, 0, 0)):
                    actual = bank_trade(state.hands[seat], target, periods, ratios)
                    expected = reference.inspect(game, "bank", target=target)["offer"]
                    if actual is None:
                        assert expected is None
                    else:
                        assert expected["give"][actual.resource] == ratios[actual.resource]
                        assert expected["receive"][actual.resource2] == 1
                    expected = reference.inspect(game, "free_resources", target=target, count=2)[
                        "resources"
                    ]
                    assert free_resources(state.hands[seat], target, periods) == tuple(expected)
                    comparisons += 2
                expected = reference.inspect(game, "monopoly")
                actual = monopoly([hand.known for hand in knowledge.hands], seat, ratios)
                assert expected["play"] == (actual is not None)
                if actual is not None:
                    assert expected["resource"] == actual
            action = policy.act(game, game.decision_player, game.legal_actions())
            events = game.step(action)
            visible = [event.visible_to(seat) for event in events]
            knowledge.observe(visible, game.num_steps)
            reference.observe(visible, game.num_steps)
        assert comparisons > 100


def test_bank_trade_uses_needed_surplus_only_with_the_original_period_rule():
    action = bank_trade([4, 0, 0, 0, 0], (0, 1, 0, 0, 0), (1, 5, 3, 3, 3), [4] * 5)
    assert action.type == ActionType.MARITIME_TRADE
    assert (action.resource, action.resource2) == (0, 1)


@JAVA
@pytest.mark.parametrize("seat", range(4))
def test_complete_opening_sequence_against_original(seat):
    from carla.engine.actions import Action
    from carla.engine.state import Phase

    game, policy, opening = Game(seed=41), HeuristicAgent(), Opening()
    with JSettlersReference() as reference:
        reference.reset(game, seat, 42)
        while game.state.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            snapshot = reference.inspect(game)
            if game.decision_player == seat:
                candidates = snapshot["players"][seat]["potential_order"]
                if game.state.phase == Phase.SETUP_SETTLEMENT:
                    second = game.state.setup_step >= 4
                    expected = reference.inspect(game, "opening", second=second)["corner"]
                    actual = (
                        opening.second_settlement(game.state.board, candidates)
                        if second
                        else opening.first_settlement(game.state.board, candidates)
                    )
                    assert actual == expected
                    action = Action(ActionType.BUILD_SETTLEMENT, corner=actual)
                else:
                    expected = reference.inspect(game, "initial_road")["edge"]
                    order = hashtable_order(
                        c for c in initial_targets(game.state.setup_corner) if c in candidates
                    )
                    actual = opening.initial_road(
                        game.state,
                        seat,
                        game.state.setup_corner,
                        candidates,
                        enemy_order=hashtable_order(candidates),
                        target_order=order,
                    )
                    assert actual == expected
                    action = Action(ActionType.BUILD_ROAD, edge=actual)
            else:
                action = policy.act(game, game.decision_player, game.legal_actions())
            events = game.step(action)
            reference.observe([e.visible_to(seat) for e in events], game.num_steps)


@JAVA
def test_native_candidate_graph_against_reference():
    from carla.agents.jsettlers_native.trackers import Tracker, update_conflicts
    from carla.engine.state import Phase

    game, policy = Game(seed=41), HeuristicAgent()
    trackers = [Tracker(p) for p in range(4)]
    with JSettlersReference() as reference:
        reference.reset(game, 1, 42)
        for step in range(900):
            if game.is_over:
                break
            if step % 11 == 0 and game.state.phase > Phase.SETUP_ROAD:
                snapshot = reference.inspect(game)
                for tracker in trackers:
                    tracker.refresh(game.state)
                update_conflicts(trackers, game.state)
                for tracker in trackers:
                    actual = {
                        (int(c.type), c.location): set(c.necessary_roads)
                        for c in (
                            *tracker.roads.values(),
                            *tracker.settlements.values(),
                            *tracker.cities.values(),
                        )
                    }
                    expected = {
                        (c["type"], c["location"]): set(c["necessary_roads"])
                        for c in snapshot["players"][tracker.seat]["candidates"]
                    }
                    assert actual == expected, (step, tracker.seat)
                    for c in snapshot["players"][tracker.seat]["candidates"]:
                        if c["type"] == 2:
                            assert tuple(c["speedup"]) == tracker.cities[c["location"]].speedup
                        else:
                            table = tracker.roads if c["type"] == 0 else tracker.settlements
                            candidate = table[c["location"]]
                            assert set(candidate.threats) == {tuple(t) for t in c["threats"]}, (
                                step,
                                tracker.seat,
                                c,
                            )
                            if c["type"] == 1:
                                assert set(candidate.conflicts) == {
                                    tuple(t) for t in c["conflicts"]
                                }
            events = game.step(policy.act(game, game.decision_player, game.legal_actions()))
            reference.observe([e.visible_to(1) for e in events], game.num_steps)


@JAVA
def test_fast_early_game_plans_against_reference():
    from carla.agents.jsettlers_native.fast import fast_plan
    from carla.agents.jsettlers_native.roads import longest_paths, longest_road_plan
    from carla.agents.jsettlers_native.trackers import Tracker
    from carla.engine.state import Phase

    comparisons = 0
    game, policy, tracker = Game(seed=12), HeuristicAgent(), Tracker(0)
    with JSettlersReference() as reference:
        reference.reset(game, 0, 42)
        for step in range(900):
            if game.is_over:
                break
            if game.state.phase == Phase.MAIN and game.decision_player == 0:
                tracker.refresh(game.state)
                snapshot = reference.inspect(game, "plan")
                nodes = snapshot["players"][0]["road_nodes"]
                actual_paths = longest_paths(game.state, 0, nodes)
                assert [(p.beginning, p.end, p.length) for p in actual_paths] == [
                    (p["beginning"], p["end"], p["length"])
                    for p in snapshot["players"][0]["longest_paths"]
                ]
                road_plan = longest_road_plan(game.state, 0, node_order=nodes)
                plan = fast_plan(
                    game.state, 0, tracker, sum(game.state.dev_deck), longest_road=road_plan
                )
                expected = snapshot["plan"]
                assert [(int(c.type), c.location) for c in plan] == [
                    (c["type"], c["location"]) for c in expected
                ], step
                comparisons += 1
            events = game.step(policy.act(game, game.decision_player, game.legal_actions()))
            reference.observe([e.visible_to(0) for e in events], game.num_steps)
        assert comparisons > 5


@JAVA
def test_smart_numeric_kernels_against_original_with_tied_and_zero_etas():
    from carla.agents.jsettlers_native.smart import eta_bonus, win_time_bonus
    from carla.planning.jsettlers_eta import f32

    cases = [
        ((100, 100, 100, 100), (80, 120, 110, 90)),
        ((100, 50, 50, 200), (70, 70, 60, 220)),
        ((1, 0, 0, 0), (0, 0, 1, 2)),
        ((500, 10, 100, 50), (450, 30, 90, 60)),
    ]
    game = Game(seed=41)
    for seat in range(4):
        with JSettlersReference("smart") as reference:
            reference.reset(game, seat, 42)
            for before, after in cases:
                for eta in (0, 1, 40):
                    expected = reference.inspect(
                        game, "smart_score", before=before, after=after, eta=eta
                    )["score"]
                    assert eta_bonus(eta, win_time_bonus(before, after, seat)) == f32(expected)


@JAVA
def test_win_time_projection_against_original():
    from carla.agents.jsettlers_native.projection import recalc_awards, win_time
    from carla.agents.jsettlers_native.trackers import Tracker
    from carla.agents.jsettlers_native.view import query_state
    from carla.engine.state import Phase

    game, policy = Game(seed=12), HeuristicAgent()
    trackers = [Tracker(p) for p in range(4)]
    knowledge = Knowledge(AgentView.from_game(game, 0).state, 0)
    with JSettlersReference("smart") as reference:
        reference.reset(game, 0, 42)
        for step in range(900):
            if game.is_over:
                break
            if step % 41 == 0 and game.state.phase > Phase.SETUP_ROAD:
                snapshot = reference.inspect(game, "win_times")
                view = AgentView.from_game(game, 0)
                knowledge.reconcile(view.state)
                state = query_state(view, knowledge)
                for tracker in trackers:
                    tracker.refresh(state)
                    # The "win_times" operation is updateWinGameETAs: awards first.
                    recalc_awards(state, tracker, view.state["development_remaining"])
                    actual = win_time(
                        state, tracker.seat, tracker, view.state["development_remaining"]
                    )
                    assert actual == snapshot["players"][tracker.seat]["win_eta"], (
                        step,
                        tracker.seat,
                    )
            events = game.step(policy.act(game, game.decision_player, game.legal_actions()))
            visible = [e.visible_to(0) for e in events]
            knowledge.observe(visible, game.num_steps)
            for tracker in trackers:
                tracker.observe(visible)
                if game.state.phase > Phase.SETUP_ROAD:
                    tracker.refresh(game.state)
            reference.observe(visible, game.num_steps)


@JAVA
def test_smart_planning_against_original():
    from carla.agents.jsettlers_native.planner import smart_plan
    from carla.agents.jsettlers_native.trackers import Tracker
    from carla.agents.jsettlers_native.view import query_state
    from carla.engine.state import Phase

    game, policy = Game(seed=12), HeuristicAgent()
    trackers = [Tracker(p) for p in range(4)]
    knowledge = Knowledge(AgentView.from_game(game, 0).state, 0)
    comparisons = 0
    with JSettlersReference("smart") as reference:
        reference.reset(game, 0, 42)
        for step in range(250):
            if game.is_over:
                break
            if game.state.phase == Phase.MAIN and game.decision_player == 0 and step % 3 == 0:
                view = AgentView.from_game(game, 0)
                knowledge.reconcile(view.state)
                state = query_state(view, knowledge)
                actual = smart_plan(state, 0, trackers, view.state["development_remaining"])
                expected = reference.inspect(game, "plan")["plan"]
                assert [(int(c.type), c.location) for c in actual] == [
                    (c["type"], c["location"]) for c in expected
                ], step
                comparisons += 1
            events = game.step(policy.act(game, game.decision_player, game.legal_actions()))
            visible = [e.visible_to(0) for e in events]
            knowledge.observe(visible, game.num_steps)
            for tracker in trackers:
                tracker.observe(visible)
                if game.state.phase > Phase.SETUP_ROAD:
                    tracker.refresh(game.state)
            reference.observe(visible, game.num_steps)
        assert comparisons > 3


@pytest.mark.parametrize("strategy", ["fast", "smart"])
@pytest.mark.parametrize("trading", [False, True])
def test_native_agents_finish_without_java(strategy, trading):
    from carla.agents.registry import make_agent
    from carla.engine.config import EngineConfig
    from carla.eval.runner import play_game

    result = play_game(
        [make_agent(f"jsettlers-{strategy}-native"), *[HeuristicAgent() for _ in range(3)]],
        12,
        EngineConfig(max_turns=220, allow_domestic_trade=trading),
    )
    assert result.winner is not None
    if trading:
        assert result.trade_offers[0] > 0


def test_native_policy_does_not_read_hidden_hands_or_change_referee():
    from carla.agents.jsettlers_native.agent import NativeJSettlersAgent
    from carla.engine.rng import Rng
    from carla.engine.state import Phase

    game, policy = Game(seed=12), HeuristicAgent()
    while game.state.phase != Phase.MAIN or game.decision_player != 0:
        game.step(policy.act(game, game.decision_player, game.legal_actions()))
    changed = game.clone()
    for player in (1, 2, 3):
        changed.state.hands[player] = list(reversed(changed.state.hands[player]))
        changed.state.dev_hands[player] = list(reversed(changed.state.dev_hands[player]))
    changed.rng = Rng(999)
    first, second = NativeJSettlersAgent("fast"), NativeJSettlersAgent("fast")
    first.reset(game, 0, 42)
    second.reset(changed, 0, 42)
    before = game.state_hash()
    assert first.act(game, 0, game.legal_actions()) == second.act(changed, 0, game.legal_actions())
    assert game.state_hash() == before


@JAVA
@pytest.mark.parametrize(
    ("strategy", "trading", "seed", "seat", "limit"),
    [
        ("smart", False, 12, 0, 80),
        ("fast", True, 12, 0, 80),
        # Each history below diverged until a specific fix (see the native port notes):
        ("fast", False, 31, 0, 40),  # robber: win ETAs reuse cached award estimates
        ("smart", False, 31, 2, 60),  # hypothetical roads update road lengths
        ("smart", False, 47, 1, 40),  # removeDependents recounts necessary roads
        ("smart", False, 47, 2, 60),  # the adapter no longer skips single-option decisions
        ("smart", False, 12, 1, 90),  # a Largest Army knight is played before planning
        ("fast", True, 12, 1, 40),  # no more offers in a turn once makeOffer found none
        ("smart", True, 12, 3, 150),  # turn resets happen when the turn ends, not lazily
        ("fast", True, 58, 2, 160),  # selling flags reset after own production, not per gain
        ("smart", True, 303, 1, 120),  # wantsAnotherOffer survives turn boundaries
        ("smart", True, 404, 3, 150),  # offer checks and trade responses do not plan early
    ],
)
def test_java_adapter_compatibility_decision_sequences(strategy, trading, seed, seat, limit):
    from carla.agents.jsettlers import JSettlersAgent
    from carla.agents.jsettlers_native.agent import NativeJSettlersAgent
    from carla.engine.config import EngineConfig

    game = Game(EngineConfig(max_turns=220, allow_domestic_trade=trading), seed=seed)
    native = NativeJSettlersAgent(strategy, compatibility="java-adapter")
    reference = JSettlersAgent(strategy)
    others, decisions = [HeuristicAgent() for _ in range(4)], 0
    try:
        native.reset(game, seat, 42)
        reference.reset(game, seat, 42)
        while not game.is_over and decisions < limit:
            actor, legal = game.decision_player, game.legal_actions()
            if actor == seat:
                actual, expected = native.act(game, seat, legal), reference.act(game, seat, legal)
                assert actual == expected, (game.num_steps, actual, expected)
                action = expected
                decisions += 1
            else:
                action = others[actor].act(game, actor, legal)
            visible = [e.visible_to(seat) for e in game.step(action)]
            native.observe(visible, game.num_steps)
            reference.observe(visible, game.num_steps)
        assert decisions == limit or game.is_over
    finally:
        native.close()
        reference.close()


def _main_phase_agent(seed=12, seat=0, decisions=12):
    from carla.agents.jsettlers_native.agent import NativeJSettlersAgent
    from carla.engine.state import Phase

    game = Game(seed=seed)
    agent, others = NativeJSettlersAgent("smart"), HeuristicAgent()
    agent.reset(game, seat, 42)
    made = 0
    while made < decisions or game.state.phase != Phase.MAIN or game.decision_player != seat:
        actor, legal = game.decision_player, game.legal_actions()
        action = agent.act(game, seat, legal) if actor == seat else others.act(game, actor, legal)
        made += actor == seat
        agent.observe([e.visible_to(seat) for e in game.step(action)], game.num_steps)
    agent.act(game, seat, game.legal_actions())
    return game, agent


def test_projection_alone_reuses_cached_award_estimates():
    from copy import deepcopy

    from carla.agents.jsettlers_native.planner import projected_win_times, win_times
    from carla.agents.jsettlers_native.roads import award_estimates
    from carla.agents.jsettlers_native.trackers import Tracker

    _, agent = _main_phase_agent()
    fresh = [Tracker(t.seat, road_nodes=list(t.road_nodes)) for t in agent.trackers]
    for tracker, source in zip(fresh, agent.trackers, strict=True):
        tracker.refresh(agent.state)
        tracker.settlements, tracker.roads = deepcopy(source.settlements), deepcopy(source.roads)
    projected_win_times(agent.state, fresh, agent.remaining)
    # SOCPlayerTracker's constructor values survive recalcWinGameETA on its own.
    assert all(
        (t.longest_road_eta, t.roads_to_go, t.largest_army_eta) == (500, 20, 500) for t in fresh
    )
    win_times(agent.state, fresh, agent.remaining)
    for tracker in fresh:
        army, road, _, roads = award_estimates(
            agent.state, tracker.seat, agent.remaining, tracker.road_nodes
        )
        assert (tracker.largest_army_eta, tracker.longest_road_eta, tracker.roads_to_go) == (
            army,
            road,
            roads,
        )


def test_hypothetical_road_updates_road_lengths_like_the_engine():
    from carla.agents.jsettlers_native.planner import put_hypothetical
    from carla.agents.jsettlers_native.trackers import Candidate, PieceType
    from carla.engine.rules import longest_road_length, road_edges

    game, agent = _main_phase_agent(decisions=20)
    seat = agent.seat
    for edge in road_edges(game.state, seat):
        state = game.state.copy()
        put_hypothetical(state, seat, Candidate(PieceType.ROAD, edge))
        assert state.road_lengths[seat] == longest_road_length(state, seat)


def test_removed_prerequisite_recounts_like_remove_dependents():
    from carla.agents.jsettlers_native.trackers import Candidate, PieceType, _incremental_count

    roads = {
        37: Candidate(PieceType.ROAD, 37, number_needed=0),
        47: Candidate(PieceType.ROAD, 47, necessary_roads=[37], number_needed=1),
    }
    previous = Candidate(PieceType.SETTLEMENT, 31, necessary_roads=[37, 47, 46], number_needed=2)
    # Another player took road 46: the stale count is replaced by the remaining minimum.
    assert _incremental_count(previous, [37, 47], roads) == 1
    # Without removals or additions the original keeps its (possibly stale) count.
    assert _incremental_count(previous, [37, 47, 46], roads) == 2
