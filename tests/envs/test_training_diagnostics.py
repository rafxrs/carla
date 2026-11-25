import pickle
from collections import Counter

import numpy as np
import pytest

from carla.config import EnvConfig, load_config
from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.constants import DevCard
from carla.engine.events import Event
from carla.engine.events import EventKind as EK
from carla.engine.game import Game, IllegalActionError
from carla.engine.state import Phase
from carla.engine.topology import CORNER_NEIGHBORS
from carla.envs import CatanAECEnv, CatanEnv
from carla.envs.training_diagnostics import TRAINING_COUNTER_FIELDS, TrainingDiagnostics
from tests.engine.scenario import (
    edge_between,
    give,
    give_dev,
    post_setup_game,
    put_road,
    put_settlement,
)


def apply(game, counters, action, outcome=None):
    actor, phase = game.decision_player, game.phase
    events = game.step(action, outcome)
    for counter in counters:
        counter.action_applied(actor, phase, action)
        counter.after_step(game, events)
    return events


def equal_observations(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            equal_observations(left[key], right[key])
        else:
            np.testing.assert_array_equal(left[key], right[key])


def test_disabled_default_and_old_config():
    assert not load_config().runtime.collect_training_diagnostics
    assert load_config(
        overrides=["runtime.collect_training_diagnostics=true"]
    ).runtime.collect_training_diagnostics
    env = CatanEnv()
    env.reset(seed=1)
    assert env.training_diagnostics is None
    assert CatanAECEnv()._training_diagnostics is None


def test_enabled_collection_preserves_gameplay_rewards_rng_and_legality_calls(monkeypatch):
    plain = CatanEnv(
        EngineConfig(max_turns=3, allow_domestic_trade=True), EnvConfig(shaping="strategic")
    )
    enabled = CatanEnv(
        EngineConfig(max_turns=3, allow_domestic_trade=True),
        EnvConfig(shaping="strategic"),
        collect_training_diagnostics=True,
    )
    original = Game.legal_actions
    calls = Counter()

    def legal(game):
        calls[id(game)] += 1
        return original(game)

    monkeypatch.setattr(Game, "legal_actions", legal)
    seats = set()
    for seed in range(5):
        calls.clear()
        obs, _ = plain.reset(seed=seed)
        collected, _ = enabled.reset(seed=seed)
        equal_observations(obs, collected)
        seats.add(enabled.learner)
        assert enabled.training_diagnostics.learner == enabled.learner
        assert enabled.training_diagnostics.diagnostic_learner_decisions == 0
        while not plain.game.is_over:
            # Do not add an asymmetrical legality query to the measured paths.
            action = original(plain.game)[0]
            obs, reward, term, trunc, _ = plain.step(action)
            collected, actual, actual_term, actual_trunc, _ = enabled.step(action)
            equal_observations(obs, collected)
            assert (reward, term, trunc) == (actual, actual_term, actual_trunc)
            assert plain.game.to_dict() == enabled.game.to_dict()
        assert calls[id(plain.game)] == calls[id(enabled.game)]
        assert plain.np_random.bit_generator.state == enabled.np_random.bit_generator.state
        for left, right in zip(plain.opponents, enabled.opponents, strict=True):
            assert left._rng.getstate() == right._rng.getstate()
        row = enabled.training_diagnostics.record()
        assert set(row) == set(TRAINING_COUNTER_FIELDS)
        assert row["setup_settlements"] == row["setup_roads"] == 2
    assert len(seats) > 1


def test_actual_domestic_trade_is_attributed_to_initiator_and_partner():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 0, 0, 0, 0))
    give(game, 1, (0, 2, 0, 0, 0))
    counters = [TrainingDiagnostics(0), TrainingDiagnostics(1)]
    proposal = Action(ActionType.PROPOSE_TRADE, trade=TradeOffer((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    for action in (
        proposal,
        A.ACCEPT_TRADE,
        A.REJECT_TRADE,
        A.REJECT_TRADE,
        A.CONFIRM_TRADE_WITH[1],
    ):
        apply(game, counters, action)
    initiator, partner = [counter.record() for counter in counters]
    assert (
        initiator["negotiations_completed"]
        == initiator["opening_trade_proposals"]
        == initiator["negotiations_opened"]
        == 1
    )
    assert partner["domestic_trades_as_partner"] == partner["trade_acceptances"] == 1
    assert (
        initiator["cards_gained_domestic_trade_brick"]
        == partner["cards_lost_domestic_trade_brick"]
        == 1
    )
    assert (
        initiator["cards_lost_domestic_trade_lumber"]
        == partner["cards_gained_domestic_trade_lumber"]
        == 1
    )
    assert (
        initiator["diagnostic_learner_decisions"] == 2
        and partner["diagnostic_learner_decisions"] == 1
    )


def test_counteroffers_invitations_cancellations_and_free_paid_construction():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    put_settlement(game, 0, 0)
    give(game, 0, (8, 1, 2, 3, 4))
    give(game, 1, (0, 2, 0, 0, 0))
    give_dev(game, 0, DevCard.ROAD_BUILDING)
    counters = [TrainingDiagnostics(0, game.victory_points(0)), TrainingDiagnostics(1)]
    counteroffer = Action(
        ActionType.PROPOSE_TRADE, trade=TradeOffer((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))
    )
    for action in (A.INVITE_TRADE, counteroffer, A.REJECT_TRADE, A.REJECT_TRADE, A.CANCEL_TRADE):
        apply(game, counters, action)
    apply(game, counters, A.PLAY_ROAD_BUILDING)
    while game.phase == Phase.ROAD_BUILDING:
        apply(game, counters, game.legal_actions()[0])
    apply(
        game,
        counters,
        next(action for action in game.legal_actions() if action.type == ActionType.BUILD_ROAD),
    )
    apply(game, counters, A.CITY_AT[0])
    apply(game, counters, A.BUY_DEV, DevCard.KNIGHT)
    row, partner = [counter.record() for counter in counters]
    assert row["trade_invitations"] == row["trade_cancellations"] == partner["counteroffers"] == 1
    assert row["free_roads_built"] == 2 and row["purchased_roads"] == 1 and row["roads_built"] == 3
    assert row["cities_built"] == row["dev_cards_purchased"] == row["dev_played_road_building"] == 1
    assert row["first_city_learner_turn"] == 0
    assert row["setup_roads"] == row["setup_settlements"] == 0


def test_resource_sources_and_losses_do_not_double_count_or_include_opponents():
    counter = TrainingDiagnostics(1)
    events = [
        Event(EK.RESOURCES_PRODUCED, 1, counts=(1, 2, 0, 0, 0)),
        Event(EK.STARTING_RESOURCES, 1, counts=(0, 0, 1, 0, 0)),
        Event(EK.TRADE_EXECUTED, 0, target=1, give=(1, 0, 0, 0, 0), receive=(0, 2, 0, 0, 0)),
        Event(EK.TRADE_EXECUTED, 1, target=2, give=(0, 0, 0, 1, 0), receive=(0, 0, 0, 0, 1)),
        Event(EK.MARITIME_TRADED, 1, counts=(-4, 0, 0, 0, 1)),
        Event(EK.RESOURCE_STOLEN, 1, target=0, value=4),
        Event(EK.RESOURCE_STOLEN, 2, target=1, value=1),
        Event(EK.YEAR_OF_PLENTY_TAKEN, 1, counts=(0, 0, 1, 1, 0)),
        Event(EK.MONOPOLY_TAKEN, 1, value=3, counts=(1, 0, 2, 0)),
        Event(EK.MONOPOLY_TAKEN, 2, value=4, counts=(0, 2, 0, 0)),
        Event(EK.DISCARDED, 1, value=2),
        Event(EK.RESOURCES_PRODUCED, 0, counts=(9, 9, 9, 9, 9)),
    ]
    counter.after_step(post_setup_game(), events)
    row = counter.record()
    expected = {
        "gained_dice": 3,
        "gained_setup": 1,
        "gained_domestic_trade": 2,
        "gained_maritime_trade": 1,
        "gained_robber_theft": 1,
        "gained_development": 5,
        "lost_discard": 1,
        "lost_robber_theft": 1,
        "lost_domestic_trade": 3,
        "lost_maritime_trade": 4,
        "lost_development": 2,
    }
    for source, total in expected.items():
        assert row[f"cards_{source}"] == total
        assert (
            sum(
                row[f"cards_{source}_{resource}"]
                for resource in ("lumber", "brick", "wool", "grain", "ore")
            )
            == total
        )


def test_pickled_counters_resume_and_old_environment_enabling_marks_partial():
    env = CatanEnv(
        EngineConfig(max_turns=2), EnvConfig(learner_seat=0), collect_training_diagnostics=True
    )
    env.reset(seed=17)
    env.step(env.game.legal_actions()[0])
    restored = pickle.loads(pickle.dumps(env))
    restored.configure_training_diagnostics(True, partial=True)
    assert restored.training_diagnostics.record() == env.training_diagnostics.record()
    assert restored.training_diagnostics.complete
    del env.__dict__["_collect_training_diagnostics"]
    del env.aec.__dict__["_training_diagnostics"]
    legacy = pickle.loads(pickle.dumps(env))
    assert legacy.training_diagnostics is None
    legacy.configure_training_diagnostics(True, partial=True)
    assert not legacy.training_diagnostics.complete
    assert all(
        value is None
        for key, value in legacy.training_diagnostics.record().items()
        if key.endswith("_learner_turn")
    )
    legacy.reset(seed=1)
    assert (
        legacy.training_diagnostics.complete
        and legacy.training_diagnostics.diagnostic_learner_decisions == 0
    )


def test_independent_environments_and_invalid_actions_are_not_counted():
    envs = [CatanEnv(collect_training_diagnostics=True) for _ in range(2)]
    for env in envs:
        env.reset(seed=17)
    envs[0].step(envs[0].game.legal_actions()[0])
    assert envs[0].training_diagnostics.diagnostic_learner_decisions == 1
    assert envs[1].training_diagnostics.diagnostic_learner_decisions == 0
    before = envs[1].training_diagnostics.record()
    with pytest.raises(IllegalActionError):
        envs[1].step(A.END_TURN)
    assert envs[1].training_diagnostics.record() == before


def test_won_and_truncated_episodes_and_missing_milestones():
    for engine in (EngineConfig(vp_to_win=2), EngineConfig(max_turns=2)):
        env = CatanEnv(engine, EnvConfig(learner_seat=0), collect_training_diagnostics=True)
        env.reset(seed=7)
        while not env.game.is_over:
            _, _, terminated, truncated, _ = env.step(env.game.legal_actions()[0])
        assert terminated != truncated
        row = env.training_diagnostics.record()
        assert row["final_victory_points"] == env.game.victory_points(0)
        assert row["first_settlement_learner_turn"] is None and row["vp_3_learner_turn"] is None
        assert row["setup_settlements"] == row["setup_roads"] == 2


def test_first_construction_and_vp_milestones_use_completed_learner_turns():
    game = post_setup_game()
    middle = CORNER_NEIGHBORS[0][0]
    end = next(c for c in CORNER_NEIGHBORS[middle] if c != 0)
    put_settlement(game, 0, 0)
    put_road(game, 0, edge_between(0, middle))
    put_road(game, 0, edge_between(middle, end))
    give(game, 0, (1, 1, 1, 3, 3))
    counter = TrainingDiagnostics(0, game.victory_points(0))
    apply(game, [counter], A.SETTLEMENT_AT[end])
    apply(game, [counter], A.CITY_AT[0])
    apply(game, [counter], A.END_TURN)
    row = counter.record()
    assert (
        row["first_settlement_learner_turn"]
        == row["first_city_learner_turn"]
        == row["vp_3_learner_turn"]
        == 0
    )
    assert row["vp_4_learner_turn"] is None and row["completed_learner_turns"] == 1
    # A public award can change the learner's score during another player's decision.
    game.state.settlements_left[0] = 1
    game.state.cities_left[0] = 2
    game.state.largest_army_owner = 0
    counter.after_step(game, [Event(EK.LARGEST_ARMY_CHANGED, 0, target=1)])
    row = counter.record()
    assert row["final_victory_points"] == 10
    assert row["vp_3_learner_turn"] == 0
    assert all(row[f"vp_{vp}_learner_turn"] == 1 for vp in range(4, 11))
