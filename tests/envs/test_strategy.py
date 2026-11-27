"""Strategy statistic z, pseudo-rewards, selection schedule and planning features."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from carla.agents import make_agent
from carla.config.schema import EnvConfig, StrategyConfig
from carla.engine.events import Event, EventKind
from carla.engine.state import Phase
from carla.envs import CatanEnv
from carla.envs.catan_env import CatanAECEnv
from carla.envs.rewards import expected_production
from carla.envs.strategy import (
    BUILD_ORDER_LENGTH,
    NONE,
    SELECT,
    Z_DIM,
    Z_PRESENT,
    Z_REWARD,
    StrategyController,
    StrategyLibrary,
    StrategyStats,
    encode_z,
    levenshtein,
    make_controller,
    reward_decay,
    strategy_distance,
    track_step,
    z_build_order,
)
from carla.envs.strategy_features import STRATEGY_FEATURE_KEYS, production_by_resource


def _built(kind: EventKind, seat: int = 0) -> Event:
    return Event(kind, player=seat)


def test_stats_skip_setup_and_track_purchases_plays_and_awards():
    stats = [StrategyStats() for _ in range(4)]
    track_step(
        stats,
        [_built(EventKind.SETTLEMENT_BUILT), _built(EventKind.ROAD_BUILT)],
        Phase.SETUP_SETTLEMENT,
    )
    assert stats[0].build_order == [] and stats[0].counts.sum() == 0
    track_step(stats, [_built(EventKind.ROAD_BUILT)], Phase.MAIN)
    track_step(stats, [_built(EventKind.CITY_BUILT)], Phase.MAIN)
    track_step(stats, [Event(EventKind.DEV_CARD_BOUGHT, player=0, value=-1)], Phase.MAIN)
    track_step(stats, [Event(EventKind.DEV_CARD_PLAYED, player=0, value=0)], Phase.PRE_ROLL)
    track_step(stats, [Event(EventKind.LONGEST_ROAD_CHANGED, player=0, target=-1)], Phase.MAIN)
    assert stats[0].build_order == [0, 2, 3]
    assert stats[0].counts.tolist() == [0, 1, 1, 1, 1, 1, 0]
    track_step(stats, [Event(EventKind.LONGEST_ROAD_CHANGED, player=2, target=0)], Phase.MAIN)
    assert stats[0].counts[5] == 0 and stats[2].counts[5] == 1
    assert stats[1].counts.sum() == 0


def test_z_layout_and_build_order_round_trip():
    stats = StrategyStats([1, 1, 2, 3, 0], np.asarray([2, 1, 1, 1, 0, 0, 1], dtype=np.float32))
    z = encode_z(stats, reward_active=False)
    assert z.shape == (Z_DIM,) and z[Z_PRESENT] == 1 and z[Z_REWARD] == 0
    assert z_build_order(z) == [1, 1, 2, 3, 0]


def test_levenshtein_and_distance_are_zero_for_the_same_play():
    assert levenshtein([0, 1, 2], [0, 1, 2]) == 0
    assert levenshtein([0, 1, 2], [0, 2]) == 1
    assert levenshtein([], [3, 3]) == 2
    stats = StrategyStats([1, 2], np.asarray([1, 1, 0, 0, 0, 0, 0], dtype=np.float32))
    assert strategy_distance(stats, encode_z(stats), 1, 1) == pytest.approx(0)
    other = StrategyStats([3, 3], np.asarray([0, 0, 0, 2, 0, 0, 0], dtype=np.float32))
    assert strategy_distance(other, encode_z(stats), 1, 1) > 0
    assert strategy_distance(other, np.zeros(Z_DIM, dtype=np.float32), 1, 1) == 0


def test_pseudo_rewards_telescope_to_the_distance_change_and_charge_switches():
    config = StrategyConfig(decay_turns=(1000,), switch_cost=0.1)
    target = StrategyStats([1, 2, 2], np.asarray([1, 2, 0, 0, 0, 0, 0], dtype=np.float32))
    library = StrategyLibrary({a: encode_z(target)[None] for a in config.archetypes})
    controller = StrategyController(
        config, "prescribed", library=library, archetype="city", reward_active=True
    )

    class G:
        turn = 5

    stats = StrategyStats()
    start = controller.adherence(stats)
    total = controller.reward(G, stats)
    for kind in (1, 2, 2):
        stats.build_order.append(kind)
        stats.counts[0 if kind == 1 else 1] += 1
        total += controller.reward(G, stats)
    assert total == pytest.approx(start - controller.adherence(stats))
    assert controller.adherence(stats) == pytest.approx(0, abs=1e-6)
    inactive = StrategyController(config, "prescribed", library=library, archetype="city")
    assert inactive.reward(G, stats) == 0.0
    chooser = StrategyController(config, SELECT, library=library, reward_active=True)
    chooser.select(0, G, stats)
    chooser.reward(G, stats)
    chooser.select(1, G, stats)
    assert chooser.switches == 1
    assert chooser.reward(G, stats) == pytest.approx(-0.1)


def test_reward_decay_halves_then_stops():
    assert reward_decay(10, (40, 80, 120)) == 1.0
    assert reward_decay(40, (40, 80, 120)) == 0.5
    assert reward_decay(80, (40, 80, 120)) == 0.25
    assert reward_decay(130, (40, 80, 120)) == 0.0


def test_selection_happens_at_the_first_decision_and_every_reconsider_turns_at_the_roll():
    config = StrategyConfig(reconsider_turns=2)
    library = StrategyLibrary({a: np.zeros((1, Z_DIM), np.float32) for a in config.archetypes})
    controller = StrategyController(config, SELECT, library=library)
    env = CatanEnv(
        env_config=EnvConfig(learner_seat=0), opponents=[make_agent("heuristic") for _ in range(3)]
    )
    env.reset(seed=3)
    selections = []
    while not env.game.is_over and env.game.turn < 40:
        game = env.game
        if controller.needs_selection(game, 0):
            assert game.phase == Phase.PRE_ROLL or not selections
            controller.select(1, game, env.aec.stats[0])
            selections.append(game.turn)
        legal = game.legal_actions()
        env.step(legal[0])
    assert selections[0] == 0
    assert all(b - a >= 8 for a, b in pairwise(selections))
    assert len(selections) >= 3
    assert make_controller(config, NONE, library, np.random.default_rng(0)).label == NONE


def test_library_round_trip(tmp_path):
    library = StrategyLibrary({"city": np.ones((3, Z_DIM), np.float32)})
    path = library.save(tmp_path / "lib.npz")
    loaded = StrategyLibrary.load(path)
    z = loaded.sample("city", np.random.default_rng(0), reward_active=False)
    assert z[Z_PRESENT] == 1 and z[Z_REWARD] == 0
    with pytest.raises(ValueError, match="shape"):
        StrategyLibrary({"city": np.ones((3, 4), np.float32)})


def test_planning_features_use_only_public_information_and_own_cards():
    env = CatanAECEnv(env_config=EnvConfig(strategic_features=True))
    env.reset(seed=11)
    bots = [make_agent("heuristic_city") for _ in range(4)]
    for seat, bot in enumerate(bots):
        bot.reset(env.game, seat, seat)
    for _ in range(200):
        seat = env.actor
        env.step(bots[seat].act(env.game, seat, env.game.legal_actions()))
    observer = env.actor
    before = env.observe(observer)
    assert all(key in before for key in STRATEGY_FEATURE_KEYS)
    state = env.game.state
    for p in range(4):
        if p != observer:
            state.hands[p] = [9, 0, 9, 0, 9]
            state.dev_hands[p] = [3, 0, 0, 0, 2]
    env.game.set_state(state)
    after = env.encoder.encode(
        env.game, observer, (), stats=env.stats, last_action_type=env.last_action_types[observer]
    )
    for key in STRATEGY_FEATURE_KEYS:
        np.testing.assert_array_equal(before[key], after[key])
    total = production_by_resource(env.game, observer).sum()
    assert total == pytest.approx(expected_production(env.game, observer))


def test_strategy_reward_reaches_the_env_info():
    config = StrategyConfig()
    stats = StrategyStats([1], np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    library = StrategyLibrary({a: encode_z(stats)[None] for a in config.archetypes})
    env = CatanEnv(
        env_config=replace(EnvConfig(), strategic_features=True),
        opponents=[make_agent("random") for _ in range(3)],
    )
    env.strategy = StrategyController(
        config, "prescribed", library=library, archetype="city", reward_active=True
    )
    env.reset(seed=0)
    info = env.step(env.action_space.sample())[4]
    assert "strategy_reward" in info
    assert len(z_build_order(encode_z(stats))) <= BUILD_ORDER_LENGTH
