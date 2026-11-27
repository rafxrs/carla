from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from carla.config import EnvConfig, ExperimentConfig, load_config, to_dict
from carla.config.loader import from_dict
from carla.engine import actions
from carla.engine.config import EngineConfig
from carla.engine.constants import DESERT, PIPS, DevCard, Resource
from carla.engine.state import Phase
from carla.engine.topology import CORNER_TILES
from carla.envs import CatanEnv, rewards
from tests.engine.scenario import BOARD, give, post_setup_game, put_settlement

V2 = EnvConfig(shaping="strategic_v2", shaping_coef=2.5)
EXPERIMENT = Path("configs/experiments/full_trade_strategic_economy_v2.yaml")


@pytest.mark.parametrize(
    ("capacity", "quantity", "diversity", "production"),
    [
        ((0, 0, 0, 0, 0), 0, 0, 0),
        ((0, 0, 0.10, 0.40, 0), 0.25, 0.40, 0.295),
        ((0.10,) * 5, 0.25, 1.0, 0.475),
        ((0.08, 0.08, 0.08, 0.16, 0.10), 0.25, 0.976, 0.4678),
        ((1 / 36, 2 / 36, 3 / 36, 10 / 36, 0), 2 / 9, 0.6, 0.70 * 2 / 9 + 0.18),
        ((0, 0, 0, 0, 2.5), 1, 0.2, 0.76),
    ],
)
def test_production_components(monkeypatch, capacity, quantity, diversity, production):
    def expected(state, seat, *, robber=True):
        return capacity if seat == 0 else (0,) * 5

    monkeypatch.setattr(rewards, "expected_production", expected)
    parts = rewards.strategic_v2_components(post_setup_game(), 0, V2)
    assert parts.production_quantity == pytest.approx(quantity)
    assert parts.production_diversity == pytest.approx(diversity)
    assert parts.production == pytest.approx(production)


@pytest.mark.parametrize("city", [False, True])
def test_production_multipliers_and_robber(city):
    game = post_setup_game()
    corner, tile = next(
        (corner, tiles[0])
        for corner, tiles in enumerate(CORNER_TILES)
        if len(tiles) == 1 and BOARD.tile_resource[tiles[0]] != DESERT
    )
    put_settlement(game, 0, corner, city=city)
    pips = (2 if city else 1) * PIPS[BOARD.tile_number[tile]]
    before = rewards.strategic_v2_components(game, 0, V2)
    assert before.production_quantity == pytest.approx(pips / 72)
    assert before.production_diversity == pytest.approx(min(pips / 3, 1) / 5)
    game.state.robber_tile = tile
    after = rewards.strategic_v2_components(game, 0, V2)
    assert after.production_quantity == 0
    assert after.production_diversity == before.production_diversity


def test_resource_conservation():
    utility = [rewards.resource_conservation([cards, 0, 0, 0, 0], 7) for cards in range(20)]
    assert utility[0] == 0
    assert utility[6] == pytest.approx(6 / 13)
    assert utility[7] == pytest.approx(0.5)
    assert rewards.resource_conservation([5, 1, 0, 0, 0], 7) == utility[6]
    assert utility[3] < utility[4] < utility[5] < utility[6] < utility[7]
    increments = np.diff(utility)
    assert np.all(increments > 0)
    assert np.all(np.diff(increments) < 0)


def test_maritime_trade_reward_is_entirely_the_potential_difference():
    game = post_setup_game()
    give(game, 0, (4, 0, 1, 1, 0))
    env = CatanEnv(env_config=replace(V2, learner_seat=0))
    try:
        env.reset(seed=17)
        env.aec._game = game
        before = rewards.strategic_v2_components(game, 0, V2)
        env._potential = before.potential
        trade = actions.MARITIME[Resource.LUMBER][Resource.ORE]
        assert trade in game.legal_actions()
        _, reward, terminated, truncated, _ = env.step(trade)
        after = rewards.strategic_v2_components(game, 0, V2)
        assert not terminated and not truncated
        assert before.resource_conservation == pytest.approx(6 / 13)
        assert after.resource_conservation == pytest.approx(3 / 10)
        assert after.dev_readiness > before.dev_readiness
        assert reward == pytest.approx(V2.shaping_gamma * after.potential - before.potential)
    finally:
        env.close()


def test_awards_and_public_opponent_threat():
    game = post_setup_game()
    state = game.state
    put_settlement(game, 0, 0)
    put_settlement(game, 1, 20, city=True)
    state.hands[0] = [1, 1, 1, 1, 2]
    before = rewards.strategic_v2_components(game, 0, V2)
    state.road_lengths[0] = 4
    state.dev_played[0][DevCard.KNIGHT] = 2
    state.road_lengths[1] = 4
    state.dev_played[1][DevCard.KNIGHT] = 2
    state.hands[1] = [2, 3, 4, 5, 6]
    state.dev_hands[1][DevCard.VICTORY_POINT] = 1
    assert rewards.strategic_v2_components(game, 0, V2) == before
    assert before.opponent_strengths[0] == pytest.approx(
        (2 / 3) * 0.2 + rewards.production_strength(state, 1) / 3
    )
    assert before.opponent_weights[0] == pytest.approx(np.exp(1) / (np.exp(1) + 2))
    assert before.opponent_threat == pytest.approx(
        np.dot(before.opponent_strengths, before.opponent_weights)
    )
    assert before.potential == pytest.approx(
        2.5
        * (
            0.50 * before.victory
            + 0.245 * before.production_quantity
            + 0.105 * before.production_diversity
            + 0.010 * before.road_readiness
            + 0.040 * before.settlement_readiness
            + 0.040 * before.city_readiness
            + 0.010 * before.dev_readiness
            + 0.050 * before.resource_conservation
            - 0.35 * before.opponent_threat
        )
    )
    state.largest_army_owner = 0
    state.longest_road_owner = 0
    state.dev_hands[0][DevCard.VICTORY_POINT] = 1
    after = rewards.strategic_v2_components(game, 0, V2)
    assert after.victory - before.victory == pytest.approx(0.5)
    assert after.learner_utility - before.learner_utility == pytest.approx(0.25)


@pytest.mark.parametrize(("winner", "base"), [(0, 1), (1, -1), (-1, 0)])
def test_game_over_potential_and_base_rewards(winner, base):
    game = post_setup_game()
    game.state.phase = Phase.GAME_OVER
    game.state.winner = winner
    assert rewards.strategic_v2_components(game, 0, V2).potential == 0
    assert rewards.potential(game, 0, V2) == 0
    assert rewards.step_rewards(None, game, V2)[0] == base


def test_discounted_potential_telescopes_across_learner_transitions(monkeypatch):
    config = replace(V2, learner_seat=3)
    engine = EngineConfig(max_turns=12, allow_domestic_trade=True)
    shaped = CatanEnv(engine, config)
    sparse = CatanEnv(engine, replace(config, shaping="none"))
    rng = np.random.default_rng(17)
    calls = []
    compute = rewards.strategic_v2_components

    def counted(game, seat, env_config):
        calls.append(game.decision_player)
        return compute(game, seat, env_config)

    monkeypatch.setattr(rewards, "strategic_v2_components", counted)
    try:
        _, info = shaped.reset(seed=17)
        sparse.reset(seed=17)
        initial = previous = shaped._potential
        assert initial != 0
        discounted = 0.0
        discount = 1.0
        while not shaped.game.is_over:
            action = info["legal_actions"][rng.integers(len(info["legal_actions"]))]
            calls.clear()
            _, reward, terminated, truncated, info = shaped.step(action)
            _, base, sparse_terminated, sparse_truncated, sparse_info = sparse.step(action)
            assert (terminated, truncated) == (sparse_terminated, sparse_truncated)
            assert info["legal_actions"] == sparse_info["legal_actions"]
            assert len(calls) == int(not (terminated or truncated))
            following = shaped._potential
            assert np.isfinite(reward)
            assert reward - base == pytest.approx(config.shaping_gamma * following - previous)
            discounted += discount * (reward - base)
            discount *= config.shaping_gamma
            previous = following
        assert truncated and following == 0
        assert discounted == pytest.approx(-initial)
    finally:
        shaped.close()
        sparse.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"v2_vp_weight": float("nan")},
        {"v2_resource_conservation_weight": -0.05},
        {"v2_production_diversity_weight": 0},
        {"v2_opponent_vp_weight": 1},
        {"v2_readiness_dev_weight": 0},
        {"v2_resource_coverage_threshold": 0},
        {"v2_resource_reserve_normalization": float("inf")},
        {"shaping_coef": float("nan")},
    ],
)
def test_invalid_v2_parameters(invalid):
    with pytest.raises(ValueError, match=r"finite|sum to one"):
        replace(V2, **invalid)


def test_experiment_round_trip_legacy_defaults_and_ablations():
    config = load_config(EXPERIMENT)
    assert config.env.shaping == "strategic_v2"
    assert config.env.shaping_coef == 2.5
    assert config.env.shaping_gamma == config.ppo.gamma == 0.999
    assert from_dict(ExperimentConfig, to_dict(config)) == config
    old = load_config(EXPERIMENT.with_name("full_trade_strategic_economy_05.yaml"))
    legacy = to_dict(old)
    legacy["env"] = {
        key: value for key, value in legacy["env"].items() if not key.startswith("v2_")
    }
    assert from_dict(ExperimentConfig, legacy) == old
    assert config.model == old.model
    ablated = load_config(
        EXPERIMENT,
        [
            "env.v2_production_quantity_weight=1.0",
            "env.v2_production_diversity_weight=0.0",
            "env.v2_vp_weight=0.55",
            "env.v2_resource_conservation_weight=0.0",
        ],
    )
    assert ablated.env.v2_resource_conservation_weight == 0
    assert ablated.env.v2_production_diversity_weight == 0


def test_training_validation_checks_v2_shaping_discount():
    pytest.importorskip("torch")
    from carla.rl.train import _validate

    config = load_config(EXPERIMENT, ["env.shaping_gamma=0.99"])
    with pytest.raises(ValueError, match=r"shaping_gamma must equal ppo\.gamma"):
        _validate(config)
