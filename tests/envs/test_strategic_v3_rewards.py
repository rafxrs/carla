import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from carla.config import EnvConfig, ExperimentConfig, load_config, to_dict
from carla.config.loader import from_dict
from carla.engine.config import EngineConfig
from carla.engine.constants import DESERT, PIPS, Building
from carla.engine.rules import road_edges, settlement_corners
from carla.engine.state import Phase
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS, CORNER_TILES
from carla.envs import CatanEnv, rewards
from tests.engine.scenario import edge_between, give, post_setup_game, put_road, put_settlement

V3 = EnvConfig(shaping="strategic_v3", shaping_coef=2.5)


def test_opening_uses_both_starting_settlements_and_survives_checkpoint():
    # Use seat 0 to check both sides of the global transition out of setup.
    env = CatanEnv(EngineConfig(max_turns=20), replace(V3, learner_seat=0))
    _, info = env.reset(seed=23)
    assert rewards.opening_quality(env.game, 0, V3) == 0
    corners = []
    while env.game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        action = info["legal_actions"][0]
        if env.game.phase == Phase.SETUP_SETTLEMENT:
            corners.append(action.corner)
        _, _, _, _, info = env.step(action)
        capacity = [0] * 5
        for corner in corners:
            for tile in CORNER_TILES[corner]:
                resource = env.game.board.tile_resource[tile]
                if resource != DESERT:
                    capacity[resource] += PIPS[env.game.board.tile_number[tile]]
        expected = 0.75 * min(sum(capacity) / 30, 1) + 0.25 * sum(
            min(pips / 3, 1) / 5 for pips in capacity
        )
        assert rewards.opening_quality(env.game, 0, V3) == pytest.approx(expected)
    assert len(corners) == 2
    restored = pickle.loads(pickle.dumps(env))
    assert restored._potential == env._potential
    assert rewards.opening_quality(restored.game, 0, V3) == pytest.approx(expected)
    state = restored.game.state
    state.corner_building[corners[0]] = Building.CITY
    state.robber_tile = CORNER_TILES[corners[1]][0]
    for turn, alpha in [(0, 0.50), (8, 0.25), (16, 0), (17, 0)]:
        state.turn = turn
        v2 = rewards.strategic_v2_components(restored.game, 0, V3)
        v3 = rewards.strategic_v3_components(restored.game, 0, V3)
        assert rewards.opening_quality(restored.game, 0, V3) == pytest.approx(expected)
        assert v3.production == pytest.approx((1 - alpha) * v2.production + alpha * expected)
    restored.reset(seed=24)
    assert rewards.opening_quality(restored.game, 0, V3) == 0
    restored.close()
    env.close()


def test_road_readiness_requires_a_new_reachable_settlement_site():
    game = post_setup_game()
    state = game.state
    start = 0
    middle = CORNER_NEIGHBORS[start][0]
    end = next(c for c in CORNER_NEIGHBORS[middle] if c != start)
    first = edge_between(start, middle)
    candidate = edge_between(middle, end)
    put_settlement(game, 0, start)
    put_road(game, 0, first)
    give(game, 0, (1, 1, 0, 0, 0))
    for edge in range(len(state.edge_owner)):
        if edge not in (first, candidate):
            state.edge_owner[edge] = 1
    assert road_edges(state, 0) == [candidate]
    assert end not in settlement_corners(state, 0)
    assert rewards.strategic_v3_components(game, 0, V3).road_readiness == 1

    state.hands[0][1] = 0
    assert rewards.strategic_v3_components(game, 0, V3).road_readiness == 0.5
    state.hands[0][1] = 1
    state.settlements_left[0] = 0
    assert rewards.strategic_v3_components(game, 0, V3).road_readiness == 0
    state.settlements_left[0] = 4

    existing = next(edge for edge in CORNER_EDGES[end] if edge != candidate)
    state.edge_owner[existing] = 0
    assert end in settlement_corners(state, 0)
    assert candidate in road_edges(state, 0)
    assert rewards.strategic_v3_components(game, 0, V3).road_readiness == 0
    assert rewards.strategic_v2_components(game, 0, V3).road_readiness == 1


def test_learner_transition_shaping_telescopes_at_truncation():
    config = replace(V3, learner_seat=3)
    engine = EngineConfig(max_turns=4, allow_domestic_trade=True)
    shaped = CatanEnv(engine, config)
    sparse = CatanEnv(engine, replace(config, shaping="none"))
    try:
        _, info = shaped.reset(seed=19)
        sparse.reset(seed=19)
        initial = previous = rewards.potential(shaped.game, 3, config)
        assert initial != 0
        discounted = 0.0
        discount = 1.0
        while not shaped.game.is_over:
            action = info["legal_actions"][0]
            _, reward, terminated, truncated, info = shaped.step(action)
            _, base, sparse_terminated, sparse_truncated, _ = sparse.step(action)
            assert (terminated, truncated) == (sparse_terminated, sparse_truncated)
            following = rewards.potential(shaped.game, 3, config)
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
    "overrides", [{"v3_opening_blend": float("nan")}, {"v3_opening_taper_turns": 0}]
)
def test_invalid_opening_config(overrides):
    with pytest.raises(ValueError, match="v3_opening_"):
        replace(V3, **overrides)


def test_matched_experiment_and_legacy_config_round_trip():
    configs = Path("configs/experiments")
    config = load_config(configs / "full_trade_strategic_v3_3h75_seed4.yaml")
    assert from_dict(ExperimentConfig, to_dict(config)) == config
    assert config.env.shaping == "strategic_v3"
    assert config.ppo.total_env_steps == 250_880
    assert config.ppo.gamma == config.env.shaping_gamma == 0.999
    assert config.ppo.gae_lambda == 0.95 and config.seed == 4

    old = load_config(configs / "full_trade_strategic_economy_v2.yaml")
    serialized = to_dict(old)
    del serialized["env"]["v3_opening_blend"]
    del serialized["env"]["v3_opening_taper_turns"]
    assert from_dict(ExperimentConfig, serialized) == old
