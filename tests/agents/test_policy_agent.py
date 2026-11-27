import argparse

import numpy as np
import pytest
import torch

from carla.agents import is_agent_name, make_agent
from carla.agents.policy_agent import PolicyAgent
from carla.cli import _agent_names
from carla.config import to_dict
from carla.config.schema import ExperimentConfig, ModelConfig
from carla.engine.config import EngineConfig
from carla.envs.catan_env import CatanAECEnv
from carla.eval.runner import play_game
from carla.models import build_policy

ENGINE = EngineConfig(allow_domestic_trade=True)
MODEL = ModelConfig(
    encoder="gnn",
    hidden_dim=32,
    gnn_layers=2,
    set_attention=True,
    belief="gru",
    belief_dim=16,
    trade_decoder="gru",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    config = ExperimentConfig(engine=ENGINE, model=MODEL)
    path = tmp_path_factory.mktemp("policy") / "latest.pt"
    torch.save({"config": to_dict(config), "policy": build_policy(MODEL).state_dict()}, path)
    return path


def _assert_same(observed, expected, path="obs"):
    if isinstance(expected, dict):
        assert observed.keys() == expected.keys(), path
        for key in expected:
            _assert_same(observed[key], expected[key], f"{path}.{key}")
    else:
        np.testing.assert_array_equal(observed, expected, err_msg=path)


def test_recurrent_policy_uses_shared_tournament_runner(checkpoint):
    from carla.rl.policy_tournament import run_policy_tournament

    result = run_policy_tournament(
        [str(checkpoint), "native-f", "heuristic", "heuristic"], boards=1, seed=552
    )
    assert len(result.records) == 16
    assert result.entries[0].games == 4


def test_trading_override_is_explicit_and_does_not_mutate_checkpoint(checkpoint):
    from dataclasses import replace

    from carla.engine.game import Game

    game = Game(replace(ENGINE, allow_domestic_trade=False), seed=3)
    with pytest.raises(ValueError, match="different engine rules"):
        PolicyAgent(checkpoint).reset(game, 0, 3)
    agent = PolicyAgent(checkpoint, allow_trading_override=True)
    agent.reset(game, 0, 3)
    assert agent.config.engine == ENGINE
    assert agent.evaluation_overrides == {"allow_domestic_trade": (True, False)}
    assert game.is_legal(agent.act(game, 0, game.legal_actions()))
    with pytest.raises(ValueError, match="different engine rules"):
        agent.reset(Game(replace(ENGINE, vp_to_win=5), seed=3), 0, 3)


def test_policy_agent_sees_the_training_observations(checkpoint):
    env = CatanAECEnv(ENGINE)
    env.reset(3)
    agents = [PolicyAgent(checkpoint) for _ in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(env.game, seat, seat)
    for _ in range(300):
        if env.game.is_over:
            break
        seat = env.actor
        expected = env.observe(seat)
        action = agents[seat].act(env.game, seat, env.game.legal_actions())
        observed = agents[seat].last_observation
        assert observed is not None
        _assert_same(observed, expected)
        env.step(action)
    assert sum(agent.fallbacks for agent in agents) == 0


def test_policy_agent_finishes_games_and_survives_seeking(checkpoint):
    agent = make_agent(f"ppo:{checkpoint}")
    assert isinstance(agent, PolicyAgent) and agent.name == f"ppo:{checkpoint}"
    result = play_game([agent, *(make_agent("heuristic") for _ in range(3))], seed=5, config=ENGINE)
    assert result.turns > 0 and agent.fallbacks == 0

    from carla.engine.game import Game

    game = Game(ENGINE, 7)
    bots = [make_agent("random") for _ in range(4)]
    for seat, bot in enumerate(bots):
        bot.reset(game, seat, seat)
    agent.reset(game, 0, 0)
    earlier = None
    while game.num_steps < 120:
        seat = game.decision_player
        if game.num_steps == 60:
            earlier = game.snapshot()
        player = agent if seat == 0 else bots[seat]
        game.step(player.act(game, seat, game.legal_actions()))
    assert earlier is not None
    game.restore(earlier)  # The GUI's step back: the agent's copy is now ahead of the game.
    while game.decision_player != 0:
        seat = game.decision_player
        game.step(bots[seat].act(game, seat, game.legal_actions()))
    assert agent.act(game, 0, game.legal_actions()) in game.legal_actions()


def test_policy_agent_rejects_other_engine_rules(checkpoint):
    from carla.engine.game import Game

    with pytest.raises(ValueError, match="different engine rules"):
        PolicyAgent(checkpoint).reset(Game(EngineConfig(), 0), 0, 0)


def test_cli_accepts_checkpoint_agents(checkpoint, tmp_path):
    assert is_agent_name(f"ppo:{checkpoint}") and not is_agent_name("ppo:")
    assert _agent_names(f"heuristic,ppo:{checkpoint}") == ("heuristic", f"ppo:{checkpoint}")
    with pytest.raises(argparse.ArgumentTypeError, match="not found"):
        _agent_names(f"ppo:{tmp_path / 'missing.pt'}")


def test_delivered_events_reproduce_the_shadow_replay(checkpoint):
    """A referee that delivers events through `observe` gives the policy exactly the
    observations its own history replay would (so history-free referees can host it)."""

    def observations(deliver):
        agent = PolicyAgent(checkpoint)
        if not deliver:
            agent.observe = lambda events, step: None
        seen = []
        original = agent.act

        def act(game, seat, legal):
            action = original(game, seat, legal)
            seen.append(agent.last_observation)
            return action

        agent.act = act
        play_game([agent, *[make_agent("heuristic") for _ in range(3)]], 21, ENGINE)
        return seen

    delivered, replayed = observations(True), observations(False)
    assert len(delivered) == len(replayed) > 10
    for a, b in zip(delivered, replayed, strict=True):
        _assert_same(a, b)


def test_referee_profile_knobs_are_overrides_not_rule_changes(checkpoint):
    from dataclasses import replace

    from carla.engine.game import Game

    agent = PolicyAgent(checkpoint)
    agent.reset(Game(replace(ENGINE, max_turns=None, catanatron_rules=True), 0), 0, 0)
    assert set(agent.evaluation_overrides) == {"catanatron_rules"}
