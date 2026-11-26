import csv
import subprocess
import sys

import numpy as np
import pytest
import torch

from carla.agents.policy_agent import PolicyAgent
from carla.config.schema import (
    ExperimentConfig,
    ModelConfig,
    OpponentPoolConfig,
    OpponentSpec,
    PPOConfig,
    RuntimeConfig,
)
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.envs.action_space import HEAD_NAMES, decode_action
from carla.models import build_policy
from carla.models.base import STATE_KEY
from carla.rl.buffer import RolloutBatch
from carla.rl.imitation import (
    behaviour_clone,
    collect_demonstrations,
    expert_action,
    recompute_states,
)
from carla.rl.ppo import PPOLearner
from carla.rl.train import _load_initial_weights

MODEL = ModelConfig(
    encoder="gnn",
    hidden_dim=32,
    gnn_layers=1,
    set_attention=True,
    belief="gru",
    belief_dim=16,
    trade_decoder="gru",
    max_trade_cards_per_side=4,
)
CONFIG = ExperimentConfig(
    engine=EngineConfig(allow_domestic_trade=True, max_trade_cards_per_side=4),
    model=MODEL,
    ppo=PPOConfig(gamma=0.99),
    opponents=OpponentPoolConfig(pool=(OpponentSpec(kind="heuristic"),)),
    runtime=RuntimeConfig(device="cpu"),
)


def test_expert_offers_become_decoder_tokens():
    offer = Action(ActionType.PROPOSE_TRADE, trade=TradeOffer((1, 0, 0, 0, 0), (0, 0, 1, 0, 0)))
    encoded = expert_action(offer, 2, len(HEAD_NAMES) + 10, 4)
    assert decode_action(encoded, 2) == offer
    end = Action(ActionType.END_TURN)
    assert decode_action(expert_action(end, 2, len(HEAD_NAMES) + 10, 4), 2) == end


def test_demonstrations_are_whole_games_with_states():
    torch.manual_seed(0)
    policy = build_policy(MODEL)
    data = collect_demonstrations(CONFIG, policy, 2, seed=3)
    assert len(data.starts) == 2 and data.lengths.sum() == len(data)
    assert data.actions.shape[0] == len(data) and np.isfinite(data.returns).all()
    states = recompute_states(policy, data, torch.device("cpu"))
    assert states is not None and states.shape == (len(data), policy.state_dim)
    assert np.all(states[data.starts] == 0)
    index = np.arange(min(64, len(data)))
    mini = _tensors(_take(data.observations, index))
    mini[STATE_KEY] = torch.as_tensor(states[index])
    with torch.no_grad():
        out = policy(mini)
        logp = out.distribution(mini["masks"]).log_prob(torch.as_tensor(data.actions[index]))
    # Every expert action is legal under the stored masks: its log-probability is finite.
    assert torch.isfinite(logp).all()


def test_behaviour_clone_lowers_expert_nll_and_seeds_ppo(tmp_path):
    path = behaviour_clone(
        CONFIG, tmp_path, games=2, epochs=6, learning_rate=3e-3, minibatch_size=128, eval_boards=0
    )
    torch.manual_seed(0)
    fresh = build_policy(MODEL)
    data = collect_demonstrations(CONFIG, fresh, 2, seed=0)
    trained = build_policy(MODEL)
    _load_initial_weights(path, trained, CONFIG)
    assert _nll(trained, data) < _nll(fresh, data) - 0.2
    agent = PolicyAgent(path)
    assert agent.config.model == MODEL
    with (tmp_path / "bc_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and all(float(row["nll"]) >= 0 for row in rows)


def test_bc_before_after_report_uses_paired_games(tmp_path):
    config = ExperimentConfig(
        engine=EngineConfig(max_turns=2),
        model=ModelConfig(hidden_dim=16, mlp_layers=1),
        opponents=OpponentPoolConfig(pool=(OpponentSpec(kind="random"),)),
        runtime=RuntimeConfig(device="cpu"),
    )
    behaviour_clone(config, tmp_path / "bc", games=1, epochs=1, eval_boards=1)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_bc.py",
            str(tmp_path / "bc"),
            "--output",
            str(tmp_path / "report"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "BC figures" in completed.stdout
    assert (tmp_path / "report" / "bc_before_after.png").stat().st_size > 1000
    with (tmp_path / "report" / "bc_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and all(int(row["games"]) == 4 for row in rows)


def _nll(policy, data) -> float:
    states = recompute_states(policy, data, torch.device("cpu"))
    mini = _tensors(data.observations)
    mini[STATE_KEY] = torch.as_tensor(states)
    with torch.no_grad():
        out = policy.eval()(mini)
        return float(
            -out.distribution(mini["masks"]).log_prob(torch.as_tensor(data.actions)).mean()
        )


def _take(value, index):
    if isinstance(value, dict):
        return {k: _take(v, index) for k, v in value.items()}
    return value[index]


def _tensors(tree):
    return {k: _tensors(v) if isinstance(v, dict) else torch.as_tensor(v) for k, v in tree.items()}


@pytest.mark.parametrize("coef", [None, 0.0])
def test_trade_entropy_has_its_own_weight(coef):
    from carla.models.distributions import TradeDistribution

    torch.manual_seed(0)
    policy = build_policy(MODEL)
    data = collect_demonstrations(CONFIG, policy, 1, seed=1)
    states = recompute_states(policy, data, torch.device("cpu"))
    assert states is not None
    observations = {**data.observations, STATE_KEY: states}
    mini = _tensors(observations)
    actions = torch.as_tensor(data.actions)
    with torch.no_grad():
        out = policy(mini)
        dist = out.distribution(mini["masks"])
        assert isinstance(dist, TradeDistribution)
        logp = dist.log_prob(actions)
        torch.testing.assert_close(dist.entropy(), dist.head_entropy() + dist.token_entropy())
    rollout = RolloutBatch(
        observations=observations,
        actions=data.actions,
        log_probs=logp.numpy(),
        values=out.value.numpy(),
        rewards=np.zeros(len(data), dtype=np.float32),
        dones=np.zeros(len(data), dtype=bool),
        last_value=0.0,
    )
    config = PPOConfig(epochs=1, minibatch_size=64, trade_entropy_coef=coef)
    stats = PPOLearner(policy, config).update([rollout])
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["entropy"])
