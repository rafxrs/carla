"""KL regularization of the action type toward a frozen reference policy (e.g. a behaviour clone)."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from carla.config import ModelConfig, PPOConfig
from carla.engine.config import EngineConfig
from carla.envs import CatanEnv
from carla.eval.analysis import read_csv
from carla.models import build_policy
from carla.rl.ppo import PPOLearner, type_kl
from carla.rl.rollout import REFERENCE_STATE_KEY, REFERENCE_TYPE_KEY, collect_rollouts
from carla.rl.train import train
from carla.utils.logging import CsvJournal
from tests.rl.test_experiments import tiny_config

INF = -math.inf


def test_type_kl_matches_hand_computation_and_ignores_illegal_types():
    ref = torch.log(torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]))
    own = torch.log(torch.tensor([[0.25, 0.25, 0.5], [0.5, 0.5, 0.0]]))
    expected = torch.tensor([math.log(2.0), math.log(2.0)])
    torch.testing.assert_close(type_kl(ref, own), expected)
    torch.testing.assert_close(type_kl(own[:1], own[:1]), torch.zeros(1))


def test_type_kl_gradient_is_finite_with_masked_types():
    logits = torch.tensor([[1.0, 2.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    own = torch.log_softmax(logits.masked_fill(~mask, INF), dim=1)
    ref = torch.tensor([[math.log(0.9), math.log(0.1), INF]])
    type_kl(ref, own).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]  # Pulled toward the reference's choice.
    assert logits.grad[0, 2] == 0


def _gnn_belief(seed: int):
    torch.manual_seed(seed)
    return build_policy(
        ModelConfig(encoder="gnn", hidden_dim=16, mlp_layers=1, gnn_layers=1, belief="gru")
    )


def test_rollouts_store_reference_type_logp_and_carry_its_own_state():
    policy, reference = _gnn_belief(0), _gnn_belief(1)
    envs = [CatanEnv(EngineConfig(max_turns=2)) for _ in range(2)]
    batches, current = collect_rollouts(envs, policy, 48, episode_seeds=[1, 2], reference=reference)
    choices = sum(int((b.observations["masks"]["type"].sum(1) > 1).sum()) for b in batches)
    assert choices > 0  # Past setup, so the KL has real type choices to compare.
    for batch in batches:
        logp = batch.observations[REFERENCE_TYPE_KEY]
        legal = batch.observations["masks"]["type"].astype(bool)
        assert logp.shape == legal.shape
        assert np.isneginf(logp[~legal]).all() and np.isfinite(logp[legal]).all()
        np.testing.assert_allclose(np.exp(logp).sum(1), 1.0, rtol=1e-5)
        states = batch.observations[REFERENCE_STATE_KEY]
        assert not np.allclose(states[1:], 0.0)  # The reference updates its own memory.
        assert not np.allclose(states, batch.observations["belief_state"])
    assert all(REFERENCE_STATE_KEY in obs and REFERENCE_TYPE_KEY not in obs for obs in current)

    learner = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=8, reference_kl_coef=0.5))
    stats = learner.update(batches)
    assert np.isfinite(stats["reference_kl"]) and stats["reference_kl"] > 0


def test_reference_coefficient_requires_reference_rollouts():
    policy = _gnn_belief(0)
    envs = [CatanEnv(EngineConfig(max_turns=2))]
    batches, _ = collect_rollouts(envs, policy, 4, episode_seeds=[1])
    learner = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=4, reference_kl_coef=0.1))
    with pytest.raises(ValueError, match="reference"):
        learner.update(batches)


@pytest.mark.parametrize("workers", [1, 2])
def test_training_with_a_reference_checkpoint_logs_reference_kl(tmp_path, workers):
    cfg = tiny_config(tmp_path / "first")
    reference = train(cfg) / "latest.pt"
    second = tiny_config(tmp_path / "second")
    regularized = replace(
        second,
        ppo=replace(cfg.ppo, reference_checkpoint=str(reference), reference_kl_coef=0.2),
        runtime=replace(second.runtime, num_workers=workers),
    )
    rows = read_csv(train(regularized) / "updates.csv")
    assert rows and all(float(row["reference_kl"]) >= 0 for row in rows)
    with pytest.raises(ValueError, match="reference_checkpoint"):
        train(replace(cfg, ppo=replace(cfg.ppo, reference_kl_coef=0.2)))


def test_csv_journal_gains_appended_fields_but_refuses_other_changes(tmp_path):
    path = tmp_path / "metrics.csv"
    old = CsvJournal(path, ("update", "value"))
    old.write({"update": 1, "value": 2})
    upgraded = CsvJournal(path, ("update", "value", "extra"), committed_rows=1)
    upgraded.write({"update": 2, "value": 3, "extra": 4})
    assert [row["extra"] for row in read_csv(path)] == ["", "4"]
    with pytest.raises(ValueError, match="schema mismatch"):
        CsvJournal(path, ("value", "update", "extra"), committed_rows=2)
