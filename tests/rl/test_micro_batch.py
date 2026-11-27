"""Gradient accumulation over micro-batches must leave the PPO update unchanged."""

import copy

import numpy as np
import pytest
import torch

from carla.config import ModelConfig, PPOConfig
from carla.engine.config import EngineConfig
from carla.envs import CatanEnv
from carla.models import build_policy
from carla.rl.ppo import PPOLearner
from carla.rl.rollout import collect_rollouts


@pytest.mark.parametrize("reference", [False, True])
def test_micro_batches_give_the_same_update(reference):
    torch.manual_seed(0)
    model = ModelConfig(encoder="gnn", hidden_dim=16, mlp_layers=1, gnn_layers=1, belief="gru")
    policy = build_policy(model)
    envs = [CatanEnv(EngineConfig(max_turns=3)) for _ in range(2)]
    batches, _ = collect_rollouts(
        envs,
        policy,
        40,
        episode_seeds=[5, 6],
        reference=build_policy(model) if reference else None,
    )
    config = PPOConfig(epochs=2, minibatch_size=32, reference_kl_coef=0.3 if reference else 0.0)
    results, weights = [], []
    for micro in (None, 7):
        learner = PPOLearner(
            copy.deepcopy(policy), PPOConfig(**{**vars(config), "micro_batch_size": micro})
        )
        torch.manual_seed(1)  # Same minibatch order for both.
        results.append(learner.update(batches))
        weights.append(learner.policy.state_dict())
    for key, value in weights[0].items():
        torch.testing.assert_close(value, weights[1][key], rtol=1e-4, atol=1e-5)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"):
        np.testing.assert_allclose(results[0][name], results[1][name], rtol=1e-4, atol=1e-6)
    assert results[0]["optimizer_steps"] == results[1]["optimizer_steps"]
    if reference:
        np.testing.assert_allclose(
            results[0]["reference_kl"], results[1]["reference_kl"], rtol=1e-4
        )
