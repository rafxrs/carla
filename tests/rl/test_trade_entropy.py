from copy import deepcopy

import numpy as np
import pytest
import torch

from carla.config.schema import ModelConfig, PPOConfig
from carla.engine.actions import ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.envs.action_space import HEAD_NAMES
from carla.envs.observation import ObservationEncoder
from carla.envs.trade_tokens import offer_tokens
from carla.models.base import batch_observations
from carla.models.factory import build_policy
from carla.rl.buffer import RolloutBatch
from carla.rl.ppo import PPOLearner
from tests.engine.scenario import give, post_setup_game


def data(*, trade=True, proposals=True):
    torch.manual_seed(11)
    policy = build_policy(
        ModelConfig(
            hidden_dim=16,
            mlp_layers=1,
            trade_decoder="gru" if trade else "none",
            max_trade_cards_per_side=2,
        )
    )
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 2, 1, 1, 1))
    obs = ObservationEncoder().encode(game, 0)
    obs["masks"]["type"][:] = False
    obs["masks"]["type"][[ActionType.END_TURN, ActionType.PROPOSE_TRADE]] = True
    batch = batch_observations([obs] * 4)
    with torch.no_grad():
        output = policy(batch)
        width = len(HEAD_NAMES) + (policy.trade_decoder.max_tokens if trade else 0)
        actions = torch.full((4, width), -1, dtype=torch.long)
        actions[:, 0] = ActionType.END_TURN
        if proposals:
            actions[::2, 0] = ActionType.PROPOSE_TRADE
            if trade:
                tokens = offer_tokens(TradeOffer((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)), 2)
                actions[::2, len(HEAD_NAMES) : len(HEAD_NAMES) + len(tokens)] = torch.tensor(tokens)
            else:
                actions[::2, 5:7] = torch.tensor([0, 1])
        dist = output.distribution(batch["masks"])
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean().item()
        token = dist.trade_token_entropy().mean().item()
    arrays = {
        key: {k: v.numpy() for k, v in value.items()} if isinstance(value, dict) else value.numpy()
        for key, value in batch.items()
    }
    rollout = RolloutBatch(
        arrays,
        actions.numpy(),
        logp.numpy(),
        output.value.numpy(),
        np.array([1, 0, -1, 1], dtype=np.float32),
        np.array([False, False, False, True]),
        0.0,
    )
    return policy, rollout, entropy, token


def update(policy, rollout, coefficient):
    learner = PPOLearner(
        deepcopy(policy), PPOConfig(epochs=1, minibatch_size=4, trade_entropy_coef=coefficient)
    )
    torch.manual_seed(123)
    stats = learner.update([rollout])
    assert all(np.isfinite(value) for value in stats.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in learner.policy.parameters())
    return learner, stats


def test_entropy_objective_none_equal_and_separate_coefficients():
    policy, rollout, entropy, token = data()
    legacy, original = update(policy, rollout, None)
    equal, equal_stats = update(policy, rollout, 0.01)
    separate, changed = update(policy, rollout, 0.001)
    assert token > 0 and entropy > token
    assert original["entropy"] == pytest.approx(entropy)
    assert original["entropy_bonus"] == pytest.approx(0.01 * entropy)
    assert equal_stats["entropy_bonus"] == pytest.approx(original["entropy_bonus"])
    for a, b in zip(legacy.policy.parameters(), equal.policy.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    assert changed["entropy"] == pytest.approx(entropy)
    assert changed["trade_token_entropy"] == pytest.approx(token)
    assert changed["entropy_bonus"] == pytest.approx(0.01 * (entropy - token) + 0.001 * token)
    assert changed["trade_proposal_fraction"] == 0.5
    assert changed["trade_token_entropy_conditional"] == pytest.approx(2 * token)
    for key in ("policy_loss", "value_loss", "auxiliary_loss"):
        assert changed[key] == pytest.approx(original[key])
    assert separate.policy.trade_decoder.output.weight.grad is not None


def test_each_entropy_component_has_its_own_finite_gradients():
    policy, rollout, _, _ = data()
    batch = {
        key: (
            {k: torch.as_tensor(v) for k, v in value.items()}
            if isinstance(value, dict)
            else torch.as_tensor(value)
        )
        for key, value in rollout.observations.items()
    }
    output = policy(batch)
    distribution = output.distribution(batch["masks"])
    distribution.log_prob(torch.as_tensor(rollout.actions))
    tokens = distribution.trade_token_entropy().mean()
    ordinary = distribution.entropy().mean() - tokens
    (token_gradient,) = torch.autograd.grad(
        tokens, policy.trade_decoder.output.weight, retain_graph=True
    )
    (ordinary_gradient,) = torch.autograd.grad(ordinary, output.logits["type"])
    assert torch.isfinite(token_gradient).all() and token_gradient.abs().sum() > 0
    assert torch.isfinite(ordinary_gradient).all() and ordinary_gradient.abs().sum() > 0


@pytest.mark.parametrize(("trade", "proposals"), [(True, False), (False, False)])
def test_no_proposals_and_baseline_distributions(trade, proposals):
    policy, rollout, entropy, token = data(trade=trade, proposals=proposals)
    _, stats = update(policy, rollout, 0.001)
    assert token == stats["trade_token_entropy"] == stats["trade_token_entropy_conditional"] == 0
    assert stats["entropy_bonus"] == pytest.approx(0.01 * entropy)


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf")])
def test_invalid_entropy_coefficients(coefficient):
    with pytest.raises(ValueError, match="trade_entropy_coef"):
        PPOConfig(trade_entropy_coef=coefficient)
