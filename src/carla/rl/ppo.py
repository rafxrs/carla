"""Synchronous, feed-forward PPO updates for the MLP baseline."""

import numpy as np
import torch

from carla.config.schema import PPOConfig
from carla.models.base import PolicyNetwork
from carla.rl.buffer import RolloutBatch
from carla.rl.gae import compute_gae


def _concat(trees: list[dict]) -> dict:
    return {
        key: (
            _concat([tree[key] for tree in trees])
            if isinstance(trees[0][key], dict)
            else np.concatenate([tree[key] for tree in trees])
        )
        for key in trees[0]
    }


def _tensor(tree: dict, device: torch.device) -> dict:
    return {
        key: (
            _tensor(value, device)
            if isinstance(value, dict)
            else torch.as_tensor(value, device=device)
        )
        for key, value in tree.items()
    }


def _select(tree: dict, indices: torch.Tensor) -> dict:
    return {
        key: _select(value, indices) if isinstance(value, dict) else value[indices]
        for key, value in tree.items()
    }


class PPOLearner:
    """One update consumes only rollouts collected with the current policy version."""

    def __init__(self, policy: PolicyNetwork, config: PPOConfig, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.version = 0

    def update(self, batches: list[RolloutBatch]) -> dict[str, float]:
        if not batches or any(len(batch) == 0 for batch in batches):
            raise ValueError("PPO needs nonempty rollout batches")
        if any(batch.policy_version != self.version for batch in batches):
            raise ValueError("synchronous PPO requires rollouts from the current policy version")

        advantages, returns = zip(
            *(
                compute_gae(
                    batch.rewards,
                    batch.values,
                    batch.dones,
                    batch.last_value,
                    self.config.gamma,
                    self.config.gae_lambda,
                )
                for batch in batches
            ),
            strict=True,
        )
        obs = _tensor(_concat([batch.observations for batch in batches]), self.device)
        actions = torch.as_tensor(np.concatenate([b.actions for b in batches]), device=self.device)
        old_logp = torch.as_tensor(
            np.concatenate([b.log_probs for b in batches]), device=self.device
        )
        old_values = torch.as_tensor(
            np.concatenate([b.values for b in batches]), device=self.device
        )
        advantages_t = torch.as_tensor(np.concatenate(advantages), device=self.device)
        returns_t = torch.as_tensor(np.concatenate(returns), device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std(unbiased=False) + 1e-8
        )

        self.policy.train()
        stats: dict[str, list[float]] = {
            name: []
            for name in (
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
            )
        }
        early_stop = False
        for _ in range(self.config.epochs):
            for indices in torch.randperm(len(actions)).split(self.config.minibatch_size):
                indices = indices.to(self.device)
                mini = _select(obs, indices)
                out = self.policy(mini)
                dist = out.distribution(mini["masks"])
                logratio = dist.log_prob(actions[indices]) - old_logp[indices]
                ratio = logratio.exp()
                surrogate = advantages_t[indices] * ratio
                clipped = advantages_t[indices] * ratio.clamp(
                    1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                policy_loss = -torch.minimum(surrogate, clipped).mean()
                value_loss = (out.value - returns_t[indices]).square().mean()
                entropy = dist.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO gradient")
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_coef).float().mean()
                for name, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                    ("grad_norm", grad_norm),
                ):
                    stats[name].append(float(value.detach()))
                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break
        self.version += 1
        result = {name: float(np.mean(values)) for name, values in stats.items()}
        result["explained_variance"] = float(
            1
            - (returns_t - old_values).var(unbiased=False) / (returns_t.var(unbiased=False) + 1e-8)
        )
        result["epochs_stopped_early"] = float(early_stop)
        return result
