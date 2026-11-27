"""Synchronous PPO updates.

Recurrent (belief) policies store their state before each decision in the observation, so each
minibatch recomputes single decisions from stored states (truncated BPTT of one decision).
"""

import numpy as np
import torch

from carla.config.schema import PPOConfig
from carla.models.base import PolicyNetwork
from carla.models.distributions import TradeDistribution
from carla.rl.buffer import RolloutBatch
from carla.rl.gae import compute_gae
from carla.rl.rollout import REFERENCE_TYPE_KEY


def type_kl(reference_logp: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """Per-row KL(reference || policy) between masked log-probability rows (-inf = illegal).

    Types the reference never plays contribute nothing; both rows share the decision's mask, so
    the policy is finite wherever the reference has probability."""
    p_ref = reference_logp.exp()
    played = p_ref > 0
    ref = torch.where(played, reference_logp, 0.0)
    own = torch.where(played, logp, 0.0)
    return (p_ref * (ref - own)).sum(1)


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
        raw_stats = {
            "advantage_mean": float(advantages_t.mean()),
            "advantage_std": float(advantages_t.std(unbiased=False)),
            "return_mean": float(returns_t.mean()),
            "return_std": float(returns_t.std(unbiased=False)),
            "value_mean": float(old_values.mean()),
            "value_std": float(old_values.std(unbiased=False)),
        }
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
                "auxiliary_loss",
            )
        }
        reference_coef = self.config.reference_kl_coef
        if reference_coef > 0:
            if REFERENCE_TYPE_KEY not in obs:
                raise ValueError("ppo.reference_kl_coef needs rollouts collected with a reference")
            stats["reference_kl"] = []
        early_stop = False
        chunk_size = self.config.micro_batch_size or self.config.minibatch_size
        for _ in range(self.config.epochs):
            for indices in torch.randperm(len(actions)).split(self.config.minibatch_size):
                indices = indices.to(self.device)
                # Gradient accumulation: each chunk's loss is a mean, so weighting it by the
                # chunk's share of the minibatch gives the same gradient as one pass.
                self.optimizer.zero_grad(set_to_none=True)
                parts: dict[str, float] = {}
                for chunk in indices.split(chunk_size):
                    weight = len(chunk) / len(indices)
                    loss, terms = self._loss(
                        _select(obs, chunk),
                        actions[chunk],
                        old_logp[chunk],
                        advantages_t[chunk],
                        returns_t[chunk],
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite PPO loss")
                    (loss * weight).backward()
                    for name, value in terms.items():
                        parts[name] = parts.get(name, 0.0) + weight * float(value)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO gradient")
                self.optimizer.step()
                parts["grad_norm"] = float(grad_norm)
                for name, value in parts.items():
                    stats[name].append(value)
                if self.config.target_kl is not None and parts["approx_kl"] > self.config.target_kl:
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
        result["optimizer_steps"] = len(stats["policy_loss"])
        result.update(raw_stats)
        return result

    def _loss(
        self,
        mini: dict,
        actions: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """PPO loss (batch means) of some decisions, plus detached terms for the update stats."""
        out = self.policy(mini)
        dist = out.distribution(mini["masks"])
        logratio = dist.log_prob(actions) - old_logp
        ratio = logratio.exp()
        surrogate = advantages * ratio
        clipped = advantages * ratio.clamp(1 - self.config.clip_coef, 1 + self.config.clip_coef)
        policy_loss = -torch.minimum(surrogate, clipped).mean()
        value_loss = (out.value - returns).square().mean()
        entropy = dist.entropy().mean()
        trade_coef = self.config.trade_entropy_coef
        if trade_coef is None or not isinstance(dist, TradeDistribution):
            bonus = self.config.entropy_coef * entropy
        else:
            bonus = (
                self.config.entropy_coef * dist.head_entropy().mean()
                + trade_coef * dist.token_entropy().mean()
            )
        loss = policy_loss + self.config.value_coef * value_loss - bonus
        terms = {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}
        if self.config.reference_kl_coef > 0:
            reference_kl = type_kl(mini[REFERENCE_TYPE_KEY], dist.logp["type"]).mean()
            loss = loss + self.config.reference_kl_coef * reference_kl
            terms["reference_kl"] = reference_kl
        auxiliary = self.policy.auxiliary_loss(out, mini)
        if auxiliary is not None:
            loss = loss + auxiliary
        else:
            auxiliary = torch.zeros((), device=self.device)
        terms["auxiliary_loss"] = auxiliary
        with torch.no_grad():
            terms["approx_kl"] = ((ratio - 1) - logratio).mean()
            terms["clip_fraction"] = ((ratio - 1).abs() > self.config.clip_coef).float().mean()
        return loss, {name: value.detach() for name, value in terms.items()}
