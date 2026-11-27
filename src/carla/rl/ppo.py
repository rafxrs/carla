"""Synchronous PPO updates.

Recurrent policies store their state before each decision in the observation. With
`ppo.bptt: step`, each minibatch recomputes single decisions from stored states (truncated BPTT
of one decision). With `ppo.bptt: sequence`, minibatches are chunks of `ppo.sequence_length`
consecutive decisions of one environment: the network unrolls each chunk from the stored state
at its first decision and restarts from zeros at episode boundaries inside the chunk.

Objectives (kept separate, each with its own coefficient):
- outcome: win/loss reward plus any potential-based shaping (`env.shaping`, which leaves the
  optimal policy unchanged), advantage from GAE on the outcome critic;
- strategy pseudo-reward (`carla.envs.strategy`): its own GAE and critic; its advantage is
  added with weight `strategy_coef` (it deliberately changes the optimum toward z);
- UPGO (`ppo.upgo_coef`): an upgoing-return policy update on the outcome stream (AlphaStar);
- imitation regularization: KL(reference || policy) over the action type or every used head;
- the selector's log-probability is part of the action's on selection decisions, so it is
  trained by the same clipped surrogate; it has its own entropy bonus.
"""

import numpy as np
import torch

from carla.config.schema import PPOConfig
from carla.envs.action_space import HEAD_INDEX
from carla.envs.strategy import SELECT_KEY
from carla.models.base import STATE_KEY, PolicyNetwork
from carla.models.distributions import _SUBS, TradeDistribution
from carla.rl.acting import selection_entropy, selection_log_prob
from carla.rl.buffer import RolloutBatch
from carla.rl.gae import compute_gae, compute_upgo
from carla.rl.rollout import REFERENCE_LOGP_KEY, REFERENCE_TYPE_KEY


def type_kl(reference_logp: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """Per-row KL(reference || policy) between masked log-probability rows (-inf = illegal).

    Types the reference never plays contribute nothing; both rows share the decision's mask, so
    the policy is finite wherever the reference has probability."""
    p_ref = reference_logp.exp()
    played = p_ref > 0
    ref = torch.where(played, reference_logp, 0.0)
    own = torch.where(played, logp, 0.0)
    return (p_ref * (ref - own)).sum(1)


def head_kl(reference: dict, dist, actions: torch.Tensor) -> torch.Tensor:
    """KL(reference || policy) over the type and every sub-head the taken action uses."""
    total = type_kl(reference["type"], dist.logp["type"])
    action_type, dev = actions[:, 0].long(), actions[:, HEAD_INDEX["dev"]].long()
    for key, (selecting, card, _) in _SUBS.items():
        if key == "trade" and isinstance(dist, TradeDistribution):
            continue  # Offers come from the decoder, not the one-for-one head.
        used = action_type == selecting
        if card is not None:
            used &= dev == card
        if used.any():
            kl = type_kl(reference[key].reshape(len(actions), -1), dist.logp[key])
            total = total + torch.where(used, kl, 0.0)
    return total


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
    """One update consumes only rollouts collected with the current policy version.

    `strategy_coef` weights the strategy advantage and `selector_entropy_coef` the selector's
    entropy (both from `StrategyConfig`; 0 without strategies).
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        config: PPOConfig,
        device: str = "cpu",
        *,
        strategy_coef: float = 0.0,
        selector_entropy_coef: float = 0.0,
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.version = 0
        self.strategy_coef = strategy_coef
        self.selector_entropy_coef = selector_entropy_coef

    @property
    def sequence_mode(self) -> bool:
        return self.config.bptt == "sequence" and self.policy.recurrent

    def update(self, batches: list[RolloutBatch]) -> dict[str, float]:
        if not batches or any(len(batch) == 0 for batch in batches):
            raise ValueError("PPO needs nonempty rollout batches")
        if any(batch.policy_version != self.version for batch in batches):
            raise ValueError("synchronous PPO requires rollouts from the current policy version")
        cfg = self.config
        advantages, returns = zip(
            *(
                compute_gae(b.rewards, b.values, b.dones, b.last_value, cfg.gamma, cfg.gae_lambda)
                for b in batches
            ),
            strict=True,
        )
        strategy = all(b.strategy_rewards is not None for b in batches)
        dev = self.device
        obs = _tensor(_concat([batch.observations for batch in batches]), dev)
        actions = torch.as_tensor(np.concatenate([b.actions for b in batches]), device=dev)
        old_logp = torch.as_tensor(np.concatenate([b.log_probs for b in batches]), device=dev)
        old_values = torch.as_tensor(np.concatenate([b.values for b in batches]), device=dev)
        advantages_t = torch.as_tensor(np.concatenate(advantages), device=dev)
        returns_t = torch.as_tensor(np.concatenate(returns), device=dev)
        raw_stats = {
            "advantage_mean": float(advantages_t.mean()),
            "advantage_std": float(advantages_t.std(unbiased=False)),
            "return_mean": float(returns_t.mean()),
            "return_std": float(returns_t.std(unbiased=False)),
            "value_mean": float(old_values.mean()),
            "value_std": float(old_values.std(unbiased=False)),
        }
        combined = advantages_t
        strategy_returns = None
        if strategy:
            s_adv, s_ret = zip(
                *(
                    compute_gae(
                        b.strategy_rewards,
                        b.strategy_values,
                        b.dones,
                        b.last_strategy_value,
                        cfg.gamma,
                        cfg.gae_lambda,
                    )
                    for b in batches
                ),
                strict=True,
            )
            strategy_adv = torch.as_tensor(np.concatenate(s_adv), device=dev)
            strategy_returns = torch.as_tensor(np.concatenate(s_ret), device=dev)
            combined = advantages_t + self.strategy_coef * strategy_adv
            raw_stats["strategy_return_mean"] = float(strategy_returns.mean())
            raw_stats["strategy_advantage_std"] = float(strategy_adv.std(unbiased=False))
        combined = (combined - combined.mean()) / (combined.std(unbiased=False) + 1e-8)
        upgo = None
        if cfg.upgo_coef > 0:
            upgo_returns = [
                compute_upgo(b.rewards, b.values, b.dones, b.last_value, cfg.gamma) for b in batches
            ]
            upgo = torch.as_tensor(np.concatenate(upgo_returns), device=dev) - old_values
            upgo = upgo / (upgo.std(unbiased=False) + 1e-8)
        if SELECT_KEY in obs:
            raw_stats["selections"] = float((obs[SELECT_KEY] > 0).sum())

        targets = {
            "actions": actions,
            "old_logp": old_logp,
            "advantages": combined,
            "returns": returns_t,
        }
        if strategy_returns is not None:
            targets["strategy_returns"] = strategy_returns
        if upgo is not None:
            targets["upgo"] = upgo

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
        reference_coef = cfg.reference_kl_coef
        if reference_coef > 0:
            needed = REFERENCE_LOGP_KEY if cfg.reference_kl_heads == "all" else REFERENCE_TYPE_KEY
            if needed not in obs:
                raise ValueError("ppo.reference_kl_coef needs rollouts collected with a reference")
            stats["reference_kl"] = []
        early_stop = False
        units = self._units(batches)
        for _ in range(cfg.epochs):
            for group in self._minibatches(units):
                self.optimizer.zero_grad(set_to_none=True)
                parts: dict[str, float] = {}
                total_rows = sum(len(rows) for rows, _ in group)
                for rows, info in group:
                    weight = len(rows) / total_rows
                    rows_t = rows.to(dev)
                    mini = _select(obs, rows_t)
                    if info is None:
                        out = self.policy(mini)
                    else:
                        starts, resets, steps = info
                        state0 = obs[STATE_KEY][starts.to(dev)].float()
                        out = self.policy.unroll(mini, state0, resets.to(dev), steps)
                    loss, terms = self._loss(out, mini, {k: v[rows_t] for k, v in targets.items()})
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite PPO loss")
                    (loss * weight).backward()
                    for name, value in terms.items():
                        parts[name] = parts.get(name, 0.0) + weight * float(value)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO gradient")
                self.optimizer.step()
                parts["grad_norm"] = float(grad_norm)
                for name, value in parts.items():
                    stats.setdefault(name, []).append(value)
                if cfg.target_kl is not None and parts["approx_kl"] > cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break
        self.version += 1
        result = {name: float(np.mean(values)) for name, values in stats.items() if values}
        result["explained_variance"] = float(
            1
            - (returns_t - old_values).var(unbiased=False) / (returns_t.var(unbiased=False) + 1e-8)
        )
        result["epochs_stopped_early"] = float(early_stop)
        result["optimizer_steps"] = len(stats["policy_loss"])
        result.update(raw_stats)
        return result

    # ------------------------------------------------------------------ minibatch layout

    def _units(self, batches: list[RolloutBatch]) -> list:
        """Sampling units: single decisions (step mode) or chunks (sequence mode)."""
        if not self.sequence_mode:
            return list(range(sum(len(b) for b in batches)))
        length = self.config.sequence_length
        chunks, offset = [], 0
        for batch in batches:
            dones = batch.dones
            for start in range(0, len(batch), length):
                size = min(length, len(batch) - start)
                resets = np.zeros(size, dtype=bool)
                resets[1:] = dones[start : start + size - 1] > 0
                chunks.append((offset + start, size, resets))
            offset += len(batch)
        return chunks

    def _minibatches(self, units: list):
        """Yield minibatches as lists of (row indices, unroll info) micro-chunks."""
        cfg = self.config
        if not self.sequence_mode:
            rows = torch.randperm(len(units))
            for indices in rows.split(cfg.minibatch_size):
                chunk = cfg.micro_batch_size or cfg.minibatch_size
                yield [(part, None) for part in indices.split(chunk)]
            return
        length = cfg.sequence_length
        per_minibatch = max(1, cfg.minibatch_size // length)
        per_micro = max(1, (cfg.micro_batch_size or cfg.minibatch_size) // length)
        by_size: dict[int, list] = {}
        for unit in units:
            by_size.setdefault(unit[1], []).append(unit)
        order = []
        for group in by_size.values():
            perm = torch.randperm(len(group)).tolist()
            for i in range(0, len(group), per_minibatch):
                order.append([group[j] for j in perm[i : i + per_minibatch]])
        for k in torch.randperm(len(order)).tolist():
            minibatch = order[k]
            parts = []
            for i in range(0, len(minibatch), per_micro):
                micro = minibatch[i : i + per_micro]
                steps = micro[0][1]
                starts = torch.as_tensor([u[0] for u in micro])
                index = starts[None, :] + torch.arange(steps)[:, None]  # (T, B) time-major
                resets = torch.as_tensor(np.stack([u[2] for u in micro], axis=1))
                parts.append((index.reshape(-1), (starts, resets, steps)))
            yield parts

    # ------------------------------------------------------------------ loss

    def _loss(self, out, mini: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """PPO loss (batch means) of some decisions, plus detached terms for the update stats."""
        cfg = self.config
        dist = out.distribution(mini["masks"])
        actions = targets["actions"]
        logp = dist.log_prob(actions) + selection_log_prob(out, mini)
        logratio = logp - targets["old_logp"]
        ratio = logratio.exp()
        clipped_ratio = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
        advantages = targets["advantages"]
        policy_loss = -torch.minimum(advantages * ratio, advantages * clipped_ratio).mean()
        value_loss = (out.value - targets["returns"]).square().mean()
        entropy = dist.entropy().mean()
        trade_coef = cfg.trade_entropy_coef
        if trade_coef is None or not isinstance(dist, TradeDistribution):
            bonus = cfg.entropy_coef * entropy
        else:
            bonus = (
                cfg.entropy_coef * dist.head_entropy().mean()
                + trade_coef * dist.token_entropy().mean()
            )
        loss = policy_loss + cfg.value_coef * value_loss - bonus
        terms = {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}
        if "upgo" in targets:
            upgo = targets["upgo"]
            upgo_loss = -torch.minimum(upgo * ratio, upgo * clipped_ratio).mean()
            loss = loss + cfg.upgo_coef * upgo_loss
            terms["upgo_loss"] = upgo_loss
        if "strategy_returns" in targets and out.strategy_value is not None:
            strategy_value_loss = (out.strategy_value - targets["strategy_returns"]).square().mean()
            loss = loss + cfg.strategy_value_coef * strategy_value_loss
            terms["strategy_value_loss"] = strategy_value_loss
        if out.strategy_logits is not None and SELECT_KEY in mini:
            selected = (mini[SELECT_KEY] > 0).float()
            if selected.sum() > 0:
                sel_entropy = selection_entropy(out, mini).sum() / selected.sum()
                loss = loss - self.selector_entropy_coef * sel_entropy
                terms["selector_entropy"] = sel_entropy
        if cfg.reference_kl_coef > 0:
            if cfg.reference_kl_heads == "all":
                reference_kl = head_kl(mini[REFERENCE_LOGP_KEY], dist, actions).mean()
            else:
                reference_kl = type_kl(mini[REFERENCE_TYPE_KEY], dist.logp["type"]).mean()
            loss = loss + cfg.reference_kl_coef * reference_kl
            terms["reference_kl"] = reference_kl
        auxiliary = self.policy.auxiliary_loss(out, mini)
        if auxiliary is not None:
            loss = loss + auxiliary
        else:
            auxiliary = torch.zeros((), device=self.device)
        terms["auxiliary_loss"] = auxiliary
        with torch.no_grad():
            terms["approx_kl"] = ((ratio - 1) - logratio).mean()
            terms["clip_fraction"] = ((ratio - 1).abs() > cfg.clip_coef).float().mean()
        return loss, {name: value.detach() for name, value in terms.items()}
