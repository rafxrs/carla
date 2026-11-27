"""One batched decision of a policy, including strategy selection.

Shared by rollouts, opponent seats, evaluation and `PolicyAgent`, so every place that runs a
strategy-conditioned policy selects and conditions it the same way:

1. each row gets its seat's z (`strategy_observation`), or zeros without a controller;
2. the policy runs once; rows at a selection point sample an archetype (or none) from the
   selector logits, which do not depend on z;
3. those rows switch to the new z and run again from the same input state, so the action
   heads act under the strategy just chosen;
4. the action is sampled. Its log-probability includes the selection term on selection rows.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from carla.envs.strategy import (
    CHOICE_KEY,
    SELECT_KEY,
    StrategyController,
    StrategyStats,
    strategy_observation,
)
from carla.models.base import PolicyNetwork, PolicyOutput, batch_observations, replace_rows


@dataclass
class Decisions:
    heads: np.ndarray
    """(B, action width) sampled actions."""
    log_probs: np.ndarray
    """(B,) joint log-probability (selection term included on selection rows)."""
    output: PolicyOutput
    batch: dict
    """Final batched observations (after selection), on the policy's device."""
    distribution: object
    """The distribution the actions were sampled from."""
    observations: list[dict]
    """Final per-row observations to store (numpy)."""


def selection_log_prob(out: PolicyOutput, obs: dict[str, Tensor]) -> Tensor:
    """(B,) log-probability of the stored selections (0 on rows without one)."""
    if out.strategy_logits is None or SELECT_KEY not in obs:
        return out.value.new_zeros(out.value.shape[0])
    logp = torch.log_softmax(out.strategy_logits.float(), dim=1)
    choice = obs[CHOICE_KEY].long().clamp(min=0)
    selected = obs[SELECT_KEY].float() > 0
    return torch.where(selected, logp.gather(1, choice[:, None]).squeeze(1), 0.0)


def selection_entropy(out: PolicyOutput, obs: dict[str, Tensor]) -> Tensor:
    """(B,) selector entropy on selection rows (0 elsewhere)."""
    if out.strategy_logits is None or SELECT_KEY not in obs:
        return out.value.new_zeros(out.value.shape[0])
    logp = torch.log_softmax(out.strategy_logits.float(), dim=1)
    entropy = -(logp.exp() * logp).sum(1)
    return torch.where(obs[SELECT_KEY].float() > 0, entropy, 0.0)


@torch.no_grad()
def decide(
    network: PolicyNetwork,
    observations: Sequence[dict],
    controllers: Sequence[StrategyController | None],
    games: Sequence,
    seats: Sequence[int],
    stats: Sequence[StrategyStats | None],
    *,
    deterministic: bool = False,
    state: Tensor | None = None,
    with_strategy: bool | None = None,
) -> Decisions:
    """Decisions of `network` for B rows (see the module docstring).

    `state` overrides the observations' stored recurrent state (evaluation keeps it outside).
    """
    device = next(network.parameters()).device
    selecting = [
        network.num_strategies > 0
        and controller is not None
        and controller.needs_selection(game, seat)
        for controller, game, seat in zip(controllers, games, seats, strict=True)
    ]
    rows = list(observations)
    if network.uses_strategy if with_strategy is None else with_strategy:
        rows = [
            strategy_observation(obs, controller, False)
            for obs, controller in zip(rows, controllers, strict=True)
        ]
    batch = batch_observations(rows, device=device)
    out = network(batch, state)
    if any(selecting):
        assert out.strategy_logits is not None
        index = [i for i, flag in enumerate(selecting) if flag]
        logits = out.strategy_logits[index].float()
        if deterministic:
            choices = logits.argmax(1)
        else:
            choices = torch.multinomial(torch.softmax(logits, 1), 1).squeeze(1)
        for i, choice in zip(index, choices.tolist(), strict=True):
            controller = controllers[i]
            assert controller is not None and stats[i] is not None
            controller.select(int(choice), games[i], stats[i])
            rows[i] = strategy_observation(observations[i], controller, True)
        sub_batch = batch_observations([rows[i] for i in index], device=device)
        sub_state = None if state is None else state[index]
        sub = network(sub_batch, sub_state)
        index_t = torch.as_tensor(index, device=device)
        out = replace_rows(out, index_t, sub)
        batch = batch_observations(rows, device=device)
    dist = out.distribution(batch["masks"])
    heads = dist.sample(deterministic=deterministic)
    log_probs = dist.log_prob(heads) + selection_log_prob(out, batch)
    return Decisions(
        heads=heads.cpu().numpy(),
        log_probs=log_probs.float().cpu().numpy(),
        output=out,
        batch=batch,
        distribution=dist,
        observations=rows,
    )
