"""Interface every policy/value network implements, plus observation batching."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from carla.models.distributions import FactoredDistribution, TradeDistribution
from carla.models.trade_decoder import TradeDecoder

STATE_KEY = "belief_state"
"""Observation entry holding a recurrent policy's state before the decision."""
TARGET_KEY = "belief_target"
"""Observation entry holding the belief auxiliary targets (training only, never a model input)."""


@dataclass
class PolicyOutput:
    logits: dict[str, Tensor]
    """One tensor per mask key, shaped (B, *MASK_SHAPES[key])."""
    value: Tensor
    """(B,) state-value estimate for the acting seat."""
    state: Tensor | None = None
    """Next recurrent state (belief module), or None for feed-forward models."""
    belief: dict[str, Tensor] | None = None
    """Predicted opponent cards (keys of `CatanAECEnv.hidden_truth`), for the auxiliary loss."""
    trade_decoder: TradeDecoder | None = None
    """Optional offer generator; the fields below are its actor-only inputs."""
    trade_context: Tensor | None = None
    """(B, hidden_dim) context from the MLP or GNN encoder."""
    trade_hand: Tensor | None = None
    """(B, 5) exact counts of the acting player's resources."""
    trade_limit: Tensor | None = None
    """(B, 1) engine card cap per side; -1 means no engine cap."""
    strategy_value: Tensor | None = None
    """(B,) critic of the strategy pseudo-reward (strategy-conditioned models)."""
    strategy_logits: Tensor | None = None
    """(B, num_strategies + 1) selector logits; the last option plays without z."""

    def distribution(self, masks: dict[str, Tensor]) -> FactoredDistribution | TradeDistribution:
        if self.trade_decoder is not None:
            assert self.trade_context is not None and self.trade_hand is not None
            assert self.trade_limit is not None
            return TradeDistribution(
                self.logits,
                masks,
                self.trade_decoder,
                self.trade_context,
                self.trade_hand,
                self.trade_limit,
            )
        return FactoredDistribution(self.logits, masks)


class PolicyNetwork(nn.Module):
    """Maps batched observations (see `ObservationEncoder`) to a `PolicyOutput`.

    Recurrent models carry a state across a seat's consecutive decisions; feed-forward models
    ignore `state` and return None. Recurrent models read their state from `state` or, when
    it is None, from `obs[STATE_KEY]`, and start from zeros when neither is given.
    """

    state_dim: int = 0
    """Size of the recurrent state; 0 for feed-forward models."""
    uses_strategy: bool = False
    """Whether the network reads the strategy statistic (`carla.envs.strategy.STRATEGY_KEY`)."""
    num_strategies: int = 0
    """Archetypes of the selector head (0: no selector)."""

    @property
    def recurrent(self) -> bool:
        return self.state_dim > 0

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor | None:
        if not self.recurrent:
            return None
        return torch.zeros(batch_size, self.state_dim, device=device)

    def auxiliary_loss(self, output: PolicyOutput, obs: dict[str, Tensor]) -> Tensor | None:
        """Weighted auxiliary loss (belief prediction), or None when there is none."""
        return None

    def forward(self, obs: dict[str, Tensor], state: Tensor | None = None) -> PolicyOutput:
        raise NotImplementedError

    def unroll(
        self, obs: dict[str, Tensor], state: Tensor | None, resets: Tensor | None, steps: int
    ) -> PolicyOutput:
        """Run `steps` consecutive decisions of B sequences (BPTT through the whole chunk).

        `obs` rows are time-major: row t * B + b is step t of sequence b. `state` (B, state_dim)
        is the state before step 0; `resets` (steps, B) marks steps that start a new episode, where
        the state restarts from zeros. Outputs are row-aligned with `obs`; `state` is the final
        state. The default loops over `forward`; feed-forward models run in one pass.
        """
        if not self.recurrent:
            return self(obs, None)
        total = obs["globals"].shape[0]
        if total % steps:
            raise ValueError("unroll rows must be a multiple of the step count")
        width = total // steps
        if state is None:
            state = self.initial_state(width, obs["globals"].device)
        assert state is not None
        outputs = []
        for t in range(steps):
            rows = slice(t * width, (t + 1) * width)
            if resets is not None:
                state = torch.where(resets[t][:, None], torch.zeros_like(state), state)
            out = self(slice_tree(obs, rows), state)
            assert out.state is not None
            state = out.state
            outputs.append(out)
        return cat_outputs(outputs, state)


def slice_tree(tree: dict, rows) -> dict:
    """Rows `rows` (a slice or index tensor) of every tensor in a nested observation dict."""
    return {k: slice_tree(v, rows) if isinstance(v, dict) else v[rows] for k, v in tree.items()}


def _cat(values: list):
    if values[0] is None:
        return None
    if isinstance(values[0], dict):
        return {k: torch.cat([v[k] for v in values]) for k in values[0]}
    return torch.cat(values)


def cat_outputs(outputs: list[PolicyOutput], state: Tensor | None) -> PolicyOutput:
    """Concatenate per-step outputs (rows in order); `state` becomes the output state."""
    first = outputs[0]
    return PolicyOutput(
        logits=_cat([o.logits for o in outputs]),
        value=_cat([o.value for o in outputs]),
        state=state,
        belief=_cat([o.belief for o in outputs]),
        trade_decoder=first.trade_decoder,
        trade_context=_cat([o.trade_context for o in outputs]),
        trade_hand=_cat([o.trade_hand for o in outputs]),
        trade_limit=_cat([o.trade_limit for o in outputs]),
        strategy_value=_cat([o.strategy_value for o in outputs]),
        strategy_logits=_cat([o.strategy_logits for o in outputs]),
    )


def replace_rows(out: PolicyOutput, index: Tensor, sub: PolicyOutput) -> PolicyOutput:
    """`out` with rows `index` taken from `sub` (same fields, len(index) rows)."""

    def put(full, part):
        if full is None:
            return None
        if isinstance(full, dict):
            return {k: put(full[k], part[k]) for k in full}
        result = full.clone()
        result[index] = part
        return result

    return PolicyOutput(
        logits=put(out.logits, sub.logits),
        value=put(out.value, sub.value),
        state=put(out.state, sub.state),
        belief=put(out.belief, sub.belief),
        trade_decoder=out.trade_decoder,
        trade_context=put(out.trade_context, sub.trade_context),
        trade_hand=put(out.trade_hand, sub.trade_hand),
        trade_limit=put(out.trade_limit, sub.trade_limit),
        strategy_value=put(out.strategy_value, sub.strategy_value),
        strategy_logits=put(out.strategy_logits, sub.strategy_logits),
    )


def batch_observations(
    observations: Sequence[dict], device: torch.device | str = "cpu"
) -> dict[str, Tensor]:
    """Stack per-step observation dicts into tensors; `masks` stays a nested dict of bools."""
    out: dict = {}
    for key, value in observations[0].items():
        if isinstance(value, dict):
            out[key] = {
                k: torch.as_tensor(np.stack([o[key][k] for o in observations]), device=device)
                for k in value
            }
        else:
            out[key] = torch.as_tensor(np.stack([o[key] for o in observations]), device=device)
    return out


def with_state(policy: PolicyNetwork, obs: dict, state: np.ndarray | Tensor | None = None) -> dict:
    """Return `obs` carrying a recurrent policy's state (zeros at an episode start).

    Feed-forward policies get `obs` back unchanged. `state` is the policy's output state after
    the previous decision of the same episode; None keeps an existing entry or starts at zeros.
    """
    if not policy.recurrent:
        return obs
    if state is None:
        if STATE_KEY in obs:
            return obs
        return {**obs, STATE_KEY: np.zeros(policy.state_dim, dtype=np.float32)}
    if isinstance(state, Tensor):
        state = state.detach().cpu().numpy()
    return {**obs, STATE_KEY: np.asarray(state, dtype=np.float32)}
