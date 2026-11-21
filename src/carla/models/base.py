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
