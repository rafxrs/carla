"""Interface every policy/value network implements, plus observation batching."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from carla.models.distributions import FactoredDistribution, TradeDistribution
from carla.models.trade_decoder import TradeDecoder


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
    trade_context: Tensor | None = None
    trade_hand: Tensor | None = None
    trade_limit: Tensor | None = None

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
    ignore `state` and return None.
    """

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor | None:
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
