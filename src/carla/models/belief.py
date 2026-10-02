"""Recurrent belief over opponents' hidden cards (TODO: models workstream).

Input: the observation's public event stream (`events`, `event_mask`; redacted for the observer)
plus the public per-seat counts. A GRU/LSTM (`model.belief`) carries a state across the observer's
decisions and predicts, for each opponent, the distribution of its resource cards and development
cards. Auxiliary loss (training only): cross-entropy / Poisson NLL against
`CatanAECEnv.hidden_truth(observer)`, weighted by `model.belief_loss_coef`.
"""

from torch import Tensor, nn


class BeliefModule(nn.Module):
    """events (B, E, F) + mask (B, E) + state -> belief features (B, d), predictions, new state."""

    def __init__(self, event_features: int, dim: int, cell: str = "gru") -> None:
        super().__init__()
        raise NotImplementedError("models workstream: recurrent belief module")

    def forward(
        self, events: Tensor, event_mask: Tensor, state: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        raise NotImplementedError
