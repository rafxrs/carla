"""Recurrent belief over opponents' hidden cards.

Input: the observation's public event stream (`events`, `event_mask`; redacted for the observer),
which holds the events since the observer's previous decision. A GRU folds those events into a
state carried across the observer's decisions, so the state summarizes the whole game. Heads
predict, for each opponent in relative seat order, the Poisson rate of each resource and
development card count. Auxiliary loss (training only): Poisson NLL against
`CatanAECEnv.hidden_truth(observer)`, weighted by `model.belief_loss_coef`.

Training stores the state before each decision in the observation (`STATE_KEY`), so PPO
recomputes one decision's events from that stored state (truncated BPTT of one decision).
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from carla.engine.constants import NUM_DEV_CARD_TYPES, NUM_PLAYERS, NUM_RESOURCES

TARGET_SHAPES = {
    "resources": (NUM_PLAYERS - 1, NUM_RESOURCES),
    "dev_cards": (NUM_PLAYERS - 1, NUM_DEV_CARD_TYPES),
}


class BeliefModule(nn.Module):
    """events (B, E, F) + mask (B, E) + state (B, d) -> features (B, d), predictions, new state."""

    def __init__(self, event_features: int, dim: int, cell: str = "gru") -> None:
        super().__init__()
        if cell != "gru":
            raise ValueError(f"unsupported belief cell {cell!r}; only 'gru' is implemented")
        if dim <= 0:
            raise ValueError("belief_dim must be positive")
        self.dim = dim
        self.event_input = nn.Sequential(nn.Linear(event_features, dim), nn.ReLU())
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.features = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.predict = nn.ModuleDict(
            {key: nn.Linear(dim, rows * cols) for key, (rows, cols) in TARGET_SHAPES.items()}
        )

    def forward(
        self, events: Tensor, event_mask: Tensor, state: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        batch = events.shape[0]
        if state is None:
            state = events.new_zeros(batch, self.dim)
        mask = event_mask.bool()
        lengths = mask.sum(dim=1)
        width = int(lengths.max()) if batch else 0
        if width > 0:
            # Events are left-aligned; padding after the last event never reaches the gather.
            outputs, _ = self.gru(self.event_input(events[:, :width].float()), state[None])
            last = outputs[torch.arange(batch, device=events.device), (lengths - 1).clamp_min(0)]
            new_state = torch.where((lengths > 0)[:, None], last, state)
        else:
            new_state = state
        features = self.features(new_state)
        predictions = {
            key: self.predict[key](features).reshape(batch, *shape)
            for key, shape in TARGET_SHAPES.items()
        }
        return features, predictions, new_state


def belief_loss(predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    """Mean Poisson NLL of the hidden card counts; predictions are log-rates."""
    return torch.stack(
        [
            F.poisson_nll_loss(predictions[key], targets[key].float(), log_input=True)
            for key in TARGET_SHAPES
        ]
    ).mean()
