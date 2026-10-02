"""Rollout data exchanged between workers and the learner."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    """T consecutive decisions of one learner seat (episodes may end inside the batch).

    `observations` holds the stacked `ObservationEncoder` arrays (masks included), `actions` the
    head indices (T, len(HEAD_NAMES)). Rewards are summed between the seat's decisions.
    """

    observations: dict
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    last_value: float
    policy_version: int = 0
    hidden_truth: dict[str, np.ndarray] | None = None
    """Opponents' cards per step, for the belief auxiliary loss."""
    initial_state: np.ndarray | None = None
    """Recurrent state before the first step (recurrent models)."""

    def __len__(self) -> int:
        return len(self.actions)
