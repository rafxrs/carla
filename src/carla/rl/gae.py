"""Generalised advantage estimation."""

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray | float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for arrays shaped (T,) or (T, N).

    `dones[t]` is 1 if the episode ended after step t (no bootstrapping across it);
    `last_value` is V(s_T) for the state following the last step.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_value = np.asarray(last_value, dtype=np.float32)
    running = np.zeros_like(next_value)
    for t in range(len(rewards) - 1, -1, -1):
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values
