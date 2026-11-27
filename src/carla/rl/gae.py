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


def compute_upgo(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
) -> np.ndarray:
    """Upgoing returns (AlphaStar's UPGO) for one sequence shaped (T,).

    G_t = r_t + gamma * (G_{t+1} if Q_{t+1} >= V_{t+1} else V_{t+1}), with the one-step action
    value Q_{t+1} = r_{t+1} + gamma * V_{t+2}: the return follows the trajectory while the
    behaviour's actions were at least as good as expected and bootstraps otherwise. G_t = r_t
    at an episode's last decision and bootstraps from `last_value` at the end of the batch.
    """
    steps = len(rewards)
    out = np.zeros(steps, dtype=np.float32)
    for t in range(steps - 1, -1, -1):
        if dones[t]:
            out[t] = rewards[t]
        elif t == steps - 1:
            out[t] = rewards[t] + gamma * last_value
        else:
            after = 0.0 if dones[t + 1] else (values[t + 2] if t + 2 < steps else last_value)
            q_next = rewards[t + 1] + gamma * after
            follow = q_next >= values[t + 1]
            out[t] = rewards[t] + gamma * (out[t + 1] if follow else values[t + 1])
    return out
