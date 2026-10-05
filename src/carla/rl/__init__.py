"""Reinforcement learning (requires torch): rollouts, GAE, PPO, opponent pool, training."""

from carla.rl.buffer import RolloutBatch
from carla.rl.gae import compute_gae

__all__ = ["RolloutBatch", "compute_gae"]
