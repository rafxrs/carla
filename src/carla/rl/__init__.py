"""Reinforcement learning (requires torch): rollouts, GAE, PPO, and evaluation."""

from carla.rl.buffer import RolloutBatch
from carla.rl.gae import compute_gae

__all__ = ["RolloutBatch", "compute_gae"]
