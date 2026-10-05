"""Training entry point (`carla train <config.yaml> [key=value ...]`).

Implemented: seeding, run directory, env + policy construction, one synchronous rollout, GAE,
checkpoint. TODO(rl): replace the dry run by the semi-asynchronous PPO loop (rollout workers ->
`PPOLearner.update` -> weight broadcast), with logging every `runtime.log_every` updates.
"""

import logging
import time
from pathlib import Path

import torch

from carla.config.schema import ExperimentConfig
from carla.envs.catan_env import CatanEnv
from carla.models.factory import build_policy
from carla.rl.checkpoint import save_checkpoint
from carla.rl.gae import compute_gae
from carla.rl.rollout import collect_rollout
from carla.utils.runs import create_run_dir
from carla.utils.seeding import derive_seed, seed_everything

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(config: ExperimentConfig) -> Path:
    seed_everything(config.seed, config.runtime.deterministic_torch)
    run_dir = create_run_dir(config)
    device = resolve_device(config.runtime.device)
    log.info("run directory %s (learner device %s)", run_dir, device)

    env = CatanEnv(config.engine, config.env)
    policy = build_policy(config.model)
    start = time.perf_counter()
    batch, _ = collect_rollout(
        env, policy, config.ppo.rollout_steps, episode_seed=derive_seed(config.seed, "worker", 0)
    )
    elapsed = time.perf_counter() - start
    advantages, _ = compute_gae(
        batch.rewards,
        batch.values,
        batch.dones,
        batch.last_value,
        config.ppo.gamma,
        config.ppo.gae_lambda,
    )
    save_checkpoint(run_dir / "checkpoint_0.pt", policy, config=config, env_steps=len(batch))
    log.info(
        "dry run: %d decisions in %.2fs (%.0f/s), %d episode ends, mean |advantage| %.3f",
        len(batch),
        elapsed,
        len(batch) / elapsed,
        int(batch.dones.sum()),
        float(abs(advantages).mean()),
    )
    log.warning("PPO updates are not implemented yet (rl workstream): stopping after the dry run")
    return run_dir
