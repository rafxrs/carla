"""Synchronous batched-environment MLP PPO with update-boundary resume."""

import logging
from pathlib import Path

import torch

from carla.agents import make_agent
from carla.config import save_config, to_dict
from carla.config.schema import ExperimentConfig
from carla.envs.catan_env import CatanEnv
from carla.models.factory import build_policy
from carla.rl.checkpoint import load_checkpoint, save_checkpoint
from carla.rl.ppo import PPOLearner
from carla.rl.rollout import collect_rollouts
from carla.utils.runs import create_run_dir
from carla.utils.seeding import derive_seed, seed_everything

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate(config: ExperimentConfig) -> None:
    if config.model.encoder != "mlp" or config.model.belief != "none":
        raise ValueError("this baseline supports only feed-forward MLP policies without belief")
    if config.model.trade_decoder != "none" or config.model.set_attention:
        raise ValueError(
            "this baseline does not support attention or variable-length trade decoding"
        )
    if config.runtime.num_workers != 1 or config.runtime.envs_per_worker <= 0:
        raise ValueError("synchronous PPO requires one worker and at least one environment")
    if len(config.opponents.pool) != 1 or config.opponents.pool[0].kind not in (
        "random",
        "heuristic",
    ):
        raise ValueError("synchronous PPO requires one fixed random or heuristic opponent kind")
    if config.ppo.total_env_steps <= 0 or config.ppo.rollout_steps <= 0:
        raise ValueError("PPO step budgets must be positive")
    if config.ppo.epochs <= 0 or config.ppo.minibatch_size <= 0:
        raise ValueError("PPO epochs and minibatch size must be positive")
    if config.runtime.checkpoint_every <= 0 or config.runtime.log_every <= 0:
        raise ValueError("checkpoint and log intervals must be positive")


def train(config: ExperimentConfig, resume: Path | None = None) -> Path:
    _validate(config)
    seed_everything(config.seed, config.runtime.deterministic_torch)
    run_dir = resume.parent if resume is not None else create_run_dir(config)
    device = resolve_device(config.runtime.device)
    log.info("run directory %s (learner device %s)", run_dir, device)
    policy = build_policy(config.model)
    learner = PPOLearner(policy, config.ppo, str(device))
    opponent_kind = config.opponents.pool[0].kind
    envs = [
        CatanEnv(config.engine, config.env, [make_agent(opponent_kind) for _ in range(3)])
        for _ in range(config.runtime.envs_per_worker)
    ]
    observations = None
    env_steps = 0
    if resume is not None:
        meta = load_checkpoint(resume, policy, learner.optimizer, restore_rng=True)
        saved = meta["config"]
        current = to_dict(config)
        if saved["seed"] != current["seed"]:
            raise ValueError("resume config mismatch: seed")
        for section in ("engine", "env", "model", "opponents"):
            if saved[section] != current[section]:
                raise ValueError(f"resume config mismatch: {section}")
        for key in current["ppo"]:
            if key != "total_env_steps" and saved["ppo"][key] != current["ppo"][key]:
                raise ValueError(f"resume config mismatch: ppo.{key}")
        state = meta["training_state"]
        if state is None:
            raise ValueError("checkpoint has no environment state; cannot resume this training run")
        if "envs" in state:
            envs, observations = state["envs"], state["observations"]
        elif config.runtime.envs_per_worker == 1:
            envs, observations = [state["env"]], [state["obs"]]
        else:
            raise ValueError(
                "single-environment checkpoint cannot resume with multiple environments"
            )
        if len(envs) != config.runtime.envs_per_worker or len(observations) != len(envs):
            raise ValueError("resume environment count does not match runtime.envs_per_worker")
        env_steps = int(meta["counters"]["env_steps"])
        learner.version = int(meta["counters"]["updates"])
        save_config(config, run_dir / "config.yaml")
        log.info("resumed update %d at %d learner decisions", learner.version, env_steps)
    else:
        observations = [
            env.reset(
                seed=(
                    derive_seed(config.seed, "worker", 0)
                    if i == 0
                    else derive_seed(config.seed, "worker", 0, "env", i)
                )
            )[0]
            for i, env in enumerate(envs)
        ]

    def checkpoint() -> None:
        save_checkpoint(
            run_dir / "latest.pt",
            policy,
            optimizer=learner.optimizer,
            config=config,
            training_state={"envs": envs, "observations": observations},
            env_steps=env_steps,
            updates=learner.version,
        )

    while env_steps < config.ppo.total_env_steps:
        remaining = config.ppo.total_env_steps - env_steps
        active = min(len(envs), remaining)
        steps = min(config.ppo.rollout_steps, remaining // active)
        batches, next_observations = collect_rollouts(
            envs[:active],
            policy,
            steps,
            observations=observations[:active],
            policy_version=learner.version,
        )
        observations[:active] = next_observations
        stats = learner.update(batches)
        env_steps += sum(map(len, batches))
        if (
            learner.version % config.runtime.log_every == 0
            or env_steps == config.ppo.total_env_steps
        ):
            log.info(
                "update %d | decisions %d/%d | episodes %d | policy %.4f | value %.4f | entropy %.3f | KL %.4f",
                learner.version,
                env_steps,
                config.ppo.total_env_steps,
                sum(int(batch.dones.sum()) for batch in batches),
                stats["policy_loss"],
                stats["value_loss"],
                stats["entropy"],
                stats["approx_kl"],
            )
        if (
            learner.version % config.runtime.checkpoint_every == 0
            or env_steps == config.ppo.total_env_steps
        ):
            checkpoint()
    for env in envs:
        env.close()
    return run_dir
