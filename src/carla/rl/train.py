"""Single-environment MLP PPO baseline with update-boundary resume."""

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
from carla.rl.rollout import collect_rollout
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
    if config.runtime.num_workers != 1 or config.runtime.envs_per_worker != 1:
        raise ValueError("synchronous PPO requires one worker and one environment")
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
    env = CatanEnv(config.engine, config.env, [make_agent(opponent_kind) for _ in range(3)])
    obs = None
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
        env, obs = state["env"], state["obs"]
        env_steps = int(meta["counters"]["env_steps"])
        learner.version = int(meta["counters"]["updates"])
        save_config(config, run_dir / "config.yaml")
        log.info("resumed update %d at %d learner decisions", learner.version, env_steps)

    def checkpoint() -> None:
        save_checkpoint(
            run_dir / "latest.pt",
            policy,
            optimizer=learner.optimizer,
            config=config,
            training_state={"env": env, "obs": obs},
            env_steps=env_steps,
            updates=learner.version,
        )

    while env_steps < config.ppo.total_env_steps:
        steps = min(config.ppo.rollout_steps, config.ppo.total_env_steps - env_steps)
        batch, obs = collect_rollout(
            env,
            policy,
            steps,
            obs=obs,
            episode_seed=derive_seed(config.seed, "worker", 0),
            policy_version=learner.version,
        )
        stats = learner.update([batch])
        env_steps += len(batch)
        if (
            learner.version % config.runtime.log_every == 0
            or env_steps == config.ppo.total_env_steps
        ):
            log.info(
                "update %d | decisions %d/%d | episodes %d | policy %.4f | value %.4f | entropy %.3f | KL %.4f",
                learner.version,
                env_steps,
                config.ppo.total_env_steps,
                int(batch.dones.sum()),
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
    env.close()
    return run_dir
