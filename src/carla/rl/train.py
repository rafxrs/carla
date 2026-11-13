"""Synchronous MLP PPO with CSV journals, isolated validation and resumable collection."""

import logging
import time
from pathlib import Path

import torch

from carla.agents import make_agent
from carla.config import save_config, to_dict
from carla.config.loader import from_dict
from carla.config.schema import ExperimentConfig
from carla.envs.catan_env import CatanEnv
from carla.models.factory import build_policy
from carla.rl.checkpoint import load_checkpoint, save_checkpoint
from carla.rl.evaluate import evaluate_policy
from carla.rl.metrics import (
    EPISODE_FIELDS,
    EVALUATION_FIELDS,
    UPDATE_FIELDS,
    EpisodeTracker,
    RolloutDiagnostics,
)
from carla.rl.ppo import PPOLearner
from carla.rl.process_rollouts import SynchronousRolloutPool
from carla.rl.rollout import collect_rollouts
from carla.utils.logging import CsvJournal, write_json
from carla.utils.runs import create_run_dir, record_session
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
    if config.runtime.num_workers <= 0 or config.runtime.envs_per_worker <= 0:
        raise ValueError("synchronous PPO requires positive workers and environments per worker")
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
    if config.runtime.torch_threads_per_worker <= 0:
        raise ValueError("torch_threads_per_worker must be positive")
    if not 0 <= config.ppo.gae_lambda <= 1 or not 0 <= config.ppo.gamma <= 1:
        raise ValueError("gamma and gae_lambda must be between zero and one")
    if config.eval.every_env_steps < 0 or config.eval.boards <= 0:
        raise ValueError("evaluation interval must be nonnegative and boards positive")
    if (
        not config.eval.opponents
        or any(name not in ("random", "heuristic") for name in config.eval.opponents)
        or len(set(config.eval.opponents)) != len(config.eval.opponents)
    ):
        raise ValueError("evaluation opponents must be distinct random/heuristic bots")


def train(
    config: ExperimentConfig, resume: Path | None = None, *, run_dir: Path | None = None
) -> Path:
    _validate(config)
    seed_everything(config.seed, config.runtime.deterministic_torch)
    torch.set_num_threads(config.runtime.torch_threads_per_worker)
    if resume is not None:
        resume = Path(resume).resolve()
        # Retained checkpoints live one directory below the run root.
        run_dir = resume.parent.parent if resume.parent.name == "checkpoints" else resume.parent
    elif run_dir is None:
        run_dir = create_run_dir(config)
    run_dir = Path(run_dir).resolve()
    device = resolve_device(config.runtime.device)
    policy = build_policy(config.model)
    learner = PPOLearner(policy, config.ppo, str(device))
    opponent_kind = config.opponents.pool[0].kind
    envs = (
        [
            CatanEnv(config.engine, config.env, [make_agent(opponent_kind) for _ in range(3)])
            for _ in range(config.runtime.envs_per_worker)
        ]
        if config.runtime.num_workers == 1
        else []
    )
    observations = None
    env_steps = 0
    state = {}
    saved_workers = None
    if resume is not None:
        meta = load_checkpoint(resume, policy, learner.optimizer, restore_rng=True)
        saved = to_dict(from_dict(ExperimentConfig, meta["config"]))
        current = to_dict(config)
        if saved["seed"] != current["seed"]:
            raise ValueError("resume config mismatch: seed")
        for section in ("engine", "env", "model", "opponents", "eval"):
            if saved[section] != current[section]:
                raise ValueError(f"resume config mismatch: {section}")
        for key in current["ppo"]:
            if key != "total_env_steps" and saved["ppo"][key] != current["ppo"][key]:
                raise ValueError(f"resume config mismatch: ppo.{key}")
        if saved["runtime"]["num_workers"] != current["runtime"]["num_workers"]:
            raise ValueError("resume config mismatch: runtime.num_workers")
        state = meta["training_state"]
        if state is None:
            raise ValueError("checkpoint has no environment state; cannot resume this training run")
        if config.runtime.num_workers > 1:
            saved_workers = state.get("worker_states")
            if saved_workers is None or len(saved_workers) != config.runtime.num_workers:
                raise ValueError("resume worker count does not match runtime.num_workers")
        elif "envs" in state:
            envs, observations = state["envs"], state["observations"]
        elif config.runtime.envs_per_worker == 1:
            envs, observations = [state["env"]], [state["obs"]]
        else:
            raise ValueError(
                "single-environment checkpoint cannot resume with multiple environments"
            )
        if config.runtime.num_workers == 1 and (
            len(envs) != config.runtime.envs_per_worker or len(observations) != len(envs)
        ):
            raise ValueError("resume environment count does not match runtime.envs_per_worker")
        env_steps = int(meta["counters"]["env_steps"])
        learner.version = int(meta["counters"]["updates"])
        if config.ppo.total_env_steps < env_steps:
            raise ValueError("total_env_steps cannot be smaller than the restored decision count")
    elif config.runtime.num_workers == 1:
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

    trackers = state.get("trackers", [EpisodeTracker(partial=resume is not None) for _ in envs])
    committed = state.get("log_rows", {})
    journals = {
        name: CsvJournal(run_dir / f"{name}.csv", fields, committed.get(name, 0))
        for name, fields in (
            ("updates", UPDATE_FIELDS),
            ("episodes", EPISODE_FIELDS),
            ("evaluation_games", EVALUATION_FIELDS),
        )
    }
    save_config(config, run_dir / "config.yaml")
    metadata = record_session(run_dir, str(device), resume=resume)
    training_seconds = state.get("training_seconds", 0.0)
    evaluation_seconds = state.get("evaluation_seconds", 0.0)
    previous_wall = state.get("wall_seconds", 0.0)
    started = time.perf_counter()
    optimizer_steps_total = state.get("optimizer_steps_total", 0)
    last_evaluation = state.get("last_evaluation", -1)
    interval = config.eval.every_env_steps
    next_evaluation = ((env_steps // interval) + 1) * interval if interval else 0
    pool = None
    file_log = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    previous_log_level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(file_log)

    def wall_seconds() -> float:
        return previous_wall + time.perf_counter() - started

    def checkpoint(path: Path) -> None:
        save_checkpoint(
            path,
            policy,
            optimizer=learner.optimizer,
            config=config,
            training_state={
                "envs": envs,
                "observations": observations,
                "trackers": trackers,
                "worker_states": pool.snapshot() if pool is not None else None,
                "log_rows": {k: v.rows for k, v in journals.items()},
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "wall_seconds": wall_seconds(),
                "optimizer_steps_total": optimizer_steps_total,
                "last_evaluation": last_evaluation,
            },
            env_steps=env_steps,
            updates=learner.version,
        )

    def evaluate(target: int) -> None:
        nonlocal evaluation_seconds, last_evaluation
        checkpoint_name = (
            f"checkpoints/step_{env_steps:012d}.pt" if config.eval.retain_checkpoints else ""
        )
        start = time.perf_counter()
        records = []
        for opponent in config.eval.opponents:
            result = evaluate_policy(
                policy,
                config,
                opponent=opponent,
                boards=config.eval.boards,
                seed=config.eval.seed,
                deterministic=config.eval.deterministic,
            )
            records.extend(result.records)
            log.info(
                "validation vs %s at %d decisions: %d/%d wins (%.3f)",
                opponent,
                env_steps,
                result.wins,
                result.games,
                result.win_rate,
            )
        evaluation_seconds += time.perf_counter() - start
        for record in records:
            journals["evaluation_games"].write(
                {
                    "update": learner.version,
                    "env_steps": env_steps,
                    "evaluation_target": target,
                    "training_seconds": training_seconds,
                    "wall_seconds": wall_seconds(),
                    "checkpoint": checkpoint_name,
                    "split": "validation",
                    **record,
                }
            )
        last_evaluation = env_steps
        if config.eval.retain_checkpoints:
            checkpoint(run_dir / checkpoint_name)

    try:
        log.info("run directory %s (learner device %s)", run_dir, device)
        if resume is not None:
            log.info("resumed update %d at %d learner decisions", learner.version, env_steps)
        if config.runtime.num_workers > 1:
            pool = SynchronousRolloutPool(config, saved_workers)
        # An untrained evaluation makes the starting performance visible on every curve.
        if interval and last_evaluation < 0:
            evaluate(env_steps)
            checkpoint(run_dir / "latest.pt")
        while env_steps < config.ppo.total_env_steps:
            remaining = config.ppo.total_env_steps - env_steps
            active = min(config.runtime.num_workers * config.runtime.envs_per_worker, remaining)
            steps = min(config.ppo.rollout_steps, remaining // active)
            records = []
            diagnostics = RolloutDiagnostics()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            if pool is not None:
                batches, records, diagnostics = pool.collect(
                    policy, learner.version, steps, active, env_steps
                )
            else:
                batches, next_observations = collect_rollouts(
                    envs[:active],
                    policy,
                    steps,
                    observations=observations[:active],
                    policy_version=learner.version,
                    trackers=trackers[:active],
                    episode_records=records,
                    diagnostics=diagnostics,
                    env_steps=env_steps,
                )
                observations[:active] = next_observations
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            collection_seconds = time.perf_counter() - start
            start = time.perf_counter()
            stats = learner.update(batches)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - start
            training_seconds += collection_seconds + update_seconds
            samples = sum(map(len, batches))
            env_steps += samples
            optimizer_steps_total += int(stats["optimizer_steps"])
            for record in records:
                journals["episodes"].write({"update": learner.version, **record})
            journals["updates"].write(
                {
                    "update": learner.version,
                    "env_steps": env_steps,
                    "samples": samples,
                    "optimizer_steps_total": optimizer_steps_total,
                    "episodes_completed": len(records),
                    "terminal_outcomes": sum(r["outcome"] != "truncated" for r in records),
                    "truncations": sum(r["outcome"] == "truncated" for r in records),
                    "nonzero_rewards": sum(int((b.rewards != 0).sum()) for b in batches),
                    "collection_seconds": collection_seconds,
                    "update_seconds": update_seconds,
                    "training_seconds": training_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "wall_seconds": wall_seconds(),
                    "decisions_per_second": samples / (collection_seconds + update_seconds),
                    **stats,
                    **diagnostics.result(),
                }
            )
            if (
                learner.version % config.runtime.log_every == 0
                or env_steps == config.ppo.total_env_steps
            ):
                log.info(
                    "update %d | decisions %d/%d | episodes %d | policy %.4f | value %.4f | entropy %.3f | KL %.4f",
                    learner.version,
                    env_steps,
                    config.ppo.total_env_steps,
                    len(records),
                    stats["policy_loss"],
                    stats["value_loss"],
                    stats["entropy"],
                    stats["approx_kl"],
                )
            if interval and (
                env_steps >= next_evaluation or env_steps == config.ppo.total_env_steps
            ):
                evaluate(next_evaluation if env_steps >= next_evaluation else env_steps)
                next_evaluation = ((env_steps // interval) + 1) * interval
            if (
                learner.version % config.runtime.checkpoint_every == 0
                or env_steps == config.ppo.total_env_steps
            ):
                checkpoint(run_dir / "latest.pt")
        if interval and last_evaluation != env_steps:
            evaluate(env_steps)
            checkpoint(run_dir / "latest.pt")
        metadata["status"] = "complete"
        metadata["env_steps"] = env_steps
        write_json(run_dir / "metadata.json", metadata)
    finally:
        if pool is not None:
            pool.close()
        for env in envs:
            env.close()
        log.removeHandler(file_log)
        log.setLevel(previous_log_level)
        file_log.close()
    return run_dir
