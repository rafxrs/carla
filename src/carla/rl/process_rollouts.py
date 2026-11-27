"""Synchronous CPU rollout processes with one batched policy call per worker step."""

import multiprocessing as mp
import random
import traceback
from contextlib import suppress
from multiprocessing.connection import Connection

import numpy as np
import torch

from carla.agents import make_agent
from carla.config.schema import ExperimentConfig
from carla.envs.catan_env import CatanEnv
from carla.models.factory import build_policy
from carla.rl.metrics import EpisodeTracker, RolloutDiagnostics
from carla.rl.opponents import League, bot_tables, is_league
from carla.rl.rollout import collect_rollouts, load_reference, reset_envs
from carla.rl.strategies import StrategyAssigner
from carla.utils.seeding import derive_seed, seed_everything


def _random_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 (legacy library RNG)
        "torch": torch.get_rng_state(),
    }


def _restore_random_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002 (legacy library RNG)
    torch.set_rng_state(state["torch"])


def _worker(
    conn: Connection,
    worker_id: int,
    config: ExperimentConfig,
    saved: dict | None,
    library=None,
) -> None:
    envs = []
    try:
        torch.set_num_threads(config.runtime.torch_threads_per_worker)
        seed_everything(
            derive_seed(config.seed, "worker", worker_id), config.runtime.deterministic_torch
        )
        policy = build_policy(config.model)
        reference = load_reference(config.ppo)
        league = (
            League(
                config.opponents,
                config.engine,
                config.env,
                derive_seed(config.seed, "worker", worker_id, "league"),
            )
            if is_league(config.opponents)
            else None
        )
        strategies = (
            StrategyAssigner(
                config.strategy, library, derive_seed(config.seed, "worker", worker_id, "strategy")
            )
            if config.strategy.enabled
            else None
        )
        observations: list[dict] | None
        if saved is None and (league is not None or strategies is not None):
            if league is not None:
                envs = [
                    CatanEnv(config.engine, config.env, league.sample())
                    for _ in range(config.runtime.envs_per_worker)
                ]
            else:
                tables = bot_tables(
                    config.opponents, config.runtime.num_workers, config.runtime.envs_per_worker
                )[worker_id]
                envs = [
                    CatanEnv(config.engine, config.env, [make_agent(kind) for kind in table])
                    for table in tables
                ]
            observations = None  # Reset once the first weights arrive: "self" seats need them.
            trackers = [EpisodeTracker() for _ in envs]
        elif saved is None:
            tables = bot_tables(
                config.opponents, config.runtime.num_workers, config.runtime.envs_per_worker
            )[worker_id]
            envs = [
                CatanEnv(config.engine, config.env, [make_agent(kind) for kind in table])
                for table in tables
            ]
            observations = [
                env.reset(seed=derive_seed(config.seed, "worker", worker_id, "env", i))[0]
                for i, env in enumerate(envs)
            ]
            trackers = [EpisodeTracker() for _ in envs]
        else:
            envs, observations, trackers = (
                saved["envs"],
                saved["observations"],
                saved["trackers"],
            )
            _restore_random_state(saved["rng"])
            if league is not None and saved.get("league") is not None:
                league.restore(saved["league"])
            if strategies is not None and saved.get("strategies") is not None:
                strategies.restore(saved["strategies"])
        conn.send(("ready", None))
        while True:
            command, payload = conn.recv()
            if command == "stop":
                break
            if command == "state":
                conn.send(
                    (
                        "ok",
                        {
                            "envs": envs,
                            "observations": observations,
                            "trackers": trackers,
                            "rng": _random_state(),
                            "league": None if league is None else league.state(),
                            "strategies": None if strategies is None else strategies.state(),
                        },
                    )
                )
                continue
            if command != "collect":
                raise ValueError(f"unknown worker command: {command}")
            policy.load_state_dict(payload["weights"])
            if league is not None:
                league.add_snapshot(policy, payload["version"])
            if observations is None:
                observations = reset_envs(
                    envs,
                    policy,
                    league,
                    [
                        derive_seed(config.seed, "worker", worker_id, "env", i)
                        for i in range(len(envs))
                    ],
                    strategies,
                )
            records: list[dict] = []
            diagnostics = RolloutDiagnostics()
            active = payload["active"]
            batches, next_observations = collect_rollouts(
                envs[:active],
                policy,
                payload["steps"],
                observations=observations[:active],
                policy_version=payload["version"],
                trackers=trackers[:active],
                episode_records=records,
                diagnostics=diagnostics,
                env_steps=payload["env_steps"],
                env_id_offset=worker_id * config.runtime.envs_per_worker,
                reference=reference,
                league=league,
                strategies=strategies,
                reference_heads=config.ppo.reference_kl_heads,
            )
            observations[:active] = next_observations
            conn.send(("ok", (batches, records, diagnostics)))
    except BaseException:
        with suppress(BrokenPipeError, OSError):
            conn.send(("error", traceback.format_exc()))
    finally:
        for env in envs:
            env.close()
        conn.close()


class SynchronousRolloutPool:
    """Keep environment and RNG state in each process; synchronize every PPO update."""

    def __init__(
        self, config: ExperimentConfig, saved: list[dict] | None = None, library=None
    ) -> None:
        if saved is not None and len(saved) != config.runtime.num_workers:
            raise ValueError("resume worker count does not match runtime.num_workers")
        self.config = config
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        context = mp.get_context("spawn")
        try:
            for i in range(config.runtime.num_workers):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, i, config, None if saved is None else saved[i], library),
                    name=f"carla-rollout-{i}",
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                self._receive(connection, "ready")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection: Connection, expected: str = "ok"):
        try:
            status, value = connection.recv()
        except EOFError as exc:
            raise RuntimeError("rollout process exited unexpectedly") from exc
        if status == "error":
            raise RuntimeError(f"rollout process failed:\n{value}")
        if status != expected:
            raise RuntimeError(f"unexpected rollout response: {status}")
        return value

    def collect(self, policy, version: int, steps: int, active: int, env_steps: int):
        per_worker = self.config.runtime.envs_per_worker
        weights = {key: value.detach().cpu() for key, value in policy.state_dict().items()}
        assignments = []
        offset = env_steps
        for i, connection in enumerate(self.connections):
            count = min(per_worker, max(0, active - i * per_worker))
            if count:
                connection.send(
                    (
                        "collect",
                        {
                            "weights": weights,
                            "version": version,
                            "steps": steps,
                            "active": count,
                            "env_steps": offset,
                        },
                    )
                )
                assignments.append(connection)
                offset += count * steps
        batches, records = [], []
        diagnostics = RolloutDiagnostics()
        for connection in assignments:
            worker_batches, worker_records, worker_diagnostics = self._receive(connection)
            batches.extend(worker_batches)
            records.extend(worker_records)
            diagnostics.merge(worker_diagnostics)
        return batches, records, diagnostics

    def snapshot(self) -> list[dict]:
        for connection in self.connections:
            connection.send(("state", None))
        return [self._receive(connection) for connection in self.connections]

    def close(self) -> None:
        for connection in self.connections:
            with suppress(BrokenPipeError, EOFError, OSError):
                connection.send(("stop", None))
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in self.connections:
            connection.close()
