"""Opponent pools.

`config.opponents.pool` lists weighted specs: "random" / "heuristic" bots, "self" (the current
policy), "snapshot" (a frozen past version of it) and "checkpoint" (a fixed policy file). A pool
of bots only keeps fixed tables (`bot_tables`); any policy kind makes it a league (`League`),
where every opponent seat of every episode samples a spec in proportion to its weight.
"""

import copy
import random
from collections import OrderedDict
from pathlib import Path

import torch

from carla.agents import make_agent
from carla.agents.base import Agent
from carla.config.loader import from_dict
from carla.config.schema import EnvConfig, ExperimentConfig, OpponentPoolConfig, OpponentSpec
from carla.engine.config import EngineConfig
from carla.envs.catan_env import ExternalAgent
from carla.models.base import PolicyNetwork
from carla.models.factory import build_policy

BOT_KINDS = ("random", "heuristic")
POLICY_KINDS = ("self", "snapshot", "checkpoint")


def bot_tables(config: OpponentPoolConfig, num_workers: int, envs_per_worker: int) -> list:
    """Fixed bot kinds for every opponent seat: `tables[worker][env]` lists three kinds.

    Synchronous PPO keeps each env's opponents for the whole run. Seats are allocated to the
    pool's random/heuristic specs in proportion to their weights, striped across envs so mixed
    pools give mixed tables (e.g. weights 2:1 give two random bots and one heuristic bot per env).
    """
    specs = config.pool
    if not specs or any(spec.kind not in BOT_KINDS or spec.weight <= 0 for spec in specs):
        raise ValueError("synchronous PPO opponents must be random/heuristic with positive weight")
    tables_count = num_workers * envs_per_worker
    total = 3 * tables_count
    weight_sum = sum(spec.weight for spec in specs)
    bounds, cumulative = [], 0.0
    for spec in specs:
        cumulative += spec.weight / weight_sum
        bounds.append(cumulative)
    kinds = [[["random"] * 3 for _ in range(envs_per_worker)] for _ in range(num_workers)]
    for seat in range(3):
        for table in range(tables_count):
            fraction = (seat * tables_count + table + 0.5) / total
            index = next((i for i, bound in enumerate(bounds) if fraction < bound), len(specs) - 1)
            kinds[table // envs_per_worker][table % envs_per_worker][seat] = specs[index].kind
    return kinds


def is_league(config: OpponentPoolConfig) -> bool:
    """Whether any opponent is a policy (then opponents are sampled per episode)."""
    return any(spec.kind in POLICY_KINDS for spec in config.pool)


def validate_pool(config: OpponentPoolConfig) -> None:
    if config.snapshot_every <= 0 or config.max_snapshots <= 0:
        raise ValueError("opponents.snapshot_every and max_snapshots must be positive")
    if not is_league(config):
        bot_tables(config, 1, 1)
        return
    if any(spec.weight <= 0 for spec in config.pool):
        raise ValueError("opponent weights must be positive")
    for spec in config.pool:
        if spec.kind not in BOT_KINDS + POLICY_KINDS:
            raise ValueError(f"unknown opponent kind {spec.kind!r}")
        if spec.kind == "checkpoint" and not spec.path:
            raise ValueError("checkpoint opponents need a path")


def load_opponent(path: str, engine: EngineConfig, env: EnvConfig) -> PolicyNetwork:
    """A frozen policy from a training checkpoint, which must share the engine rules and the
    observation history length (its own architecture may differ)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("config") is None:
        raise ValueError(f"opponent checkpoint has no experiment config: {path}")
    config = from_dict(ExperimentConfig, data["config"])
    if config.engine != engine:
        raise ValueError(f"opponent checkpoint uses different engine rules: {path}")
    if config.env.max_events != env.max_events:
        raise ValueError(f"opponent checkpoint uses a different env.max_events: {path}")
    network = build_policy(config.model)
    network.load_state_dict(data["policy"])
    network.requires_grad_(False)
    return network.eval()


class League:
    """One rollout process's opponents: samples three per episode and owns their networks.

    Snapshots live in the process: every `snapshot_every`-th policy version it receives is copied
    (at most `max_snapshots` kept, oldest dropped). They are not saved in checkpoints, so after a
    resume "snapshot" seats play the current policy until the next snapshot. All "snapshot" seats
    of episodes started in one rollout share one snapshot, so their decisions batch together.
    """

    def __init__(
        self,
        config: OpponentPoolConfig,
        engine: EngineConfig,
        env: EnvConfig,
        seed: int,
        device: torch.device | str = "cpu",
    ) -> None:
        validate_pool(config)
        self.config = config
        self.rng = random.Random(seed)
        self.device = torch.device(device)
        self.checkpoints: dict[str, PolicyNetwork] = {}
        self.names: dict[str, str] = {}
        for i, spec in enumerate(config.pool):
            if spec.kind == "checkpoint":
                assert spec.path is not None
                key = f"checkpoint:{i}"
                self.checkpoints[key] = load_opponent(spec.path, engine, env).to(self.device)
                self.names[key] = spec.name or Path(spec.path).stem
        self.snapshots: OrderedDict[int, PolicyNetwork] = OrderedDict()
        self.active: int | None = None

    def add_snapshot(self, policy: PolicyNetwork, version: int) -> None:
        """Keep a frozen copy of `policy` when `version` is a snapshot version."""
        if version % self.config.snapshot_every or version in self.snapshots:
            return
        snapshot = copy.deepcopy(policy).to(self.device)
        snapshot.requires_grad_(False)
        self.snapshots[version] = snapshot.eval()
        while len(self.snapshots) > self.config.max_snapshots:
            self.snapshots.popitem(last=False)

    def begin_rollout(self) -> None:
        """Choose the snapshot that episodes starting in this rollout play against."""
        self.active = self.rng.choice(list(self.snapshots)) if self.snapshots else None

    def sample(self) -> list[Agent]:
        specs = list(enumerate(self.config.pool))
        weights = [spec.weight for _, spec in specs]
        return [self._agent(*self.rng.choices(specs, weights)[0]) for _ in range(3)]

    def _agent(self, index: int, spec: OpponentSpec) -> Agent:
        if spec.kind in BOT_KINDS:
            return make_agent(spec.kind)
        if spec.kind == "checkpoint":
            key = f"checkpoint:{index}"
            return ExternalAgent(self.names[key], key)
        if spec.kind == "snapshot" and self.active is not None:
            return ExternalAgent(f"snapshot:{self.active}", f"snapshot:{self.active}")
        return ExternalAgent("self", "self")  # Also "snapshot" before the first snapshot.

    def network(self, key: str, policy: PolicyNetwork) -> PolicyNetwork:
        """The network playing `key`; the current policy for "self" and dropped snapshots."""
        if key.startswith("snapshot:"):
            return self.snapshots.get(int(key.removeprefix("snapshot:")), policy)
        return self.checkpoints.get(key, policy)

    def state(self) -> dict:
        return {"rng": self.rng.getstate(), "active": self.active}

    def restore(self, state: dict) -> None:
        self.rng.setstate(state["rng"])
        self.active = None  # Snapshots are not restored.
