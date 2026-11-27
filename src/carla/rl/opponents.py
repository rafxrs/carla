"""Opponent pool for self-play (TODO: rl workstream).

Holds the specs of `config.opponents.pool`: "self" (current policy), "snapshot" (frozen past
versions, one added every `snapshot_every` updates, at most `max_snapshots` kept), "random" /
"heuristic" bots and fixed "checkpoint" files. Each opponent seat of each episode samples a spec
in proportion to its weight, using the worker's seeded RNG.
"""

from carla.agents.base import Agent
from carla.config.schema import OpponentPoolConfig, OpponentSpec

BOT_KINDS = ("random", "heuristic", "agent", "checkpoint")
PROCESS_BACKED = ("jsettlers-smart-adapted", "jsettlers-fast-adapted", "external")
"""Agents running another process; their state cannot be pickled with a rollout worker."""


def agent_name(spec: OpponentSpec) -> str:
    """The registry name a fixed opponent spec builds (`carla.agents.make_agent`)."""
    from carla.agents.registry import is_agent_name

    if spec.kind in ("random", "heuristic"):
        return spec.kind
    if spec.kind == "checkpoint":
        if not spec.path:
            raise ValueError("checkpoint opponents need a path")
        return "ppo:" + spec.path
    if spec.kind == "agent":
        if not spec.name or not is_agent_name(spec.name):
            raise ValueError(f"unknown opponent agent {spec.name!r}")
        if spec.name in PROCESS_BACKED:
            raise ValueError(
                f"{spec.name} runs in another process and cannot be a training opponent; "
                "use its native port or evaluate against it instead"
            )
        return spec.name
    raise ValueError(f"opponent kind {spec.kind!r} is not supported by synchronous PPO")


def bot_tables(config: OpponentPoolConfig, num_workers: int, envs_per_worker: int) -> list:
    """Fixed opponents for every seat: `tables[worker][env]` lists three agent registry names.

    Synchronous PPO keeps each env's opponents for the whole run. Seats are allocated to the
    pool's fixed specs (random, heuristic, a registry agent such as a benchmark port, or a
    checkpoint) in proportion to their weights, striped across envs so mixed pools give mixed
    tables (e.g. weights 2:1 give two random bots and one heuristic bot per env).
    """
    specs = config.pool
    if not specs or any(spec.kind not in BOT_KINDS or spec.weight <= 0 for spec in specs):
        raise ValueError(
            "synchronous PPO opponents must be fixed (random, heuristic, agent, checkpoint) "
            "with positive weight"
        )
    names = [agent_name(spec) for spec in specs]
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
            kinds[table // envs_per_worker][table % envs_per_worker][seat] = names[index]
    return kinds


class OpponentPool:
    def __init__(self, config: OpponentPoolConfig, seed: int) -> None:
        self.config = config
        self.seed = seed

    def add_snapshot(self, state_dict: dict, version: int) -> None:
        raise NotImplementedError("rl workstream: opponent snapshots")

    def sample(self, episode: int) -> list[Agent]:
        """Three agents for the opponent seats of `episode`. TODO(rl)."""
        raise NotImplementedError("rl workstream: opponent sampling")
