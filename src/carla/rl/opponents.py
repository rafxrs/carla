"""Opponent pool for self-play (TODO: rl workstream).

Holds the specs of `config.opponents.pool`: "self" (current policy), "snapshot" (frozen past
versions, one added every `snapshot_every` updates, at most `max_snapshots` kept), "random" /
"heuristic" bots and fixed "checkpoint" files. Each opponent seat of each episode samples a spec
in proportion to its weight, using the worker's seeded RNG.
"""

from carla.agents.base import Agent
from carla.config.schema import OpponentPoolConfig

BOT_KINDS = (
    "random",
    "heuristic05",
    "heuristic10",
    "heuristic15",
    "heuristic25",
    "heuristic50",
    "heuristic75",
    "heuristic",
)

def bot_tables(config: OpponentPoolConfig, num_workers: int, envs_per_worker: int) -> list:
    """Fixed bot kinds for every opponent seat: `tables[worker][env]` lists three kinds.

    Synchronous PPO keeps each env's opponents for the whole run. Seats are allocated to the
    pool's random/heuristic specs in proportion to their weights, striped across envs so mixed
    pools give mixed tables (e.g. weights 2:1 give two random bots and one heuristic bot per env).
    """
    specs = config.pool
    if not specs or any(spec.kind not in BOT_KINDS or spec.weight <= 0 for spec in specs):
        raise ValueError("synchronous PPO opponents must be registered bot kinds with positive weight"
)
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


class OpponentPool:
    def __init__(self, config: OpponentPoolConfig, seed: int) -> None:
        self.config = config
        self.seed = seed

    def add_snapshot(self, state_dict: dict, version: int) -> None:
        raise NotImplementedError("rl workstream: opponent snapshots")

    def sample(self, episode: int) -> list[Agent]:
        """Three agents for the opponent seats of `episode`. TODO(rl)."""
        raise NotImplementedError("rl workstream: opponent sampling")
