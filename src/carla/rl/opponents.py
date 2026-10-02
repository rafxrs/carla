"""Opponent pool for self-play (TODO: rl workstream).

Holds the specs of `config.opponents.pool`: "self" (current policy), "snapshot" (frozen past
versions, one added every `snapshot_every` updates, at most `max_snapshots` kept), "random" /
"heuristic" bots and fixed "checkpoint" files. Each opponent seat of each episode samples a spec
in proportion to its weight, using the worker's seeded RNG.
"""

from carla.agents.base import Agent
from carla.config.schema import OpponentPoolConfig


class OpponentPool:
    def __init__(self, config: OpponentPoolConfig, seed: int) -> None:
        self.config = config
        self.seed = seed

    def add_snapshot(self, state_dict: dict, version: int) -> None:
        raise NotImplementedError("rl workstream: opponent snapshots")

    def sample(self, episode: int) -> list[Agent]:
        """Three agents for the opponent seats of `episode`. TODO(rl)."""
        raise NotImplementedError("rl workstream: opponent sampling")
