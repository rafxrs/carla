"""Behavioural diagnostics: what each agent builds, from the public event stream.

`behaviour_profile` plays seeded games with seat rotation and reports, per agent, its win rate,
mean victory points and the mean `carla.envs.strategy` statistics (settlements, cities, roads,
development cards bought, knights, longest road and largest army held at the end), plus the
opening build order frequencies. Used to check that teacher styles are distinct and that a
strategy-conditioned policy changes behaviour with its z.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from carla.agents.base import Agent
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.game import Game
from carla.engine.rng import derive_seed
from carla.envs.strategy import (
    NUM_PURCHASE_KINDS,
    PURCHASE_KINDS,
    STAT_NAMES,
    StrategyStats,
    track_step,
)


@dataclass
class AgentProfile:
    name: str
    games: int = 0
    wins: int = 0
    victory_points: float = 0.0
    stats: np.ndarray = field(default_factory=lambda: np.zeros(len(STAT_NAMES)))
    opening: np.ndarray = field(default_factory=lambda: np.zeros(NUM_PURCHASE_KINDS))
    """Counts of each purchase kind among the first four purchases."""
    turns: float = 0.0

    def add(self, stats: StrategyStats, won: bool, vp: int, turns: int) -> None:
        self.games += 1
        self.wins += int(won)
        self.victory_points += vp
        self.stats += stats.counts
        self.turns += turns
        for kind in stats.build_order[:4]:
            self.opening[kind] += 1

    def summary(self) -> dict:
        n = max(1, self.games)
        out = {
            "agent": self.name,
            "games": self.games,
            "win_rate": self.wins / n,
            "mean_vp": self.victory_points / n,
            "mean_turns": self.turns / n,
        }
        out.update({f"mean_{k}": float(v / n) for k, v in zip(STAT_NAMES, self.stats, strict=True)})
        total = max(1.0, float(self.opening.sum()))
        out.update(
            {
                f"opening_{k}": float(v / total)
                for k, v in zip(PURCHASE_KINDS, self.opening, strict=True)
            }
        )
        return out


def play_tracked(
    agents: Sequence[Agent], seed: int, engine: EngineConfig | None = None
) -> tuple[Game, list[StrategyStats]]:
    """One game between `agents` (seat order) with every seat's statistics tracked."""
    game = Game(engine, seed)
    stats = [StrategyStats() for _ in range(NUM_PLAYERS)]
    for seat, agent in enumerate(agents):
        agent.reset(game, seat, derive_seed(seed, "seat", seat))
    while not game.is_over:
        seat, phase = game.decision_player, game.phase
        action = agents[seat].act(game, seat, game.legal_actions())
        track_step(stats, game.step(action), phase)
    return game, stats


def behaviour_profile(
    agent_factories: Sequence,
    games: int,
    seed: int = 0,
    engine: EngineConfig | None = None,
    rotate: bool = True,
) -> list[dict]:
    """Per-agent profile over `games` boards (each in four rotations when `rotate`)."""
    profiles: dict[str, AgentProfile] = {}
    rotations = range(NUM_PLAYERS) if rotate else range(1)
    for g in range(games):
        game_seed = derive_seed(seed, "profile", g)
        for shift in rotations:
            agents = [agent_factories[(i - shift) % NUM_PLAYERS]() for i in range(NUM_PLAYERS)]
            game, stats = play_tracked(agents, game_seed, engine)
            for seat, agent in enumerate(agents):
                profile = profiles.setdefault(agent.name, AgentProfile(agent.name))
                profile.add(stats[seat], game.winner == seat, game.victory_points(seat), game.turn)
    return [p.summary() for p in profiles.values()]


def format_profiles(rows: list[dict]) -> str:
    keys = ["games", "win_rate", "mean_vp", *(f"mean_{k}" for k in STAT_NAMES)]
    keys += [f"opening_{k}" for k in PURCHASE_KINDS]
    width = max(len(r["agent"]) for r in rows) + 2
    header = "agent".ljust(width) + " ".join(k.replace("mean_", "")[:10].rjust(10) for k in keys)
    lines = [header]
    for r in rows:
        values = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys]
        lines.append(r["agent"].ljust(width) + " ".join(v.rjust(10) for v in values))
    return "\n".join(lines)


def aggregate(rows: list[dict]) -> dict:
    """Mean of numeric columns (for summaries over several runs)."""
    out = defaultdict(float)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, int | float):
                out[k] += v / len(rows)
    return dict(out)
