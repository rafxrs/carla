"""Play one game between agents."""

from collections.abc import Sequence
from dataclasses import dataclass

from carla.agents.base import Agent
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.game import Game
from carla.engine.record import GameRecord
from carla.engine.rng import derive_seed


@dataclass(frozen=True)
class GameResult:
    seed: int
    agents: tuple[str, ...]
    """Agent name per seat."""
    winner: int | None
    victory_points: tuple[int, ...]
    turns: int
    steps: int
    record: GameRecord | None = None

    @property
    def winner_name(self) -> str | None:
        return None if self.winner is None else self.agents[self.winner]


def play_game(
    agents: Sequence[Agent],
    seed: int,
    config: EngineConfig | None = None,
    record: bool = False,
) -> GameResult:
    """Play a full game with `agents[i]` in seat i; agent i is seeded from (seed, i)."""
    if len(agents) != NUM_PLAYERS:
        raise ValueError(f"need {NUM_PLAYERS} agents")
    game = Game(config, seed)
    for seat, agent in enumerate(agents):
        agent.reset(game, seat, derive_seed(seed, "seat", seat))
    while not game.is_over:
        seat = game.decision_player
        game.step(agents[seat].act(game, seat, game.legal_actions()))
    names = tuple(a.name for a in agents)
    return GameResult(
        seed=seed,
        agents=names,
        winner=game.winner,
        victory_points=tuple(game.victory_points(p) for p in range(NUM_PLAYERS)),
        turns=game.turn,
        steps=game.num_steps,
        record=GameRecord.from_game(game, {"agents": list(names)}) if record else None,
    )
