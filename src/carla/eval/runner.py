"""Play one game between agents."""

from collections.abc import Sequence
from dataclasses import dataclass

from carla.agents.base import Agent
from carla.engine.actions import ActionType
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
    decisions: tuple[int, ...] = ()
    trade_offers: tuple[int, ...] = ()
    multi_card_offers: tuple[int, ...] = ()
    public_victory_points: tuple[int, ...] = ()

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
    decisions, offers, multi = [0] * 4, [0] * 4, [0] * 4
    try:
        for seat, agent in enumerate(agents):
            agent.reset(game, seat, derive_seed(seed, "seat", seat))
        while not game.is_over:
            seat = game.decision_player
            action = agents[seat].act(game, seat, game.legal_actions())
            if not game.is_legal(action):
                raise ValueError(f"{agents[seat].name} returned an illegal action: {action}")
            events = game.step(action)
            decisions[seat] += 1
            if action.type == ActionType.PROPOSE_TRADE:
                offers[seat] += 1
                if max(sum(action.trade.give), sum(action.trade.receive)) > 1:
                    multi[seat] += 1
            for observer, agent in enumerate(agents):
                agent.observe(tuple(e.visible_to(observer) for e in events), game.num_steps)
        for agent in agents:
            agent.finish({"winner": game.winner, "turns": game.turn})
    finally:
        # Attempt every cleanup even when an earlier cleanup itself fails.
        import sys

        active_error = sys.exc_info()[0] is not None
        errors = []
        for agent in agents:
            try:
                agent.close()
            except Exception as error:
                errors.append(error)
        if errors and not active_error:
            raise ExceptionGroup("agent cleanup failed", errors)
    names = tuple(a.name for a in agents)
    return GameResult(
        seed=seed,
        agents=names,
        winner=game.winner,
        victory_points=tuple(game.victory_points(p) for p in range(NUM_PLAYERS)),
        turns=game.turn,
        steps=game.num_steps,
        record=GameRecord.from_game(game, {"agents": list(names)}) if record else None,
        decisions=tuple(decisions),
        trade_offers=tuple(offers),
        multi_card_offers=tuple(multi),
        public_victory_points=tuple(
            game.victory_points(p, include_hidden=False) for p in range(NUM_PLAYERS)
        ),
    )
