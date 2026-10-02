"""Uniformly random legal play (the weakest baseline)."""

from collections.abc import Sequence

from carla.agents.base import Agent
from carla.engine.actions import Action
from carla.engine.game import Game
from carla.engine.rng import Rng, derive_seed


class RandomAgent(Agent):
    """Picks an action type uniformly, then one of its actions (or any action if not `by_type`)."""

    name = "random"

    def __init__(self, by_type: bool = True) -> None:
        self.by_type = by_type
        self._rng = Rng(0)

    def reset(self, game: Game, seat: int, seed: int) -> None:
        self._rng = Rng(derive_seed(seed, "random-agent"))

    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        if self.by_type:
            types = sorted({a.type for a in legal})
            kind = types[self._rng.randbelow(len(types))]
            legal = [a for a in legal if a.type == kind]
        return legal[self._rng.randbelow(len(legal))]
