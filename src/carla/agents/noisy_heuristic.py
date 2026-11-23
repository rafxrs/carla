from collections.abc import Sequence

from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.random_agent import RandomAgent
from carla.engine.actions import Action
from carla.engine.game import Game
from carla.engine.rng import Rng, derive_seed


class NoisyHeuristicAgent(Agent):
    """Follow the heuristic with probability p, otherwise use random play."""

    def __init__(self, heuristic_probability: float):
        if not 0.0 <= heuristic_probability <= 1.0:
            raise ValueError("heuristic_probability must be in [0, 1]")

        self.heuristic_probability = heuristic_probability

        self.heuristic = HeuristicAgent()
        self.random = RandomAgent()

        self._rng = Rng(0)

    def reset(self, game: Game, seat: int, seed: int) -> None:
        self._rng = Rng(derive_seed(seed, "noisy-heuristic-selector"))

        self.heuristic.reset(game, seat, seed)
        self.random.reset(game, seat, seed)

    def act(
        self,
        game: Game,
        seat: int,
        legal: Sequence[Action],
    ) -> Action:

        if self._rng.random() < self.heuristic_probability:
            return self.heuristic.act(game, seat, legal)

        return self.random.act(game, seat, legal)
