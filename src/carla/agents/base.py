"""The `Agent` interface shared by bots, trained policies, humans (GUI) and the evaluation code."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from carla.engine.actions import Action
from carla.engine.game import Game


class Agent(ABC):
    """Chooses actions for one seat.

    Contract: an agent may read the public part of `game` plus its own seat's hand and
    development cards, never other seats' hidden cards (`game.state.hands[other]`,
    `game.state.dev_hands[other]`) nor `game.rng`. Agents must be deterministic given the seed
    they receive in `reset`, so that tournaments and imitation datasets are reproducible.
    """

    name: str = "agent"

    def reset(self, game: Game, seat: int, seed: int) -> None:  # noqa: B027
        """Called once per game before the first decision."""

    @abstractmethod
    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        """Return one of `legal` (the legal actions of `seat`, which is `game.decision_player`)."""
