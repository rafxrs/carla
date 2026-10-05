"""Headless self-play with uniformly random legal actions (engine smoke test and benchmark)."""

from carla.engine.config import EngineConfig
from carla.engine.game import Game
from carla.engine.rng import Rng, derive_seed


def play_random_game(
    seed: int = 0, config: EngineConfig | None = None, by_type: bool = True
) -> Game:
    """Play one game to the end with random legal actions and return it.

    With `by_type`, an action type is drawn uniformly first and then one of its actions, like a
    factored policy with uniform heads; otherwise every legal action is equally likely.
    """
    game = Game(config, seed)
    rng = Rng(derive_seed(seed, "random-policy"))
    while not game.is_over:
        actions = game.legal_actions()
        if by_type:
            types = sorted({a.type for a in actions})
            kind = types[rng.randbelow(len(types))]
            actions = [a for a in actions if a.type == kind]
        game.step(actions[rng.randbelow(len(actions))], validate=False)
    return game
