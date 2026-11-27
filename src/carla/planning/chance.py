"""Enumerate exact chance outcomes in a private, complete CARLA world."""

from carla.engine.actions import ActionType
from carla.engine.constants import PIPS


def outcomes(game, action):
    if action.type == ActionType.ROLL_DICE:
        return [((n // 2, n - n // 2), PIPS[n] / 36) for n in range(2, 13)]
    if action.type == ActionType.BUY_DEV_CARD:
        counts = game.state.dev_deck
    elif action.type == ActionType.MOVE_ROBBER and action.player >= 0:
        counts = game.state.hands[action.player]
    else:
        return [(None, 1.0)]
    total = sum(counts)
    return [(c, n / total) for c, n in enumerate(counts) if n] if total else [(None, 1.0)]


def successors(game, action):
    for outcome, probability in outcomes(game, action):
        child = game.clone()
        child.step(action, outcome)
        yield child, probability
