"""Inventory-constrained worlds. Never clone the real referee for planning.

This initial prior conditions on current public inventories, not the complete event
history. It is explicitly distinct from a history-conditioned belief model.
"""

from carla.agents.api import AgentView
from carla.engine.constants import DEV_CARD_COUNTS
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.state import GameState


def sample_world(view: AgentView, seed: int) -> Game:
    data = view.state
    rng = Rng(seed)
    hand_sizes = data.pop("hand_sizes")
    dev_sizes = data.pop("development_sizes")
    dev_remaining = data.pop("development_remaining")
    own = view.seat
    resources = [19 - data["bank"][r] - data["hands"][own][r] for r in range(5)]
    cards = [r for r, n in enumerate(resources) for _ in range(n)]
    if any(n < 0 for n in resources) or len(cards) != sum(hand_sizes) - hand_sizes[own]:
        raise ValueError("inconsistent public resource inventory")
    rng.shuffle(cards)
    for p in range(4):
        if p != own:
            hand = [0] * 5
            for _ in range(hand_sizes[p]):
                hand[cards.pop()] += 1
            data["hands"][p] = hand
    devs = [
        DEV_CARD_COUNTS[c] - data["dev_hands"][own][c] - sum(h[c] for h in data["dev_played"])
        for c in range(5)
    ]
    deck = [c for c, n in enumerate(devs) for _ in range(n)]
    if any(n < 0 for n in devs) or len(deck) != dev_remaining + sum(dev_sizes) - dev_sizes[own]:
        raise ValueError("inconsistent public development inventory")
    rng.shuffle(deck)
    for p in range(4):
        if p != own:
            hand = [0] * 5
            for _ in range(dev_sizes[p]):
                hand[deck.pop()] += 1
            data["dev_hands"][p] = hand
    data["dev_deck"] = [deck.count(c) for c in range(5)]
    if data["dev_bought_this_turn"] is None:
        # Other players' new-card types are not public. The actor's types are exact.
        data["dev_bought_this_turn"] = [0] * 5
    world = Game.__new__(Game)
    world.seed = seed
    world.state = GameState.from_dict(data)
    world.rng = Rng(seed)
    world.history = []
    world._legal = world._legal_set = None
    return world
