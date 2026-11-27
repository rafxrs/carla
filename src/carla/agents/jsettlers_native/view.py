"""Public board-query state, explicitly unsuitable for referee transitions."""

from carla.engine.state import GameState


def query_state(view, knowledge):
    data = view.state
    data.pop("hand_sizes")
    data.pop("development_sizes")
    remaining = data.pop("development_remaining")
    data["hands"] = [hand.known[:] for hand in knowledge.hands]
    data["hands"][view.seat] = view.state["hands"][view.seat]
    data["dev_hands"] = [hand if hand is not None else [0] * 5 for hand in data["dev_hands"]]
    # Only the public remaining count is used by these algorithms. This is
    # deliberately not a hypothetical deck, and must never be sampled/stepped.
    data["dev_deck"] = [remaining, 0, 0, 0, 0]
    if data["dev_bought_this_turn"] is None:
        data["dev_bought_this_turn"] = [0] * 5
    return GameState.from_dict(data)
