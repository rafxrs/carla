"""Arcade front-end (TODO: gui workstream).

Two modes, both on top of `carla.engine` only:
- Play: a human seat is a `HumanAgent` whose `act` waits for clicks on highlighted legal
  targets (corners/edges/tiles from `game.legal_actions()`) or on buttons (roll, end turn,
  trade, development cards); the other seats are any `Agent` (bots or trained policies).
- Watch / replay: step through a `GameRecord` (`record.replay(k)` gives position k) with a
  configurable delay, a log panel showing `describe_event` lines, and pause/step/seek controls.
Drawing: `carla.engine.topology.TILE_XY / CORNER_XY / EDGE_XY` give unit-hex coordinates
(y grows downwards); scale and translate them to the window. `scripts/render_board.py` is a
matplotlib reference renderer of the same geometry.
"""

from collections.abc import Sequence

from carla.agents.base import Agent
from carla.engine.actions import Action
from carla.engine.game import Game


class HumanAgent(Agent):
    """Seat controlled through the GUI. TODO(gui)."""

    name = "human"

    def act(self, game: Game, seat: int, legal: Sequence[Action]) -> Action:
        raise NotImplementedError("gui workstream: wait for the human's choice in the window")


def main(record_path: str | None = None) -> None:
    """Open the replay viewer for `record_path`, or a new game against bots. TODO(gui)."""
    raise NotImplementedError("gui workstream: Arcade viewer (pip install -e '.[gui]')")
