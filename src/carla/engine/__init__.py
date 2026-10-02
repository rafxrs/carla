"""Catan rules engine: pure Python (stdlib only), deterministic, snapshot-able.

Entry points: `Game` (play, legal actions, snapshots), `GameRecord` (logs, replay to any step).
"""

from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS, NUM_RESOURCES, Building, DevCard, Resource
from carla.engine.events import Event, EventKind
from carla.engine.game import Game, IllegalActionError, Snapshot, Step
from carla.engine.record import GameRecord, ReplayError
from carla.engine.state import GameState, Phase

__all__ = [
    "NUM_PLAYERS",
    "NUM_RESOURCES",
    "Action",
    "ActionType",
    "Board",
    "Building",
    "DevCard",
    "EngineConfig",
    "Event",
    "EventKind",
    "Game",
    "GameRecord",
    "GameState",
    "IllegalActionError",
    "Phase",
    "ReplayError",
    "Resource",
    "Snapshot",
    "Step",
    "TradeOffer",
]
