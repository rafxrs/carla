"""Full game state as flat Python containers (fast to copy, trivially serialisable)."""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Any

from carla.engine.actions import TradeOffer
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.constants import (
    CITIES_PER_PLAYER,
    DEV_CARD_COUNTS,
    NUM_DEV_CARD_TYPES,
    NUM_PLAYERS,
    NUM_RESOURCES,
    RESOURCE_CARDS_PER_TYPE,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
)
from carla.engine.topology import NUM_CORNERS, NUM_EDGES


class Phase(IntEnum):
    """Which decision the game is waiting for (see the phase diagram in the README)."""

    SETUP_SETTLEMENT = 0
    SETUP_ROAD = 1
    PRE_ROLL = 2
    DISCARD = 3
    MOVE_ROBBER = 4
    MAIN = 5
    ROAD_BUILDING = 6
    TRADE_RESPONSE = 7
    TRADE_CONFIRM = 8
    GAME_OVER = 9


SETUP_ORDER = (0, 1, 2, 3, 3, 2, 1, 0)
"""Seat placing a settlement + road at each setup step (snake order). Seat 0 then starts."""


def _per_player(value: int) -> Callable[[], list[int]]:
    return lambda: [value] * NUM_PLAYERS


def _matrix(width: int) -> Callable[[], list[list[int]]]:
    return lambda: [[0] * width for _ in range(NUM_PLAYERS)]


@dataclass(slots=True, eq=False)
class GameState:
    """Everything needed to continue a game. Seats are absolute (0..3).

    Hidden information: `hands` (only sizes are public) and `dev_hands` (only sizes are public).
    """

    board: Board
    config: EngineConfig
    phase: Phase = Phase.SETUP_SETTLEMENT
    current_player: int = 0
    """Seat whose turn it is."""
    decision_player: int = 0
    """Seat that must act now (differs from current_player while others discard); -1 when over."""
    turn: int = 0
    """Number of player turns completed since setup."""
    setup_step: int = 0
    setup_corner: int = -1
    """Settlement just placed during setup; its road must touch it."""
    has_rolled: bool = False
    last_roll: int = 0
    dev_played_this_turn: bool = False
    dev_bought_this_turn: list[int] = field(default_factory=lambda: [0] * NUM_DEV_CARD_TYPES)
    free_roads: int = 0
    robber_tile: int = -1
    corner_owner: list[int] = field(default_factory=lambda: [-1] * NUM_CORNERS)
    corner_building: list[int] = field(default_factory=lambda: [0] * NUM_CORNERS)
    edge_owner: list[int] = field(default_factory=lambda: [-1] * NUM_EDGES)
    hands: list[list[int]] = field(default_factory=_matrix(NUM_RESOURCES))
    bank: list[int] = field(default_factory=lambda: [RESOURCE_CARDS_PER_TYPE] * NUM_RESOURCES)
    dev_deck: list[int] = field(default_factory=lambda: list(DEV_CARD_COUNTS))
    dev_hands: list[list[int]] = field(default_factory=_matrix(NUM_DEV_CARD_TYPES))
    """Unplayed development cards per seat, including victory point cards."""
    dev_played: list[list[int]] = field(default_factory=_matrix(NUM_DEV_CARD_TYPES))
    settlements_left: list[int] = field(default_factory=_per_player(SETTLEMENTS_PER_PLAYER))
    cities_left: list[int] = field(default_factory=_per_player(CITIES_PER_PLAYER))
    roads_left: list[int] = field(default_factory=_per_player(ROADS_PER_PLAYER))
    road_lengths: list[int] = field(default_factory=_per_player(0))
    """Longest continuous road of each seat, kept up to date after every build."""
    longest_road_owner: int = -1
    largest_army_owner: int = -1
    discards_pending: list[int] = field(default_factory=_per_player(0))
    winner: int = -1
    pending_trade: TradeOffer | None = None
    """Terms the current player offered in the negotiation under way (None: none, or it only
    asked for offers). Like the rest of a negotiation, public."""
    trade_proposals_this_turn: int = 0
    """Negotiations the current player opened this turn (offers and invitations)."""
    domestic_trades_this_turn: int = 0
    """Domestic trades the current player completed this turn."""
    trade_responses: int = 0
    """Seats that have answered the negotiation (0..3), in turn order after the current player."""
    trade_acceptances: list[bool] = field(default_factory=lambda: [False] * NUM_PLAYERS)
    """Seats that accepted `pending_trade`."""
    trade_counteroffers: list[TradeOffer | None] = field(
        default_factory=lambda: [None] * NUM_PLAYERS
    )
    """Each seat's counteroffer, from that seat's side (what it gives, what it wants)."""

    @classmethod
    def initial(cls, board: Board, config: EngineConfig) -> "GameState":
        return cls(board=board, config=config, robber_tile=board.desert_tile)

    def copy(self) -> "GameState":
        """Independent copy; `board` and `config` are immutable and shared."""
        new = object.__new__(GameState)
        for name in _SHARED:
            object.__setattr__(new, name, getattr(self, name))
        for name in _LISTS:
            object.__setattr__(new, name, getattr(self, name)[:])
        for name in _MATRICES:
            object.__setattr__(new, name, [row[:] for row in getattr(self, name)])
        return new

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation (inverse of `from_dict`). Without domestic trade, the
        negotiation fields (then always at their defaults) are left out, so such a state
        serializes and hashes exactly as before domestic trade existed."""
        out: dict[str, Any] = {}
        trading = self.config.allow_domestic_trade
        for f in fields(self):
            if f.name in _NEGOTIATION and not trading:
                continue
            value = getattr(self, f.name)
            if f.name in ("board", "config"):
                value = value.to_dict()
            elif f.name == "pending_trade":
                value = _offer_to_json(value)
            elif f.name == "trade_counteroffers":
                value = [_offer_to_json(offer) for offer in value]
            elif isinstance(value, list):
                value = [row[:] if isinstance(row, list) else row for row in value]
            elif isinstance(value, IntEnum):
                value = int(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameState":
        """Fields missing from `data` (the negotiation without domestic trade) get defaults."""
        kwargs = dict(data)
        kwargs["board"] = Board.from_dict(data["board"])
        kwargs["config"] = EngineConfig.from_dict(data["config"])
        kwargs["phase"] = Phase(data["phase"])
        kwargs["pending_trade"] = _offer_from_json(data["pending_trade"])
        for name in _LISTS:
            if name in data:
                kwargs[name] = list(data[name])
        if "trade_counteroffers" in data:
            kwargs["trade_counteroffers"] = [
                _offer_from_json(o) for o in data["trade_counteroffers"]
            ]
        for name in _MATRICES:
            kwargs[name] = [list(row) for row in data[name]]
        return cls(**kwargs)

    def hand_size(self, player: int) -> int:
        return sum(self.hands[player])


_MATRICES = ("hands", "dev_hands", "dev_played")
_LISTS = tuple(
    f.name for f in fields(GameState) if f.name not in _MATRICES and str(f.type).startswith("list[")
)
_SHARED = tuple(f.name for f in fields(GameState) if f.name not in _LISTS + _MATRICES)
_NEGOTIATION = (
    "trade_proposals_this_turn",
    "domestic_trades_this_turn",
    "trade_responses",
    "trade_acceptances",
    "trade_counteroffers",
)
"""Fields added with domestic trade (`pending_trade` was reserved for it from the start)."""


def _offer_to_json(offer: TradeOffer | None) -> list[list[int]] | None:
    return None if offer is None else [list(offer.give), list(offer.receive)]


def _offer_from_json(data: list[list[int]] | None) -> TradeOffer | None:
    return None if data is None else TradeOffer(tuple(data[0]), tuple(data[1]))
