"""Observation encoding from one seat's point of view (no hidden information of other seats).

All seat-indexed features use relative seats (0 = the observer). Node arrays follow the engine's
ids (`carla.engine.topology`) and carry no coordinates, so graph models stay invariant to board
rotations and reflections; the MLP baseline simply flattens them.
"""

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from carla.engine.constants import (
    DESERT,
    NUM_DEV_CARD_TYPES,
    NUM_PLAYERS,
    NUM_RESOURCES,
    PIPS,
    DevCard,
)
from carla.engine.events import NUM_EVENT_KINDS, Event, EventKind
from carla.engine.game import Game
from carla.engine.rules import playable_dev_cards, trade_ratios, victory_points
from carla.engine.state import Phase
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, NUM_TILES
from carla.envs.action_space import MASK_SHAPES, build_masks

TILE_FEATURES = 8  # terrain one-hot (5 resources + desert), dice probability, robber
CORNER_FEATURES = 14  # settlement/city of each relative seat (8), port type (6)
EDGE_FEATURES = 4  # road of each relative seat
PLAYER_FEATURES = 16
GLOBAL_FEATURES = len(Phase) + NUM_RESOURCES + 1 + 11 + 2 + 1 + 1 + NUM_RESOURCES
TRADE_BASE_FEATURES = 2 * NUM_RESOURCES + NUM_PLAYERS + 2
TRADE_FEATURES = TRADE_BASE_FEATURES + NUM_PLAYERS * 2 * NUM_RESOURCES
EVENT_FEATURES = (
    NUM_EVENT_KINDS
    + 2 * (NUM_PLAYERS + 1)
    + 2 * NUM_RESOURCES
    + 1
    + 11
    + NUM_RESOURCES
    + NUM_PLAYERS
    + 1
    + 2 * NUM_RESOURCES
)

_RESOURCE_EVENTS = (EventKind.DISCARDED, EventKind.RESOURCE_STOLEN, EventKind.MONOPOLY_TAKEN)
_DEV_EVENTS = (EventKind.DEV_CARD_BOUGHT, EventKind.DEV_CARD_PLAYED)


class ObservationEncoder:
    """Turns (game, observer, recent events) into a dict of float32 arrays plus action masks.

    Keys and shapes: tiles (19, TILE_FEATURES), corners (54, CORNER_FEATURES),
    edges (72, EDGE_FEATURES), players (4, PLAYER_FEATURES), hand (5,) own resources,
    dev_cards (10,) own cards held + playable now, globals (GLOBAL_FEATURES,),
    trade_context (TRADE_FEATURES,) public pending offer, acceptances, remaining limits and
    counteroffers by relative seat,
    events (max_events, EVENT_FEATURES) + event_mask: public events since the observer's last
    decision (input of the belief module), masks: see `carla.envs.action_space`.
    """

    def __init__(self, max_events: int = 64) -> None:
        self.max_events = max_events
        self._static_board = None
        self._static: tuple[np.ndarray, np.ndarray] | None = None

    def observation_space(self) -> spaces.Dict:
        def box(*shape: int) -> spaces.Box:
            return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

        return spaces.Dict(
            {
                "tiles": box(NUM_TILES, TILE_FEATURES),
                "corners": box(NUM_CORNERS, CORNER_FEATURES),
                "edges": box(NUM_EDGES, EDGE_FEATURES),
                "players": box(NUM_PLAYERS, PLAYER_FEATURES),
                "hand": box(NUM_RESOURCES),
                "dev_cards": box(2 * NUM_DEV_CARD_TYPES),
                "globals": box(GLOBAL_FEATURES),
                "trade_context": box(TRADE_FEATURES),
                "events": box(self.max_events, EVENT_FEATURES),
                "event_mask": spaces.MultiBinary(self.max_events),
                "masks": spaces.Dict({k: spaces.MultiBinary(s) for k, s in MASK_SHAPES.items()}),
            }
        )

    def encode(self, game: Game, observer: int, events: Sequence[Event] = ()) -> dict:
        """Observation of `observer`; `events` must already be redacted for that seat."""
        s = game.state
        rel = [(p - observer) % NUM_PLAYERS for p in range(NUM_PLAYERS)]
        tiles, port_features = self._board_features(game)

        tiles = tiles.copy()
        tiles[s.robber_tile, 7] = 1.0

        corners = np.zeros((NUM_CORNERS, CORNER_FEATURES), dtype=np.float32)
        corners[:, 8:] = port_features
        owner = np.asarray(s.corner_owner)
        occupied = np.flatnonzero(owner >= 0)
        if occupied.size:
            seats = np.asarray(rel)[owner[occupied]]
            levels = np.asarray(s.corner_building)[occupied]
            corners[occupied, 2 * seats + levels - 1] = 1.0

        edges = np.zeros((NUM_EDGES, EDGE_FEATURES), dtype=np.float32)
        road_owner = np.asarray(s.edge_owner)
        built = np.flatnonzero(road_owner >= 0)
        if built.size:
            edges[built, np.asarray(rel)[road_owner[built]]] = 1.0

        players = np.zeros((NUM_PLAYERS, PLAYER_FEATURES), dtype=np.float32)
        for p in range(NUM_PLAYERS):
            played = s.dev_played[p]
            players[rel[p]] = (
                sum(s.hands[p]) / 20,
                sum(s.dev_hands[p]) / 10,
                played[DevCard.KNIGHT] / 5,
                played[DevCard.ROAD_BUILDING] / 2,
                played[DevCard.YEAR_OF_PLENTY] / 2,
                played[DevCard.MONOPOLY] / 2,
                victory_points(s, p, include_hidden=False) / 10,
                s.road_lengths[p] / 15,
                s.longest_road_owner == p,
                s.largest_army_owner == p,
                s.settlements_left[p] / 5,
                s.cities_left[p] / 4,
                s.roads_left[p] / 15,
                s.discards_pending[p] / 10,
                s.current_player == p,
                s.decision_player == p,
            )

        hand = np.asarray(s.hands[observer], dtype=np.float32) / 10
        dev_cards = np.zeros(2 * NUM_DEV_CARD_TYPES, dtype=np.float32)
        dev_cards[:NUM_DEV_CARD_TYPES] = np.asarray(s.dev_hands[observer]) / 5
        dev_cards[NUM_DEV_CARD_TYPES:] = np.asarray(playable_dev_cards(s, observer)) > 0

        g = np.zeros(GLOBAL_FEATURES, dtype=np.float32)
        g[s.phase] = 1.0
        i = len(Phase)
        g[i : i + NUM_RESOURCES] = np.asarray(s.bank) / 19
        i += NUM_RESOURCES
        g[i] = sum(s.dev_deck) / 25
        i += 1
        if s.last_roll:
            g[i + s.last_roll - 2] = 1.0
        i += 11
        g[i], g[i + 1], g[i + 2] = s.has_rolled, s.dev_played_this_turn, s.free_roads / 2
        g[i + 3] = min(s.turn / 500, 1.0)
        i += 4
        g[i : i + NUM_RESOURCES] = np.asarray(trade_ratios(s, observer)) / 4

        trade_context = np.zeros(TRADE_FEATURES, dtype=np.float32)
        if s.pending_trade is not None:
            trade_context[:NUM_RESOURCES] = np.asarray(s.pending_trade.give) / 4
            trade_context[NUM_RESOURCES : 2 * NUM_RESOURCES] = (
                np.asarray(s.pending_trade.receive) / 4
            )
        offset = 2 * NUM_RESOURCES
        for p in range(NUM_PLAYERS):
            trade_context[offset + rel[p]] = s.trade_acceptances[p]
        proposal_limit = s.config.max_trade_proposals_per_turn
        trade_limit = s.config.max_domestic_trades_per_turn
        trade_context[TRADE_BASE_FEATURES - 2] = (
            1.0
            if proposal_limit is None
            else max(0, proposal_limit - s.trade_proposals_this_turn) / max(1, proposal_limit)
        )
        trade_context[TRADE_BASE_FEATURES - 1] = (
            1.0
            if trade_limit is None
            else max(0, trade_limit - s.domestic_trades_this_turn) / max(1, trade_limit)
        )
        for p, offer in enumerate(s.trade_counteroffers):
            if offer is None:
                continue
            start = TRADE_BASE_FEATURES + rel[p] * 2 * NUM_RESOURCES
            trade_context[start : start + NUM_RESOURCES] = np.asarray(offer.give) / 4
            trade_context[start + NUM_RESOURCES : start + 2 * NUM_RESOURCES] = (
                np.asarray(offer.receive) / 4
            )

        event_array, event_mask = self._encode_events(events, observer)
        masks = (
            build_masks(game.legal_actions(), observer)
            if observer == game.decision_player
            else {k: np.zeros(shape, dtype=bool) for k, shape in MASK_SHAPES.items()}
        )
        return {
            "tiles": tiles,
            "corners": corners,
            "edges": edges,
            "players": players,
            "hand": hand,
            "dev_cards": dev_cards,
            "globals": g,
            "trade_context": trade_context,
            "events": event_array,
            "event_mask": event_mask,
            "masks": masks,
        }

    def _board_features(self, game: Game) -> tuple[np.ndarray, np.ndarray]:
        """Per-game constant features (cached while the board is unchanged)."""
        board = game.board
        if board is not self._static_board:
            tiles = np.zeros((NUM_TILES, TILE_FEATURES), dtype=np.float32)
            for t, (r, n) in enumerate(zip(board.tile_resource, board.tile_number, strict=True)):
                tiles[t, NUM_RESOURCES if r == DESERT else r] = 1.0
                tiles[t, 6] = PIPS[n] / 36 if n else 0.0
            ports = np.zeros((NUM_CORNERS, 6), dtype=np.float32)
            for c, kind in enumerate(board.corner_port):
                if kind >= 0:
                    ports[c, kind] = 1.0
            self._static_board, self._static = board, (tiles, ports)
        assert self._static is not None
        return self._static

    def _encode_events(
        self, events: Sequence[Event], observer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((self.max_events, EVENT_FEATURES), dtype=np.float32)
        mask = np.zeros(self.max_events, dtype=bool)
        recent = list(events)[-self.max_events :]
        for row, e in enumerate(recent):
            mask[row] = True
            v = out[row]
            v[e.kind] = 1.0
            i = NUM_EVENT_KINDS
            v[i + (NUM_PLAYERS if e.player < 0 else (e.player - observer) % NUM_PLAYERS)] = 1.0
            i += NUM_PLAYERS + 1
            v[i + (NUM_PLAYERS if e.target < 0 else (e.target - observer) % NUM_PLAYERS)] = 1.0
            i += NUM_PLAYERS + 1
            if e.kind in _RESOURCE_EVENTS and e.value >= 0:
                v[i + e.value] = 1.0
            elif e.kind in _DEV_EVENTS and e.value >= 0:
                v[i + NUM_RESOURCES + e.value] = 1.0
            i += 2 * NUM_RESOURCES
            v[i] = e.value < 0 and e.kind in (EventKind.RESOURCE_STOLEN, EventKind.DEV_CARD_BOUGHT)
            i += 1
            if e.kind == EventKind.DICE_ROLLED:
                v[i + e.value - 2] = 1.0
            i += 11
            if e.kind == EventKind.MONOPOLY_TAKEN:
                for p, n in enumerate(e.counts):
                    v[i + NUM_RESOURCES + (p - observer) % NUM_PLAYERS] = n / 10
            elif e.counts and e.kind != EventKind.DICE_ROLLED:
                v[i : i + NUM_RESOURCES] = np.asarray(e.counts) / 5
            if e.kind == EventKind.TRADE_RESPONDED:
                v[-(2 * NUM_RESOURCES + 1)] = e.value
            if e.give:
                v[-2 * NUM_RESOURCES : -NUM_RESOURCES] = np.asarray(e.give) / 4
            if e.receive:
                v[-NUM_RESOURCES:] = np.asarray(e.receive) / 4
        return out, mask
