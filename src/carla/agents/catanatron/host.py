"""CARLA agents seated in Catanatron's own engine: the reverse of `agent.py`.

Catanatron is the referee here. It seats, rolls, validates and applies every move. For each
decision, `CarlaPlayer` converts Catanatron's state into a private CARLA `Game` and maps the
chosen CARLA action back to one of Catanatron's playable actions. Nothing is carried between
decisions except what the wrapped agent keeps itself.

Information modes:

- "view": the seat's own cards plus public counts (the CARLA referee boundary). Opponents'
  hands and the development deck become placeholders that keep their sizes.
- "open": every hand and the deck composition, which Catanatron's own bots read from the
  `Game` they are given. Use it to compare a port with its original under the original's
  information, never under CARLA's referee.

Mirror (default): `CarlaPlayer` also keeps a CARLA `Game` that replays every action record
Catanatron executes, with its chance outcome (dice, stolen card, drawn card). Agents then
receive a CARLA game with a real history, exactly as under CARLA's referee, plus their
redacted events through `observe`, so history- and event-based agents (PPO checkpoints,
native JSettlers) play unchanged. At each decision the mirror is checked against a direct
conversion of Catanatron's state; any disagreement between the engines raises.

Only the four-player base game without domestic trades is supported. Catanatron never lists
domestic offers among its playable actions.
"""

import random
from collections import Counter
from functools import lru_cache

from catanatron.models.enums import SETTLEMENT, ActionPrompt, ActionType
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
from catanatron.models.player import Player
from catanatron.models.tiles import Port

from carla.agents.base import Agent
from carla.agents.catanatron.coordinates import mappings
from carla.engine.actions import Action
from carla.engine.actions import ActionType as T
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.constants import GENERIC_PORT, DevCard
from carla.engine.events import Event, EventKind
from carla.engine.game import Game
from carla.engine.rng import Rng, derive_seed
from carla.engine.state import GameState, Phase

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
"""Catanatron's resource names in CARLA's resource order."""
CARDS = tuple(card.name for card in DevCard)
"""Development card names, the same strings in both engines, in CARLA's order."""
PLAYABLE_CARDS = CARDS[:4]


@lru_cache(maxsize=1)
def frame():
    """Catanatron node id -> corner, axial hex -> tile, node pair -> edge (CARLA ids)."""
    _, nodes, tiles, edges = mappings()
    return (
        {node: corner for corner, node in enumerate(nodes)},
        {axial: tile for tile, axial in enumerate(tiles)},
        {pair: edge for edge, pair in enumerate(edges)},
    )


def axial(coordinate):
    x, _, z = coordinate
    return x, z


def carla_board(catan_map):
    corners, tiles, _ = frame()
    resources, numbers = [-1] * 19, [0] * 19
    for coordinate, tile in catan_map.land_tiles.items():
        t = tiles[axial(coordinate)]
        if tile.resource is not None:
            resources[t] = RESOURCES.index(tile.resource)
            numbers[t] = tile.number
    port_types, port_corners = [], []
    for tile in catan_map.tiles.values():
        if isinstance(tile, Port):
            a, b = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            port_corners.append(tuple(sorted((corners[tile.nodes[a]], corners[tile.nodes[b]]))))
            port_types.append(
                GENERIC_PORT if tile.resource is None else RESOURCES.index(tile.resource)
            )
    order = sorted(range(len(port_corners)), key=port_corners.__getitem__)
    return Board(
        tuple(resources),
        tuple(numbers),
        tuple(port_types[i] for i in order),
        tuple(port_corners[i] for i in order),
    )


def carla_action(action, colors):
    """One of Catanatron's playable actions as a CARLA `Action` (seats index `colors`)."""
    corners, tiles, edges = frame()
    kind, value = action.action_type, action.value
    if kind == ActionType.ROLL:
        return Action(T.ROLL_DICE)
    if kind == ActionType.END_TURN:
        return Action(T.END_TURN)
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(T.BUY_DEV_CARD)
    if kind == ActionType.BUILD_SETTLEMENT:
        return Action(T.BUILD_SETTLEMENT, corner=corners[value])
    if kind == ActionType.BUILD_CITY:
        return Action(T.BUILD_CITY, corner=corners[value])
    if kind == ActionType.BUILD_ROAD:
        return Action(T.BUILD_ROAD, edge=edges[tuple(sorted(value))])
    if kind == ActionType.MOVE_ROBBER:
        coordinate, victim = value[0], value[1]
        return Action(
            T.MOVE_ROBBER,
            tile=tiles[axial(coordinate)],
            player=-1 if victim is None else colors.index(victim),
        )
    if kind == ActionType.DISCARD_RESOURCE:
        return Action(T.DISCARD, resource=RESOURCES.index(value))
    if kind == ActionType.MARITIME_TRADE:
        return Action(
            T.MARITIME_TRADE,
            resource=RESOURCES.index(value[0]),
            resource2=RESOURCES.index(value[-1]),
        )
    if kind == ActionType.PLAY_KNIGHT_CARD:
        return Action(T.PLAY_DEV_CARD, dev=DevCard.KNIGHT)
    if kind == ActionType.PLAY_ROAD_BUILDING:
        return Action(T.PLAY_DEV_CARD, dev=DevCard.ROAD_BUILDING)
    if kind == ActionType.PLAY_MONOPOLY:
        return Action(T.PLAY_DEV_CARD, dev=DevCard.MONOPOLY, resource=RESOURCES.index(value))
    if kind == ActionType.PLAY_YEAR_OF_PLENTY and len(value) == 2:
        a, b = sorted(RESOURCES.index(r) for r in value)
        return Action(T.PLAY_DEV_CARD, dev=DevCard.YEAR_OF_PLENTY, resource=a, resource2=b)
    raise ValueError(f"no CARLA conversion for Catanatron action {action}")


def _phase(state):
    prompt = state.current_prompt
    if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return Phase.SETUP_SETTLEMENT
    if prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        return Phase.SETUP_ROAD
    if prompt == ActionPrompt.DISCARD:
        return Phase.DISCARD
    if prompt == ActionPrompt.MOVE_ROBBER:
        return Phase.MOVE_ROBBER
    if prompt == ActionPrompt.PLAY_TURN:
        if state.is_road_building and state.free_roads_available > 0:
            return Phase.ROAD_BUILDING
        rolled = state.player_state[f"P{state.current_turn_index}_HAS_ROLLED"]
        return Phase.MAIN if rolled else Phase.PRE_ROLL
    raise ValueError(f"unsupported Catanatron prompt {prompt} (domestic trading is not)")


def carla_game(game, seat, information="view"):
    """A private CARLA `Game` for `seat` in a Catanatron game (see the module docstring)."""
    if information not in ("view", "open"):
        raise ValueError("information must be 'view' or 'open'")
    corners, tiles, edges = frame()
    cs = game.state
    ps = cs.player_state
    colors = list(cs.colors)
    if len(colors) != 4:
        raise ValueError("only four-player Catanatron games are supported")
    config = EngineConfig(
        vp_to_win=game.vps_to_win,
        discard_limit=cs.discard_limit,
        allow_domestic_trade=False,
        catanatron_rules=True,
    )
    s = GameState.initial(carla_board(cs.board.map), config)
    s.phase = _phase(cs)
    s.current_player, s.decision_player = cs.current_turn_index, cs.current_player_index
    s.turn = cs.num_turns
    s.has_rolled = bool(ps[f"P{cs.current_turn_index}_HAS_ROLLED"])
    s.robber_tile = tiles[axial(cs.board.robber_coordinate)]
    for node, (color, building) in cs.board.buildings.items():
        s.corner_owner[corners[node]] = colors.index(color)
        s.corner_building[corners[node]] = 1 if building == SETTLEMENT else 2
    for pair, color in cs.board.roads.items():
        s.edge_owner[edges[tuple(sorted(pair))]] = colors.index(color)
    for p in range(4):
        key = f"P{p}"
        hand = [ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES]
        cards = [ps[f"{key}_{c}_IN_HAND"] for c in CARDS]
        hidden = information == "view" and p != seat
        s.hands[p] = [sum(hand), 0, 0, 0, 0] if hidden else hand
        s.dev_hands[p] = [sum(cards), 0, 0, 0, 0] if hidden else cards
        s.dev_played[p] = [ps.get(f"{key}_PLAYED_{c}", 0) for c in CARDS]
        s.roads_left[p] = ps[f"{key}_ROADS_AVAILABLE"]
        s.settlements_left[p] = ps[f"{key}_SETTLEMENTS_AVAILABLE"]
        s.cities_left[p] = ps[f"{key}_CITIES_AVAILABLE"]
        # The board keeps live lengths; player_state lags during the initial placements.
        s.road_lengths[p] = cs.board.road_lengths.get(colors[p], 0)
        if ps[f"{key}_HAS_ROAD"]:
            s.longest_road_owner = p
        if ps[f"{key}_HAS_ARMY"]:
            s.largest_army_owner = p
    turn = f"P{cs.current_turn_index}"
    # Catanatron records, per card type, whether one was owned when the turn started.
    s.dev_bought_this_turn = [
        0 if ps[f"{turn}_{c}_OWNED_AT_START"] else ps[f"{turn}_{c}_IN_HAND"] for c in PLAYABLE_CARDS
    ] + [0]
    s.dev_played_this_turn = bool(ps[f"{turn}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])
    s.bank = list(cs.resource_freqdeck)
    deck = Counter(cs.development_listdeck)
    s.dev_deck = (
        [deck[c] for c in CARDS]
        if information == "open"
        else [len(cs.development_listdeck), 0, 0, 0, 0]
    )
    s.discards_pending = list(cs.discard_counts)
    if s.phase == Phase.ROAD_BUILDING:
        s.free_roads = min(cs.free_roads_available, s.roads_left[s.decision_player])
    if s.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        built = sum(1 for _, b in cs.board.buildings.values() if b == SETTLEMENT)
        s.setup_step = built - (s.phase == Phase.SETUP_ROAD)
        if s.phase == Phase.SETUP_ROAD:
            own = colors[s.decision_player]
            last = next(
                r.action.value
                for r in reversed(cs.action_records)
                if r.action.action_type == ActionType.BUILD_SETTLEMENT and r.action.color == own
            )
            s.setup_corner = corners[last]
    carla = Game.__new__(Game)
    carla.seed = 0
    carla.state = s
    carla.rng = Rng(0)
    # Placeholder steps: `num_steps` then counts Catanatron's executed actions, which agents
    # use to derive per-decision seeds. They are never replayed.
    carla.history = [None] * len(cs.action_records)
    carla._legal = carla._legal_set = None
    return carla


def record_outcome(record):
    """The CARLA chance outcome of an executed Catanatron action record (None if none)."""
    kind, result = record.action.action_type, record.result
    if kind == ActionType.ROLL:
        return tuple(result)
    if kind == ActionType.MOVE_ROBBER and record.action.value[1] is not None:
        return None if result is None else RESOURCES.index(result)
    if kind == ActionType.BUY_DEVELOPMENT_CARD:
        return CARDS.index(result)
    return None


_CHECKED = (
    "road_lengths",
    "phase",
    "decision_player",
    "current_player",
    "hands",
    "dev_hands",
    "dev_played",
    "corner_owner",
    "corner_building",
    "edge_owner",
    "robber_tile",
    "bank",
    "longest_road_owner",
    "largest_army_owner",
)


class CarlaPlayer(Player):
    """A CARLA `Agent` playing one color in a Catanatron game."""

    def __init__(
        self, color, agent: Agent, *, information="view", seed=0, share_rng=True, mirror=True
    ):
        """`share_rng`: an upstream-chance agent draws from Catanatron's own `random`, as the
        original does (its draws then advance the real game's RNG). False gives it a copy,
        for side-by-side comparisons that must not disturb the original's draws.
        `mirror`: replay Catanatron's records on a CARLA game (see the module docstring);
        False hands the agent a history-free conversion of the current state instead."""
        super().__init__(color)
        self.agent, self.information, self.seed = agent, information, seed
        self.share_rng, self.mirror = share_rng, mirror
        self._game_id = None
        self._mirror = None
        self._applied = 0
        self.road_resyncs = 0
        """Road-length or Longest Road corrections adopted from Catanatron (see `_check`)."""

    def _advance(self, game, seat):
        """Replay Catanatron's new records on the mirror and deliver the seat's events."""
        colors = list(game.state.colors)
        for record in game.state.action_records[self._applied :]:
            events = self._mirror.step(
                carla_action(record.action, colors), record_outcome(record), validate=False
            )
            self.agent.observe(tuple(e.visible_to(seat) for e in events), self._mirror.num_steps)
        self._applied = len(game.state.action_records)

    def _check(self, game, seat):
        truth = carla_game(game, seat, "open").state
        mine = self._mirror.state
        # Catanatron's road lengths depend on its incremental component bookkeeping, which the
        # rules profile follows in most but not all positions: adopt them, and announce a
        # changed Longest Road holder as the event the agent would have seen.
        if truth.road_lengths != mine.road_lengths:
            mine.road_lengths[:] = truth.road_lengths
            self.road_resyncs += 1
        if truth.longest_road_owner != mine.longest_road_owner:
            event = Event(
                EventKind.LONGEST_ROAD_CHANGED,
                truth.longest_road_owner,
                target=mine.longest_road_owner,
            )
            mine.longest_road_owner = truth.longest_road_owner
            self.road_resyncs += 1
            self.agent.observe((event,), self._mirror.num_steps)
        self._mirror._legal = self._mirror._legal_set = None
        differing = [f for f in _CHECKED if getattr(truth, f) != getattr(mine, f)]
        if differing:
            raise RuntimeError(
                f"CARLA and Catanatron disagree after {self._applied} actions on {differing}"
            )

    def decide(self, game, playable_actions):
        colors = list(game.state.colors)
        seat = colors.index(self.color)
        if self._game_id != game.id:
            self._game_id = game.id
            if self.mirror:
                world = carla_game(game, seat, "open")
                self._mirror = Game(world.config, game.seed, world.state.board)
                self._applied = 0
                self.agent.reset(
                    self._mirror, seat, derive_seed(self.seed, "catanatron", game.seed)
                )
            else:
                self.agent.reset(
                    carla_game(game, seat, self.information),
                    seat,
                    derive_seed(self.seed, "catanatron", game.seed),
                )
        if self.mirror:
            self._advance(game, seat)
            self._check(game, seat)
            world = self._mirror
        else:
            world = carla_game(game, seat, self.information)
        actions = [carla_action(a, colors) for a in playable_actions]
        if getattr(self.agent, "chance", None) == "upstream":
            rng = game.state.random
            if not self.share_rng:
                rng = random.Random()
                rng.setstate(game.state.random.getstate())
            self.agent.chance_rng = rng
            # The deck is pre-shuffled and drawn from the end: top card first.
            self.agent.chance_deck = (
                tuple(CARDS.index(c) for c in reversed(game.state.development_listdeck))
                if self.information == "open"
                else None
            )
        chosen = self.agent.act(world, seat, actions)
        if chosen not in actions:
            raise ValueError(f"{self.agent.name} chose {chosen}, which Catanatron does not list")
        return playable_actions[actions.index(chosen)]

    def after(self, game):
        """GameObserver hook: finish and release the agent once Catanatron's game ends."""
        if self._game_id != game.id:
            return
        winner = game.winning_color()
        colors = list(game.state.colors)
        try:
            if self.mirror:  # Deliver the moves after our last decision, the winning one too.
                self._advance(game, colors.index(self.color))
            self.agent.finish(
                {
                    "winner": None if winner is None else colors.index(winner),
                    "turns": game.state.num_turns,
                }
            )
        finally:
            self.agent.close()
