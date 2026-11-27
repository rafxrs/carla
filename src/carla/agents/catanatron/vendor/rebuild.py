"""
Catanatron's bots as an Open Seat Agent (see ../PROTOCOL.md): they play seats for a referee of another engine, e.g.
Lanzenbrecher's Arena or simulator. For every decision the agent rebuilds a Catanatron game from the referee's state (rebuild)
and lets the bot choose among the moves the referee listed, so the referee's rules decide what is legal and its state
is always the one played on; nothing is carried from move to move, and no step is needed.

The bot needs every hand: the agent asks for a world drawn by the referee from its seat's view ("world", a fair bot) or
for the true hands ("truth", open-hand, as in Catanatron's own games). Where the referee's rules allow more than
Catanatron's, the bot picks among the moves both allow; a robber move without a victim (Catanatron's choice when nobody
there has cards) takes a victim the referee lists on that hex. Offers between players are declined and none are made.
If a game can't be rebuilt, the first listed move is played and the failure is logged on standard error.

Usage: python open_seat_agent.py --bot AB:2 [--needs world|truth]
Bots by Catanatron's registry codes (R random, W weighted random, VP, G greedy playouts, AB:N alpha-beta of depth N, ...).
One JSON object per line on standard input and output, logs and a statistics line per game on standard error.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict

import catanatron.players  # noqa: F401 (registers the bots)
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.board import Board
from catanatron.models.enums import (CITY, DEVELOPMENT_CARDS, KNIGHT, MONOPOLY, RESOURCES, ROAD, ROAD_BUILDING, SETTLEMENT,
                                     VICTORY_POINT, YEAR_OF_PLENTY, ActionPrompt)
from catanatron.registry import REGISTRY
from catanatron.state import PLAYER_INITIAL_STATE, State

from . import osp_board
from .osp_game import CARD_NAMES, RESOURCE_NAMES, Encoder, card_set
from catanatron.models.player import Color
COLORS = tuple(Color)[:4]

# The development cards of a classic game
DEVELOPMENT_DECK = {KNIGHT: 14, VICTORY_POINT: 5, ROAD_BUILDING: 2, YEAR_OF_PLENTY: 2, MONOPOLY: 2}
# The move that plays each card that can be played
PLAY_MOVES = {KNIGHT: "play_knight", ROAD_BUILDING: "play_road_building", YEAR_OF_PLENTY: "play_year_of_plenty", MONOPOLY: "play_monopoly"}
PROMPTS = {"setup_settlement": ActionPrompt.BUILD_INITIAL_SETTLEMENT, "setup_road": ActionPrompt.BUILD_INITIAL_ROAD,
           "discard": ActionPrompt.DISCARD, "move_robber": ActionPrompt.MOVE_ROBBER, "roll": ActionPrompt.PLAY_TURN,
           "main": ActionPrompt.PLAY_TURN, "road_building": ActionPrompt.PLAY_TURN}


def canonical(node):
    return json.dumps(node, sort_keys=True)


def listed(move, legal):
    """The referee's entry for a move, or None: the same move, or for a robber move without a victim one on the same hex"""
    wanted = canonical(move)
    for entry in legal:
        if canonical(entry) == wanted:
            return entry
    if move[0] == "move_robber" and move[1]["victim"] is None:
        for entry in legal:
            if entry[0] == "move_robber" and entry[1]["hex"] == move[1]["hex"]:
                return entry
    return None


def _key(hexes):
    return tuple(sorted(tuple(h) for h in hexes))


def _settle_order(settlements, roads):
    """Settlements with a road next to them first: the one placed last in the setup has none yet"""
    road_nodes = defaultdict(set)
    for color, edge in roads:
        road_nodes[color].update(edge)
    return sorted(settlements, key=lambda s: s[1] not in road_nodes[s[0]])


def rebuild(board, state, legal, ruleset, players=None):
    """
    A Catanatron game in the referee's state: its map, buildings (placed on a fresh board, so the board's own caches
    follow), hands, development cards, bank and deck, whose turn and what the player has to do. What the referee's state
    doesn't say is taken from its legal moves (which development cards can be played this turn, whether the player has
    rolled) or left at Catanatron's start values; the deck holds the cards nobody holds or played, shuffled.
    """
    catan_map = osp_board.catan_map(board)
    keys = osp_board.node_keys(catan_map)
    node_of = {key: node_id for node_id, key in keys.items()}
    edge_of = {}
    for tile in catan_map.land_tiles.values():
        for edge in tile.edges.values():
            edge_of[osp_board.path_key(keys, edge)] = edge

    seats = len(state["players"])
    colors = tuple(COLORS[:seats])
    players = players or [REGISTRY.build("R", color) for color in colors]

    s = State([], None, initialize=False)
    s.random = random.Random()
    s.colors = colors
    s.color_to_index = {color: i for i, color in enumerate(colors)}
    s.players = list(players)
    s.discard_limit = ruleset.get("discard_limit", 7)
    s.friendly_robber = False

    # Buildings by color, settlements and cities by node, roads by edge
    settlements, cities, roads = [], [], []
    for seat, key in state["buildings"]["settlements"]:
        settlements.append((colors[seat], node_of[_key(key)]))
    for seat, key in state["buildings"]["cities"]:
        cities.append((colors[seat], node_of[_key(key)]))
    for seat, path in state["buildings"]["roads"]:
        roads.append((colors[seat], edge_of[tuple(sorted(_key(end) for end in path))]))

    # A fresh board: every settlement and city of every color as a settlement first (no road needed, as in the setup),
    # then each color's roads as they connect to its network, which ends at other colors' buildings as Catanatron's
    # does; then the cities
    b = Board(catan_map)
    built = defaultdict(lambda: defaultdict(list))
    for color in colors:
        for c, node in _settle_order([x for x in settlements + cities if x[0] == color], roads):
            b.build_settlement(c, node, initial_build_phase=True)
            if (c, node) in settlements:
                built[c][SETTLEMENT].append(node)
    for color in colors:
        pending = [edge for c, edge in roads if c == color]
        while pending:
            buildable = set(b.buildable_edges(color))
            ready = [e for e in pending if e in buildable or (e[1], e[0]) in buildable]
            if not ready:
                # Roads beyond another color's building (it was built on the road later): placed as they are, each piece
                # a network of its own, walked as Catanatron walks a road cut by a settlement
                for e in pending:
                    b.roads[e] = color
                    b.roads[(e[1], e[0])] = color
                    built[color][ROAD].append(e)
                covered = set().union(*b.connected_components[color]) if b.connected_components[color] else set()
                for e in pending:
                    for end in e:
                        if end not in covered and not b.is_enemy_node(end, color):
                            component = b.dfs_walk(end, color)
                            b.connected_components[color].append(component)
                            covered |= component
                b.buildable_edges_cache = {}
                break
            for e in ready:
                b.build_road(color, e)
                built[color][ROAD].append(e)
                pending.remove(e)
    for color, node in cities:
        b.build_city(color, node)
        built[color][CITY].append(node)
    b.robber_coordinate = osp_board.cube(state["robber"])
    # Longest Road as the referee awards it
    b.road_lengths = defaultdict(int, {colors[p["seat"]]: p["longest_road"] for p in state["players"]})
    holders = [p for p in state["players"] if p["has_longest_road"]]
    b.road_color = colors[holders[0]["seat"]] if holders else None
    b.road_length = holders[0]["longest_road"] if holders else 0
    s.board = b
    s.buildings_by_color = {color: built[color] for color in colors}

    # Which cards the player on turn may play: those the referee lists a move for
    legal_types = {move[0] for move in legal}
    prompt = state["prompt"]
    current = state["current"]
    on_turn = state.get("on_turn", current)
    s.player_state = {}
    for p in state["players"]:
        seat, prefix = p["seat"], f"P{p['seat']}_"
        for key, value in PLAYER_INITIAL_STATE.items():
            s.player_state[prefix + key] = value
        ps = lambda key, value: s.player_state.__setitem__(prefix + key, value)  # noqa: E731
        ps("VICTORY_POINTS", p["public_vp"])
        ps("ACTUAL_VICTORY_POINTS", p["vp"] if p.get("vp") is not None else p["public_vp"])
        ps("ROADS_AVAILABLE", p["pieces_left"]["roads"])
        ps("SETTLEMENTS_AVAILABLE", p["pieces_left"]["settlements"])
        ps("CITIES_AVAILABLE", p["pieces_left"]["cities"])
        ps("HAS_ROAD", p["has_longest_road"])
        ps("HAS_ARMY", p["has_largest_army"])
        ps("LONGEST_ROAD_LENGTH", p["longest_road"])
        for resource in RESOURCES:
            ps(f"{resource}_IN_HAND", (p["hand"] or {}).get(RESOURCE_NAMES[resource], 0))
        for card in DEVELOPMENT_CARDS:
            ps(f"{card}_IN_HAND", (p["development"] or {}).get(CARD_NAMES[card], 0))
            if card != VICTORY_POINT:
                ps(f"PLAYED_{card}", p["played"].get(CARD_NAMES[card], 0))
        for card, move in PLAY_MOVES.items():
            held = (p["development"] or {}).get(CARD_NAMES[card], 0) > 0
            # On its own turn the listed moves say what may be played; the others own theirs by their next turn
            ps(f"{card}_OWNED_AT_START", move in legal_types if seat == on_turn and prompt in ("roll", "main") else held)
        if seat == on_turn:
            ps("HAS_ROLLED", prompt in ("main", "discard", "move_robber") or (prompt == "road_building" and "roll" not in legal_types))

    # An unlimited bank (JSettlers') counts below 0 once the hands hold more than 19 of a resource: an empty supply here
    s.resource_freqdeck = [max(0, state["bank"][RESOURCE_NAMES[r]]) for r in RESOURCES]
    # The deck: the cards nobody holds or played, as many as the referee's bank has (Lanzenbrecher reports only knights played)
    remaining = []
    for card, total in DEVELOPMENT_DECK.items():
        used = sum((p["development"] or {}).get(CARD_NAMES[card], 0) + p["played"].get(CARD_NAMES[card], 0) for p in state["players"])
        remaining += [card] * max(0, total - used)
    s.random.shuffle(remaining)
    size = state["bank"]["development"]
    s.development_listdeck = (remaining + [KNIGHT] * size)[:size]

    s.action_records = []
    s.num_turns = state.get("turn", 0)
    s.current_player_index = current
    s.current_turn_index = on_turn
    s.current_prompt = PROMPTS[prompt]
    s.is_initial_build_phase = prompt in ("setup_settlement", "setup_road")
    s.is_discarding = prompt == "discard"
    s.discard_counts = [0] * seats
    if prompt == "discard":
        s.discard_counts[current] = sum(legal[0][1].values())
    s.is_moving_knight = False
    s.is_road_building = prompt == "road_building"
    s.free_roads_available = state.get("free_roads", 0) if prompt == "road_building" else 0
    s.is_resolving_trade = False
    s.current_trade = (0,) * 11
    s.acceptees = tuple(False for _ in colors)

    game = Game(players=[], initialize=False)
    game.id = "open-seat"
    game.seed = 0
    game.vps_to_win = ruleset.get("victory_points", 10)
    game.friendly_robber = False
    game.random = s.random
    game.state = s
    game.playable_actions = generate_playable_actions(s)
    return game

