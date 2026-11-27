"""
Catanatron's moves, results, legal moves and states in the Open Seat Protocol (see ../PROTOCOL.md), and back.

Seats are the players' indices in Catanatron's turn order (state.colors). Catanatron discards one card per decision; the
protocol discards the whole set at once, so a discard is enumerated as every set of the right size the hand can pay
(discard_options) and played card by card (discard_actions). The state is the full one (every hand), as a referee or an
open-hand agent sees it.
"""

from collections import Counter
from itertools import product

from catanatron.models.enums import (
    BRICK, DEVELOPMENT_CARDS, KNIGHT, MONOPOLY, ORE, RESOURCES, ROAD_BUILDING, SHEEP, VICTORY_POINT, WHEAT, WOOD,
    YEAR_OF_PLENTY, CITY, SETTLEMENT, Action, ActionPrompt, ActionType)

from . import osp_board

RESOURCE_NAMES = {WOOD: "lumber", BRICK: "brick", SHEEP: "wool", WHEAT: "grain", ORE: "ore"}
RESOURCES_BY_NAME = {name: resource for resource, name in RESOURCE_NAMES.items()}
CARD_NAMES = {KNIGHT: "knight", ROAD_BUILDING: "road_building", YEAR_OF_PLENTY: "year_of_plenty", MONOPOLY: "monopoly",
              VICTORY_POINT: "victory_point"}

# The protocol's prompt by Catanatron's (PLAY_TURN is roll or main, depending on whether the player has rolled)
PROMPTS = {ActionPrompt.BUILD_INITIAL_SETTLEMENT: "setup_settlement", ActionPrompt.BUILD_INITIAL_ROAD: "setup_road",
           ActionPrompt.DISCARD: "discard", ActionPrompt.MOVE_ROBBER: "move_robber",
           ActionPrompt.DECIDE_TRADE: "answer_offer", ActionPrompt.DECIDE_ACCEPTEES: "pick_partner"}


def card_set(resources):
    """A card set {name: count} of an iterable of Catanatron resources, only the names with a count"""
    counts = Counter(resources)
    return {RESOURCE_NAMES[r]: counts[r] for r in RESOURCES if counts[r] > 0}


class Encoder:
    """The protocol's view of one Catanatron game: its keys are fixed once the map is"""

    def __init__(self, game):
        self.game = game
        self.keys = osp_board.node_keys(game.state.board.map)
        self.seats = {color: seat for seat, color in enumerate(game.state.colors)}

    def intersection(self, node_id):
        return [list(hex_) for hex_ in self.keys[node_id]]

    def path(self, edge):
        return [[list(hex_) for hex_ in end] for end in osp_board.path_key(self.keys, edge)]

    def move(self, action):
        """The protocol's move of a Catanatron action (not a discard: see discard_options)"""
        kind, value = action.action_type, action.value
        if kind == ActionType.ROLL:
            return ["roll"]
        if kind == ActionType.END_TURN:
            return ["end_turn"]
        if kind == ActionType.BUILD_SETTLEMENT:
            return ["build_settlement", self.intersection(value)]
        if kind == ActionType.BUILD_CITY:
            return ["build_city", self.intersection(value)]
        if kind == ActionType.BUILD_ROAD:
            return ["build_road", self.path(value)]
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            # The drawn card is the result, not part of the move
            return ["buy_development"]
        if kind == ActionType.PLAY_KNIGHT_CARD:
            return ["play_knight"]
        if kind == ActionType.PLAY_ROAD_BUILDING:
            return ["play_road_building"]
        if kind == ActionType.PLAY_YEAR_OF_PLENTY:
            return ["play_year_of_plenty", sorted(RESOURCE_NAMES[r] for r in value)]
        if kind == ActionType.PLAY_MONOPOLY:
            return ["play_monopoly", RESOURCE_NAMES[value]]
        if kind == ActionType.MOVE_ROBBER:
            coordinate, victim = value
            return ["move_robber", {"hex": list(osp_board.axial(coordinate)), "victim": None if victim is None else self.seats[victim]}]
        if kind == ActionType.MARITIME_TRADE:
            # Up to four cards given (None pads 2:1 and 3:1 trades), the last one asked for
            return ["bank_trade", {"give": card_set(r for r in value[:-1] if r is not None), "get": card_set([value[-1]])}]
        raise ValueError(f"No protocol move for {action}")

    def legal(self, playable_actions):
        """The legal moves in the protocol, in Catanatron's order"""
        return [self.move(action) for action in playable_actions]

    def decode(self, move, playable_actions):
        """The playable action the protocol's move stands for, matched against what is legal (not a discard)"""
        matches = [action for action in playable_actions if self.move(action) == move]
        if len(matches) != 1:
            raise ValueError(f"{move} matches {len(matches)} playable actions")
        return matches[0]

    def result(self, record, monopoly_taken=None):
        """The protocol's result of a Catanatron action record, as the referee knows it"""
        kind = record.action.action_type
        if kind == ActionType.ROLL:
            return {"dice": list(record.result)}
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            return {"card": CARD_NAMES[record.result]}
        if kind == ActionType.MOVE_ROBBER:
            return {"stolen": None if record.result is None else RESOURCE_NAMES[record.result]}
        if kind == ActionType.PLAY_MONOPOLY:
            return {"taken": monopoly_taken}
        return None

    def monopoly_taken(self, action):
        """What a monopoly would take from each other seat, by seat (asked before it is played)"""
        state = self.game.state
        return {str(seat): state.player_state[f"P{seat}_{action.value}_IN_HAND"]
                for seat, color in enumerate(state.colors) if color != action.color}

    def discard_options(self, seat):
        """Every card set the seat can discard now: the right number of cards out of its hand, sorted"""
        state = self.game.state
        count = state.discard_counts[seat]
        hand = [state.player_state[f"P{seat}_{r}_IN_HAND"] for r in RESOURCES]
        options = []
        for amounts in product(*(range(min(h, count) + 1) for h in hand)):
            if sum(amounts) == count:
                options.append(["discard", {RESOURCE_NAMES[r]: n for r, n in zip(RESOURCES, amounts) if n > 0}])
        return options

    def discard_actions(self, move, color):
        """A protocol discard as Catanatron's card-by-card actions"""
        return [Action(color, ActionType.DISCARD_RESOURCE, RESOURCES_BY_NAME[name])
                for name, count in move[1].items() for _ in range(count)]

    def prompt(self):
        state = self.game.state
        # Placing the roads of a Road Building card (Catanatron allows nothing else then, even before the roll)
        if state.is_road_building and state.free_roads_available > 0:
            return "road_building"
        if state.current_prompt == ActionPrompt.PLAY_TURN:
            return "main" if state.player_state[f"P{state.current_player_index}_HAS_ROLLED"] else "roll"
        return PROMPTS[state.current_prompt]

    def state(self):
        """The full state in the protocol's shape (every hand, as the referee sees it)"""
        state = self.game.state
        ps = state.player_state
        players = []
        for seat in range(len(state.colors)):
            key = f"P{seat}"
            players.append({
                "seat": seat,
                "public_vp": ps[f"{key}_VICTORY_POINTS"],
                "vp": ps[f"{key}_ACTUAL_VICTORY_POINTS"],
                "hand": {RESOURCE_NAMES[r]: ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES},
                "development": {CARD_NAMES[c]: ps[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS},
                # The counts every seat sees, also where a view hides the cards
                "hand_size": sum(ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES),
                "development_size": sum(ps[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS),
                "played": {CARD_NAMES[c]: ps[f"{key}_PLAYED_{c}"] for c in DEVELOPMENT_CARDS if c != VICTORY_POINT},
                "longest_road": ps[f"{key}_LONGEST_ROAD_LENGTH"],
                "has_longest_road": ps[f"{key}_HAS_ROAD"],
                "has_largest_army": ps[f"{key}_HAS_ARMY"],
                "pieces_left": {"roads": ps[f"{key}_ROADS_AVAILABLE"], "settlements": ps[f"{key}_SETTLEMENTS_AVAILABLE"],
                                "cities": ps[f"{key}_CITIES_AVAILABLE"]},
            })
        settlements, cities = [], []
        for node_id, (color, building) in state.board.buildings.items():
            (settlements if building == SETTLEMENT else cities if building == CITY else None).append([self.seats[color], self.intersection(node_id)])
        # Catanatron stores every road in both directions
        roads = {}
        for edge, color in state.board.roads.items():
            path = self.path(edge)
            roads[str(path)] = [self.seats[color], path]
        return {
            "prompt": self.prompt(),
            "current": state.current_player_index,
            # Whose turn it is (the current player may be another one, e.g. a player discarding after a 7)
            "on_turn": state.current_turn_index,
            "turn": state.num_turns,
            # The free roads that can still be placed (Catanatron counts 2 when the card is played, even with 1 road left)
            "free_roads": min(state.free_roads_available, ps[f"P{state.current_player_index}_ROADS_AVAILABLE"]) if state.is_road_building else 0,
            "robber": list(osp_board.axial(state.board.robber_coordinate)),
            "buildings": {"settlements": sorted(settlements), "cities": sorted(cities), "roads": sorted(roads.values())},
            "bank": {**{RESOURCE_NAMES[r]: n for r, n in zip(RESOURCES, state.resource_freqdeck)},
                     "development": len(state.development_listdeck)},
            "players": players,
        }
