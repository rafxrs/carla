"""CARLA seats under an OSP v2 referee such as the JSettlers server.

Snapshots preserve foreign port endpoints. Unknown opponent cards are placeholders
for public counts, never sampled facts. Event-driven agents (native JSettlers, PPO
checkpoints `ppo:<path>`) additionally receive a redacted event history in the
no-domestic-trading transfer profile; a PPO belief therefore sees opponents' discard
types as unknown, as JSettlers players do. Search agents that simulate hidden worlds
need an explicit belief conversion and are not supported.
"""

import argparse
import json
import sys

from carla.agents.catanatron.coordinates import corner_keys
from carla.agents.registry import make_agent
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.constants import RESOURCE_NAMES
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.state import GameState, Phase
from carla.engine.topology import EDGE_CORNERS, TILE_AXIAL

CARDS = ("knight", "road_building", "year_of_plenty", "monopoly", "victory_point")
CODES = {"FO": 0, "H": 1, "P": 2, "F": 3, "M": 4, "D": -1}
HARBORS = {"HRL": 0, "HRB": 1, "HRW": 2, "HRG": 3, "HRO": 4, "HR": 5}


def key(value):
    return tuple(sorted(tuple(p) for p in value))


class OSPCodec:
    def __init__(self, board):
        self.nodes = {k: c for c, k in enumerate(corner_keys())}
        self.edges = {tuple(sorted((a, b))): e for e, (a, b) in enumerate(EDGE_CORNERS)}
        tiles = {tuple(t["tile"]): t for t in board["tiles"] if t["tCode"] != "S"}
        ports = tuple(
            tuple(sorted(self.nodes[key(end)] for end in p["path"])) for p in board["harbors"]
        )
        self.board = Board(
            tuple(CODES[tiles[p]["tCode"]] for p in TILE_AXIAL),
            tuple(tiles[p]["rNumber"] or 0 for p in TILE_AXIAL),
            tuple(HARBORS[p["hCode"]] for p in board["harbors"]),
            ports,
        )

    def decode(self, move):
        kind = move[0]
        value = move[1] if len(move) > 1 else None
        simple = {
            "roll": ActionType.ROLL_DICE,
            "end_turn": ActionType.END_TURN,
            "buy_development": ActionType.BUY_DEV_CARD,
            "cancel": ActionType.CANCEL_TRADE,
        }
        if kind in simple:
            return Action(simple[kind])
        if kind in ("build_settlement", "build_city"):
            return Action(
                ActionType.BUILD_CITY if kind == "build_city" else ActionType.BUILD_SETTLEMENT,
                corner=self.nodes[key(value)],
            )
        if kind == "build_road":
            ends = tuple(sorted(self.nodes[key(end)] for end in value))
            return Action(ActionType.BUILD_ROAD, edge=self.edges[ends])
        if kind == "move_robber":
            return Action(
                ActionType.MOVE_ROBBER,
                tile=TILE_AXIAL.index(tuple(value["hex"])),
                player=-1 if value["victim"] is None else value["victim"],
            )
        if kind == "bank_trade":
            return Action(
                ActionType.MARITIME_TRADE,
                resource=RESOURCE_NAMES.index(next(iter(value["give"]))),
                resource2=RESOURCE_NAMES.index(next(iter(value["get"]))),
            )
        if kind == "respond":
            return Action(ActionType.ACCEPT_TRADE if value["accept"] else ActionType.REJECT_TRADE)
        if kind == "confirm":
            return Action(ActionType.CONFIRM_TRADE, player=value["partner"])
        if kind == "discard":
            return Action(ActionType.DISCARD, resource=RESOURCE_NAMES.index(next(iter(value))))
        if kind == "play_knight":
            return Action(ActionType.PLAY_DEV_CARD, dev=0)
        if kind == "play_road_building":
            return Action(ActionType.PLAY_DEV_CARD, dev=1)
        if kind == "play_year_of_plenty":
            a, b = sorted(RESOURCE_NAMES.index(r) for r in value)
            return Action(ActionType.PLAY_DEV_CARD, dev=2, resource=a, resource2=b)
        if kind == "play_monopoly":
            return Action(ActionType.PLAY_DEV_CARD, dev=3, resource=RESOURCE_NAMES.index(value))
        raise ValueError(f"unsupported OSP move {kind}")

    def game(self, data, seat):
        config = EngineConfig(allow_domestic_trade=True)
        state = GameState.initial(self.board, config)
        prompt = data["prompt"]
        state.phase = {
            "setup_settlement": Phase.SETUP_SETTLEMENT,
            "setup_road": Phase.SETUP_ROAD,
            "roll": Phase.PRE_ROLL,
            "main": Phase.MAIN,
            "discard": Phase.DISCARD,
            "move_robber": Phase.MOVE_ROBBER,
            "road_building": Phase.ROAD_BUILDING,
            "answer_offer": Phase.TRADE_RESPONSE,
            "pick_partner": Phase.TRADE_CONFIRM,
        }[prompt]
        state.current_player = data.get("on_turn", data["current"])
        state.decision_player = data["current"]
        state.turn = data.get("turn", 0)
        state.has_rolled = prompt not in ("roll", "setup_settlement", "setup_road")
        state.free_roads = data.get("free_roads", 0)
        state.robber_tile = TILE_AXIAL.index(tuple(data["robber"]))
        for typ, mult in (("settlements", 1), ("cities", 2)):
            for owner, node in data["buildings"][typ]:
                corner = self.nodes[key(node)]
                state.corner_owner[corner] = owner
                state.corner_building[corner] = mult
        for owner, path in data["buildings"]["roads"]:
            edge = self.edges[tuple(sorted(self.nodes[key(end)] for end in path))]
            state.edge_owner[edge] = owner
        for p in data["players"]:
            index = p["seat"]
            state.hands[index] = (
                [p["hand"].get(r, 0) for r in RESOURCE_NAMES]
                if p["hand"] is not None
                else [p["hand_size"], 0, 0, 0, 0]
            )
            state.dev_hands[index] = (
                [p["development"].get(c, 0) for c in CARDS]
                if p["development"] is not None
                else [p["development_size"], 0, 0, 0, 0]
            )
            state.dev_played[index] = [p["played"].get(c, 0) for c in CARDS]
            state.roads_left[index] = p["pieces_left"]["roads"]
            state.settlements_left[index] = p["pieces_left"]["settlements"]
            state.cities_left[index] = p["pieces_left"]["cities"]
            state.road_lengths[index] = p["longest_road"]
            if p["has_longest_road"]:
                state.longest_road_owner = index
            if p["has_largest_army"]:
                state.largest_army_owner = index
        state.bank = [data["bank"][r] for r in RESOURCE_NAMES]
        state.dev_deck = [data["bank"]["development"], 0, 0, 0, 0]
        offer = data.get("offer")
        if offer:
            state.pending_trade = TradeOffer(
                tuple(offer["give"].get(r, 0) for r in RESOURCE_NAMES),
                tuple(offer["get"].get(r, 0) for r in RESOURCE_NAMES),
            )
        game = Game.__new__(Game)
        game.seed = 0
        game.state = state
        game.rng = Rng(0)
        game.history = []
        game._legal = game._legal_set = None
        return game

    def choose(self, bot, data, seat, legal):
        game = self.game(data, seat)
        if data["prompt"] == "discard":
            # Whole-hand OSP discard becomes a sequence of CARLA discard decisions.
            needed = sum(legal[0][1].values())
            game.state.discards_pending[seat] = needed
            counts = {}
            for _ in range(needed):
                actions = [
                    Action(ActionType.DISCARD, resource=r)
                    for r, n in enumerate(game.state.hands[seat])
                    if n
                ]
                chosen = bot.act(game, seat, actions)
                if chosen not in actions:
                    raise ValueError(f"{bot.name} discarded a card it does not hold: {chosen}")
                name = RESOURCE_NAMES[chosen.resource]
                counts[name] = counts.get(name, 0) + 1
                game.state.hands[seat][chosen.resource] -= 1
                game.state.bank[chosen.resource] += 1
                game.state.discards_pending[seat] -= 1
                # The next card is a new decision: its legal set (and masks) must be recomputed.
                game._legal = game._legal_set = None
            for move in legal:
                if move[1] == counts:
                    return move
            raise ValueError("compound discard not listed by referee")
        actions = [self.decode(m) for m in legal]
        chosen = bot.act(game, seat, actions)
        if chosen not in actions:
            raise ValueError("CARLA bot selected an unsupported external action")
        return legal[actions.index(chosen)]


SUPPORTED = (
    "heuristic",
    "random",
    "random_flat",
    "jsettlers-fast-native",
    "jsettlers-smart-native",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        default="heuristic",
        help="heuristic, random, random_flat, jsettlers-{fast,smart}-native or ppo:<checkpoint>",
    )
    args = parser.parse_args()
    if args.agent not in SUPPORTED and not args.agent.startswith("ppo:"):
        parser.error(f"unsupported agent under an OSP referee: {args.agent}")
    bot = make_agent(args.agent)
    if args.agent.startswith("ppo:"):
        bot.allow_trading_override = True  # Engine trading knobs; the referee forbids trades.
    native = args.agent.endswith("-native") or args.agent.startswith("ppo:")
    codec = None
    seat = 0
    for line in sys.stdin:
        message = json.loads(line)
        kind = message["type"]
        try:
            if kind == "handshake":
                reply = {
                    "type": "handshake",
                    "protocol": "osp",
                    "version": 2,
                    "name": f"carla-{args.agent}",
                    "rulesets": ["standard"],
                    "needs": "view",
                    "observe": native,
                    "lean": False,
                }
            elif kind == "before":
                seat = message["seat"]
                if native:
                    if message["players"] != 4:
                        raise ValueError("native JSettlers transfer requires four players")
                    from carla.agents.jsettlers_native.osp import NativeOSPCodec

                    codec = NativeOSPCodec(message["board"], message["ruleset"], seat)
                else:
                    codec = OSPCodec(message["board"])
                bot.reset(codec.game(message["state"], seat), seat, message["seed"])
                reply = {"type": "ready", "game_id": message["game_id"]}
            elif kind == "decide":
                reply = {
                    "type": "decide",
                    "game_id": message["game_id"],
                    "n": message["n"],
                    "action": codec.choose(bot, message["state"], seat, message["legal"]),
                }
            elif kind == "step" and native:
                codec.observe(bot, message)
                continue
            elif kind in ("after", "step", "failure"):
                continue
            else:
                raise ValueError(f"unsupported message {kind}")
        except Exception as error:
            reply = {
                "type": "failure",
                "code": "unsupported",
                "message": str(error),
                "detail": str(error),
            }
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
