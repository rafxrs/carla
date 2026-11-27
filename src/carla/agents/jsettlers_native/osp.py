"""Observable OSP history for native agents under the JSettlers referee.

Initial transfer profile is the four-player base game with domestic trading off.
Unknown steals and opponents' discard TYPES remain hidden. Referee transitions
are never executed by CARLA, and legal moves come directly from the referee.
"""

from carla.agents.osp import CARDS, OSPCodec
from carla.engine.config import EngineConfig
from carla.engine.constants import RESOURCE_NAMES
from carla.engine.events import Event, EventKind
from carla.engine.game import Game
from carla.engine.state import Phase
from carla.engine.topology import CORNER_TILES


class PublicGameSnapshot(Game):
    """Query-compatible snapshot with a real event count and no referee step."""

    __slots__ = ("visible_step",)

    @property
    def num_steps(self):
        return self.visible_step

    def step(self, *_args, **_kwargs):
        raise RuntimeError("the external referee owns transitions")

    def clone(self):
        raise RuntimeError("an external observation is not a hidden-world simulator")


class NativeOSPCodec(OSPCodec):
    def __init__(self, board, ruleset, seat):
        super().__init__(board)
        if ruleset.get("player_trading") != "none":
            raise ValueError(
                "native JSettlers OSP transfer currently requires domestic trading off"
            )
        self.seat = seat
        self.last_step, self.turn, self.setup_roads = -1, 0, 0
        self.settlements = [0] * 4
        self.last_corner = [-1] * 4
        self.new_cards = [[0] * 5 for _ in range(4)]
        self.played = [False] * 4
        # Public buildings and robber, for production: the codec serves any event-driven
        # agent (native JSettlers, PPO checkpoints), not only one with its own board copy.
        self.corner_owner, self.corner_building = [-1] * 54, [0] * 54
        self.robber_tile = self.board.desert_tile

    def game(self, data, seat):
        original = super().game(data, seat)
        game = PublicGameSnapshot.__new__(PublicGameSnapshot)
        for name in Game.__slots__:
            setattr(game, name, getattr(original, name))
        game.visible_step = self.last_step + 1
        state = game.state
        state.config = EngineConfig(vp_to_win=data.get("vp_to_win", 10), allow_domestic_trade=False)
        state.turn = self.turn
        state.setup_step = min(self.setup_roads, 8)
        if state.phase == Phase.SETUP_ROAD:
            state.setup_corner = self.last_corner[seat]
            if state.setup_corner < 0:
                raise ValueError("initial-road prompt lacks its chronological settlement event")
        state.dev_bought_this_turn = self.new_cards[state.current_player][:]
        state.dev_played_this_turn = self.played[state.current_player]
        # The referee's snapshot is authoritative for the public board used for production.
        self.corner_owner, self.corner_building = state.corner_owner[:], state.corner_building[:]
        self.robber_tile = state.robber_tile
        return game

    def observe(self, bot, message):
        index = message["n"]
        if index == self.last_step:
            return
        if index != self.last_step + 1:
            raise ValueError("OSP event history has a gap")
        player = message["seat"]
        move = message["action"]
        kind, result = move[0], message.get("result") or {}
        action = self.decode(move)
        events = []
        if kind == "build_settlement":
            self.corner_owner[action.corner], self.corner_building[action.corner] = player, 1
            events.append(Event(EventKind.SETTLEMENT_BUILT, player=player, location=action.corner))
            self.settlements[player] += 1
            self.last_corner[player] = action.corner
            if self.setup_roads < 8 and self.settlements[player] == 2:
                counts = [0] * 5
                for tile in CORNER_TILES[action.corner]:
                    resource = self.board.tile_resource[tile]
                    if resource >= 0:
                        counts[resource] += 1
                events.append(
                    Event(EventKind.STARTING_RESOURCES, player=player, counts=tuple(counts))
                )
        elif kind == "build_city":
            events.append(Event(EventKind.CITY_BUILT, player=player, location=action.corner))
            self.corner_building[action.corner] = 2
        elif kind == "build_road":
            events.append(Event(EventKind.ROAD_BUILT, player=player, location=action.edge))
            self.setup_roads += int(self.setup_roads < 8)
        elif kind == "roll":
            dice = result["dice"]
            total = sum(dice)
            events.append(
                Event(EventKind.DICE_ROLLED, player=player, value=total, counts=tuple(dice))
            )
            production = [[0] * 5 for _ in range(4)]
            if total != 7:
                for corner, owner in enumerate(self.corner_owner):
                    if owner < 0:
                        continue
                    for tile in CORNER_TILES[corner]:
                        resource = self.board.tile_resource[tile]
                        if (
                            tile != self.robber_tile
                            and resource >= 0
                            and self.board.tile_number[tile] == total
                        ):
                            production[owner][resource] += self.corner_building[corner]
            for owner, counts in enumerate(production):
                if any(counts):
                    events.append(
                        Event(EventKind.RESOURCES_PRODUCED, player=owner, counts=tuple(counts))
                    )
        elif kind == "end_turn":
            events.append(Event(EventKind.TURN_ENDED, player=player))
            self.turn += 1
            self.new_cards[player] = [0] * 5
            self.played[player] = False
        elif kind == "bank_trade":
            counts = tuple(
                move[1]["get"].get(r, 0) - move[1]["give"].get(r, 0) for r in RESOURCE_NAMES
            )
            events.append(Event(EventKind.MARITIME_TRADED, player=player, counts=counts))
        elif kind == "buy_development":
            card = result.get("card")
            if player == self.seat and card is None:
                raise ValueError("referee omitted our private development-card type")
            value = CARDS.index(card) if card is not None else -1
            events.append(Event(EventKind.DEV_CARD_BOUGHT, player=player, value=value))
            if value >= 0:
                self.new_cards[player][value] += 1
        elif kind == "discard":
            if player == self.seat:
                resources = [
                    RESOURCE_NAMES.index(r) for r, count in move[1].items() for _ in range(count)
                ]
            else:
                # JSettlers opponents' discarded card types are private even
                # if an older OSP logger happens to include them in its move.
                resources = [-1] * sum(move[1].values())
            events.extend(Event(EventKind.DISCARDED, player=player, value=r) for r in resources)
        elif kind == "move_robber":
            events.append(Event(EventKind.ROBBER_MOVED, player=player, location=action.tile))
            self.robber_tile = action.tile
            if action.player >= 0:
                resource = result.get("stolen")
                value = RESOURCE_NAMES.index(resource) if resource in RESOURCE_NAMES else -1
                events.append(
                    Event(
                        EventKind.RESOURCE_STOLEN, player=player, target=action.player, value=value
                    )
                )
        elif kind.startswith("play_"):
            events.append(Event(EventKind.DEV_CARD_PLAYED, player=player, value=action.dev))
            self.played[player] = True
            if kind == "play_year_of_plenty":
                counts = [0] * 5
                counts[action.resource] += 1
                counts[action.resource2] += 1
                events.append(
                    Event(EventKind.YEAR_OF_PLENTY_TAKEN, player=player, counts=tuple(counts))
                )
            elif kind == "play_monopoly":
                counts = tuple(result["taken"].get(str(p), 0) for p in range(4))
                events.append(
                    Event(
                        EventKind.MONOPOLY_TAKEN,
                        player=player,
                        value=action.resource,
                        counts=counts,
                    )
                )
        else:
            raise ValueError(f"unsupported native OSP history action {kind}")
        bot.observe(events, index + 1)
        self.last_step = index
