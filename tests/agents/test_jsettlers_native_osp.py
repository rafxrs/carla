"""History conversion for native JSettlers under the original referee."""

from types import SimpleNamespace

import pytest

from carla.agents.catanatron.coordinates import corner_keys
from carla.agents.jsettlers_native.osp import NativeOSPCodec
from carla.agents.osp import CODES, HARBORS
from carla.engine import Game
from carla.engine.constants import RESOURCE_NAMES
from carla.engine.events import EventKind
from carla.engine.topology import TILE_AXIAL


def board_json(board):
    keys = corner_keys()
    tiles, ports = {v: k for k, v in CODES.items()}, {v: k for k, v in HARBORS.items()}
    return {
        "tiles": [
            {"tile": tile, "tCode": tiles[r], "rNumber": n}
            for tile, r, n in zip(TILE_AXIAL, board.tile_resource, board.tile_number, strict=True)
        ],
        "harbors": [
            {"path": [keys[a], keys[b]], "hCode": ports[kind]}
            for kind, (a, b) in zip(board.port_types, board.port_corners, strict=True)
        ],
    }


def observer(state):
    calls = []
    return (
        SimpleNamespace(state=state, observe=lambda events, step: calls.append((events, step))),
        calls,
    )


def test_opponents_discard_types_are_hidden_in_foreign_history():
    game = Game(seed=41)
    outputs = []
    for cards in ({"wool": 2, "grain": 1}, {"ore": 3}):
        codec = NativeOSPCodec(board_json(game.board), {"player_trading": "none"}, 0)
        bot, calls = observer(game.state)
        codec.observe(bot, {"n": 0, "seat": 1, "action": ["discard", cards], "result": None})
        outputs.append(calls)
    assert outputs[0] == outputs[1]
    events, step = outputs[0][0]
    assert step == 1 and len(events) == 3
    assert all(e.kind == EventKind.DISCARDED and e.value == -1 for e in events)


def test_foreign_own_development_type_is_required_and_age_is_tracked():
    game = Game(seed=41)
    codec = NativeOSPCodec(board_json(game.board), {"player_trading": "none"}, 0)
    bot, _ = observer(game.state)
    with pytest.raises(ValueError, match="private development"):
        codec.observe(
            bot, {"n": 0, "seat": 0, "action": ["buy_development"], "result": {"card": None}}
        )
    codec.observe(
        bot, {"n": 0, "seat": 0, "action": ["buy_development"], "result": {"card": "knight"}}
    )
    assert codec.new_cards[0] == [1, 0, 0, 0, 0]
    codec.observe(bot, {"n": 1, "seat": 0, "action": ["end_turn"], "result": None})
    assert codec.new_cards[0] == [0] * 5 and codec.turn == 1
    with pytest.raises(ValueError, match="gap"):
        codec.observe(bot, {"n": 3, "seat": 1, "action": ["end_turn"], "result": None})


def test_foreign_roll_production_is_computed_from_public_board():
    from carla.engine.topology import TILE_CORNERS

    game = Game(seed=41)
    tile = next(t for t, r in enumerate(game.board.tile_resource) if r >= 0)
    corner = TILE_CORNERS[tile][0]
    game.state.corner_owner[corner], game.state.corner_building[corner] = 1, 2
    game.state.robber_tile = game.board.desert_tile
    codec = NativeOSPCodec(board_json(game.board), {"player_trading": "none"}, 0)
    # The codec keeps its own public board (from events and decision snapshots).
    codec.corner_owner, codec.corner_building = (
        game.state.corner_owner[:],
        game.state.corner_building[:],
    )
    codec.robber_tile = game.state.robber_tile
    bot, calls = observer(game.state)
    number = game.board.tile_number[tile]
    dice = [number // 2, number - number // 2]
    codec.observe(bot, {"n": 0, "seat": 0, "action": ["roll"], "result": {"dice": dice}})
    production = [e for e in calls[0][0] if e.kind == EventKind.RESOURCES_PRODUCED]
    assert len(production) == 1 and production[0].player == 1
    assert production[0].counts[game.board.tile_resource[tile]] >= 2
    assert len(production[0].counts) == len(RESOURCE_NAMES)


def test_native_foreign_trading_profile_is_rejected_explicitly():
    with pytest.raises(ValueError, match="domestic trading off"):
        NativeOSPCodec(board_json(Game(seed=41).board), {"player_trading": "first_accept"}, 0)


def test_foreign_snapshot_preserves_event_clock_age_and_hidden_information():
    from carla.agents.api import AgentView

    original = Game(seed=41)
    codec = NativeOSPCodec(board_json(original.board), {"player_trading": "none"}, 0)
    codec.last_step, codec.turn = 12, 7
    codec.new_cards[0][0] = 1
    codec.played[0] = True
    players = [
        {
            "seat": p,
            "hand_size": 3,
            "development_size": 1,
            "hand": {"lumber": 3} if p == 0 else None,
            "development": {"knight": 1} if p == 0 else None,
            "played": {"knight": 0},
            "longest_road": 0,
            "has_longest_road": False,
            "has_largest_army": False,
            "pieces_left": {"roads": 15, "settlements": 5, "cities": 4},
        }
        for p in range(4)
    ]
    data = {
        "prompt": "roll",
        "current": 0,
        "on_turn": 0,
        "players": players,
        "vp_to_win": 10,
        "free_roads": 0,
        "robber": TILE_AXIAL[original.board.desert_tile],
        "buildings": {"settlements": [], "cities": [], "roads": []},
        "bank": {**dict.fromkeys(RESOURCE_NAMES, 19), "development": 25},
    }
    snapshot = codec.game(data, 0)
    view = AgentView.from_game(snapshot, 0)
    assert view.step == 13 and view.state["turn"] == 7
    assert view.state["dev_bought_this_turn"] == [1, 0, 0, 0, 0]
    assert view.state["dev_played_this_turn"]
    assert view.state["hands"][1:] == [None] * 3
    assert view.state["dev_hands"][1:] == [None] * 3
    with pytest.raises(RuntimeError, match="external referee"):
        snapshot.step(None)
    with pytest.raises(RuntimeError, match="hidden-world"):
        snapshot.clone()


def test_compound_discard_recomputes_legal_moves_between_cards():
    """OSP discards a whole set at once; each CARLA card choice is a new decision whose legal
    moves (the masks a policy reads) exclude cards already discarded."""
    from carla.agents.base import Agent
    from carla.agents.osp import OSPCodec

    game = Game(seed=41)
    names = ("lumber", "brick", "wool", "grain", "ore")
    hand = {"lumber": 1, "brick": 0, "wool": 0, "grain": 3, "ore": 0}
    players = [
        {
            "seat": p,
            "hand": dict(hand) if p == 0 else None,
            "hand_size": 4,
            "development": (
                dict.fromkeys(
                    ("knight", "road_building", "year_of_plenty", "monopoly", "victory_point"), 0
                )
                if p == 0
                else None
            ),
            "development_size": 0,
            "played": {},
            "pieces_left": {"roads": 15, "settlements": 5, "cities": 4},
            "longest_road": 0,
            "has_longest_road": False,
            "has_largest_army": False,
        }
        for p in range(4)
    ]
    data = {
        "prompt": "discard",
        "current": 0,
        "on_turn": 1,
        "robber": list(TILE_AXIAL[game.board.desert_tile]),
        "buildings": {"settlements": [], "cities": [], "roads": []},
        "players": players,
        "bank": {**dict.fromkeys(names, 15), "development": 25},
    }

    class FirstLegal(Agent):
        name = "first-legal"

        def act(self, game, seat, legal):
            return game.legal_actions()[0]  # masks come from the game, as a policy's do

    legal = [["discard", {"lumber": 1, "grain": 1}], ["discard", {"grain": 2}]]
    assert OSPCodec(board_json(game.board)).choose(FirstLegal(), data, 0, legal) == legal[0]
