"""Extension compatibility beyond the first reference bots."""

import sys

import pytest

from carla.agents.api import AgentSpec
from carla.agents.heuristic import HeuristicAgent
from carla.agents.registry import make_agent
from carla.engine import Game
from carla.engine.board import Board
from carla.engine.config import EngineConfig
from carla.engine.record import GameRecord
from carla.engine.topology import COAST_EDGES, EDGE_CORNERS, PORT_EDGES
from carla.eval.runner import play_game


def test_alternate_port_frame_roundtrip_and_replay():
    original = Game(seed=19).board
    ports = tuple(EDGE_CORNERS[COAST_EDGES[(COAST_EDGES.index(e) + 1) % 30]] for e in PORT_EDGES)
    board = Board(original.tile_resource, original.tile_number, original.port_types, ports)
    assert Board.from_dict(board.to_dict()) == board
    assert "port_corners" not in original.to_dict()
    game = Game(EngineConfig(max_turns=4), 19, board)
    bot = HeuristicAgent()
    while not game.is_over:
        game.step(bot.act(game, game.decision_player, game.legal_actions()))
    assert GameRecord.from_game(game).replay().state_hash() == game.state_hash()


def test_trade_ratios_follow_custom_port_endpoints():
    from carla.engine.rules import trade_ratios

    original = Game(seed=19).board
    ports = tuple(EDGE_CORNERS[COAST_EDGES[(COAST_EDGES.index(e) + 1) % 30]] for e in PORT_EDGES)
    board = Board(original.tile_resource, original.tile_number, original.port_types, ports)
    for corner in range(54):
        state = Game(seed=19, board=board).state
        state.corner_owner[corner] = 0
        port = board.corner_port[corner]
        expected = [2 if r == port else 3 if port == 5 else 4 for r in range(5)]
        assert trade_ratios(state, 0) == expected


def test_protocol_rejects_unsupported_capabilities_and_cleans_up(tmp_path):
    script = tmp_path / "two_player.py"
    script.write_text(
        'import sys,json\nm=json.loads(sys.stdin.readline())\nprint(json.dumps({"type":"handshake","request":m["request"],"protocol":"carla-agent","version":1,"capabilities":{"players":[2]}}),flush=True)\nsys.stdin.read()\n'
    )
    bot = make_agent(AgentSpec("external", {"command": [sys.executable, str(script)]}))
    with pytest.raises(ValueError, match="four players"):
        play_game([bot, *[HeuristicAgent() for _ in range(3)]], 4)
    assert bot.process is None


def test_protocol_rejects_stale_reply(tmp_path):
    script = tmp_path / "stale.py"
    script.write_text(
        'import sys,json\nm=json.loads(sys.stdin.readline())\nprint(json.dumps({"type":"handshake","request":-1}),flush=True)\n'
    )
    bot = make_agent(AgentSpec("external", {"command": [sys.executable, str(script)]}))
    with pytest.raises(ValueError, match="out-of-sequence"):
        play_game([bot, *[HeuristicAgent() for _ in range(3)]], 4)
    assert bot.process is None


def test_process_that_stops_reading_cannot_block_large_pipe_write(tmp_path):
    script = tmp_path / "stop_reading.py"
    script.write_text(
        "import json,sys,time\n"
        "for _ in range(2):\n"
        " m=json.loads(sys.stdin.readline())\n"
        ' print(json.dumps({"type":m["type"],"request":m["request"],"protocol":"carla-agent","version":1}),flush=True)\n'
        "time.sleep(30)\n"
    )
    bot = make_agent(
        AgentSpec("external", {"command": [sys.executable, str(script)], "timeout": 1.0})
    )
    try:
        bot.reset(Game(seed=4), 0, 4)
        with pytest.raises(TimeoutError, match="stopped reading"):
            bot._send("step", events=["x" * 2_000_000])
    finally:
        bot.close()
    assert bot.process is None and not bot._writer.is_alive()


def test_bot_only_policy_tournament_uses_general_registry():
    pytest.importorskip("torch")
    from carla.rl.policy_tournament import run_policy_tournament

    result = run_policy_tournament(
        ["native-f", "heuristic", "heuristic", "heuristic"], boards=1, seed=42
    )
    assert len(result.records) == 16
    assert all(entry.games == 4 for entry in result.entries)
    assert all(row["decisions"] > 0 for row in result.records)
