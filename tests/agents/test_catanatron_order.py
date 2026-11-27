"""Catanatron's action order, which breaks the upstream ports' exact ties under CARLA."""

import json

import pytest

from carla.agents.catanatron.order import (
    ADJACENCY,
    LAND_ORDER,
    NODES,
    TILE_NODES,
    catanatron_order,
)
from carla.agents.heuristic import HeuristicAgent
from carla.engine import Game
from carla.engine.actions import ActionType
from carla.engine.config import EngineConfig
from carla.engine.state import Phase


def test_dev_cards_come_before_rolling_and_settlements_by_node_id():
    game = Game(seed=3)
    s = game.state
    assert s.phase == Phase.SETUP_SETTLEMENT
    ordered = catanatron_order(s, s.decision_player, list(game.legal_actions()))
    nodes = [NODES[a.corner] for a in ordered]
    assert nodes == sorted(nodes)
    bot = HeuristicAgent()
    for seat in range(4):
        bot.reset(game, seat, 0)
    while s.phase != Phase.PRE_ROLL:
        game.step(bot.act(game, game.decision_player, game.legal_actions()))
    seat = s.decision_player
    s.dev_hands[seat][0] = 1  # a knight bought last turn
    game._legal = game._legal_set = None
    legal = list(game.legal_actions())
    assert [a.type for a in legal] == [ActionType.ROLL_DICE, ActionType.PLAY_DEV_CARD]
    assert [a.type for a in catanatron_order(s, seat, legal)] == [
        ActionType.PLAY_DEV_CARD,
        ActionType.ROLL_DICE,
    ]


def test_tables_match_catanatron():
    pytest.importorskip("catanatron")
    from catanatron.models.board import Board
    from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap

    from carla.agents.catanatron.coordinates import mappings
    from carla.agents.catanatron.host import axial, frame

    template = CatanMap.from_template(BASE_MAP_TEMPLATE)
    _, tile_of, _ = frame()
    assert mappings()[1] == NODES
    assert tuple(tile_of[axial(c)] for c in template.land_tiles) == LAND_ORDER
    by_tile = {tile_of[axial(c)]: tuple(t.nodes.values()) for c, t in template.land_tiles.items()}
    assert tuple(by_tile[t] for t in range(19)) == TILE_NODES
    graph = Board(template).buildable_subgraph
    assert tuple(tuple(graph.adj[n]) for n in range(54)) == ADJACENCY


def test_order_matches_the_rebuilt_catanatron_game():
    """Exact except roads, whose set order is emulated (see order.py)."""
    pytest.importorskip("catanatron")
    from carla.agents.api import AgentView
    from carla.agents.catanatron.codec import moves, reconstruct, representable
    from carla.agents.catanatron.vendor.osp_game import Encoder
    from carla.planning import sample_world

    exact = {"roads": [0, 0], "other": [0, 0]}
    for seed in (0, 1):
        game = Game(seed=seed, config=EngineConfig(max_turns=120))
        bots = [HeuristicAgent() for _ in range(4)]
        for seat, bot in enumerate(bots):
            bot.reset(game, seat, seed)
        while not game.is_over:
            p, legal = game.decision_player, game.legal_actions()
            actions = representable(legal)
            if len(actions) > 1 and game.phase not in (Phase.TRADE_RESPONSE, Phase.TRADE_CONFIRM):
                world = sample_world(AgentView.from_game(game, p), 1)
                native = reconstruct(world, 1)
                by_move = {
                    json.dumps(m, sort_keys=True): a
                    for a, m in zip(actions, moves(world, actions), strict=True)
                }
                encoder, upstream = Encoder(native), []
                for action in native.playable_actions:
                    try:
                        move = json.dumps(encoder.move(action), sort_keys=True)
                    except Exception:  # discards have no single-move encoding
                        continue
                    if (a := by_move.get(move)) is not None and a not in upstream:
                        upstream.append(a)
                ours = [a for a in catanatron_order(game.state, p, actions) if a in upstream]
                kind = exact[
                    "roads" if any(a.type == ActionType.BUILD_ROAD for a in actions) else "other"
                ]
                kind[0] += ours == upstream
                kind[1] += 1
            game.step(bots[p].act(game, p, legal))
    assert exact["other"][0] == exact["other"][1] > 100
    assert exact["roads"][0] >= 0.9 * exact["roads"][1]
