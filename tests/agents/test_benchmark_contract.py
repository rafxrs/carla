"""Privacy, real process lifecycle, optional reference parity and complete games."""

import sys
from pathlib import Path

import pytest

from carla.agents.api import AgentSpec, AgentView
from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.registry import make_agent
from carla.engine import Game
from carla.engine.config import EngineConfig
from carla.eval.runner import play_game
from carla.planning import sample_world
from carla.planning.chance import outcomes


def test_view_and_world_ignore_hidden_truth():
    game = Game(seed=12)
    bot = HeuristicAgent()
    for _ in range(45):
        game.step(bot.act(game, game.decision_player, game.legal_actions()))
    seat = game.decision_player
    view = AgentView.from_game(game, seat)
    assert "seed" not in view.state and "rng" not in view.state
    other = (seat + 1) % 4
    game.state.hands[other] = list(reversed(game.state.hands[other]))
    game.state.dev_deck = list(reversed(game.state.dev_deck))
    game.rng.random()
    assert view == AgentView.from_game(game, seat)
    first = sample_world(view, 42)
    second = sample_world(AgentView.from_game(game, seat), 42)
    assert first.state.to_dict() == second.state.to_dict()
    assert first.state.hands[seat] == game.state.hands[seat]
    assert [sum(h) for h in first.state.hands] == view.state["hand_sizes"]
    for r in range(5):
        assert first.state.bank[r] + sum(h[r] for h in first.state.hands) == 19
    mutated = view.state
    mutated["hands"][seat][0] = 999
    assert view.state["hands"][seat][0] != 999


def test_external_process_completes_and_closes():
    script = Path(__file__).resolve().parents[2] / "scripts" / "example_external_agent.py"
    agent = make_agent(AgentSpec("external", {"command": [sys.executable, str(script)]}))
    result = play_game(
        [agent, *[HeuristicAgent() for _ in range(3)]], 12, EngineConfig(max_turns=30)
    )
    assert result.steps > 16 and agent.process is None


def test_timeout_cleanup(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import time; time.sleep(30)")
    agent = make_agent(
        AgentSpec("external", {"command": [sys.executable, str(script)], "timeout": 0.1})
    )
    with pytest.raises(TimeoutError):
        play_game([agent, *[HeuristicAgent() for _ in range(3)]], 2)
    assert agent.process is None


def test_cleanup_all_agents_on_error():
    class Broken(Agent):
        closed = False

        def act(self, game, seat, legal):
            raise RuntimeError("decision failed")

        def close(self):
            self.closed = True

    bots = [Broken() for _ in range(4)]
    with pytest.raises(RuntimeError, match="decision failed"):
        play_game(bots, 2)
    assert all(bot.closed for bot in bots)


def test_chance_probabilities_and_rng_isolation():
    from carla.engine.actions import ROLL

    game = Game(seed=4)
    rng = game.rng.getstate()
    assert sum(p for _, p in outcomes(game, ROLL)) == pytest.approx(1)
    world = sample_world(AgentView.from_game(game, 0), 42)
    world.rng.random()
    assert game.rng.getstate() == rng


@pytest.mark.parametrize("name", ["native-f", "native-ab2"])
def test_native_bots_complete_games(name):
    agent = make_agent(AgentSpec(name, {"node_budget": 500}))
    result = play_game(
        [agent, *[HeuristicAgent() for _ in range(3)]], 12, EngineConfig(max_turns=30)
    )
    assert result.steps > 16


@pytest.mark.parametrize("name", ["catanatron-f", "catanatron-ab2"])
def test_optional_original_catanatron_plays_with_domestic_trading(name):
    """Catanatron has no domestic trade: it must decline offers and never open one, instead of
    failing to convert the INVITE_TRADE/PROPOSE_TRADE actions that are legal on its turn."""
    pytest.importorskip("catanatron")
    from carla.engine.actions import ActionType

    class Recording(Agent):
        def __init__(self, inner):
            self.inner, self.name, self.types = inner, inner.name, []

        def reset(self, game, seat, seed):
            self.inner.reset(game, seat, seed)

        def act(self, game, seat, legal):
            action = self.inner.act(game, seat, legal)
            self.types.append(action.type)
            return action

    bots = [Recording(make_agent(name)) for _ in range(3)]
    result = play_game(
        [HeuristicAgent(), *bots], 777, EngineConfig(max_turns=20, allow_domestic_trade=True)
    )
    assert result.turns >= 20
    types = {t for bot in bots for t in bot.types}
    assert not types & {ActionType.PROPOSE_TRADE, ActionType.INVITE_TRADE}
    assert not types & {ActionType.ACCEPT_TRADE, ActionType.CONFIRM_TRADE}


def test_optional_original_f_feature_parity():
    pytest.importorskip("catanatron")
    from catanatron.models.player import Color
    from catanatron.players.value import base_fn

    from carla.agents.catanatron.codec import reconstruct
    from carla.agents.catanatron.native import WEIGHTS, features

    game = Game(seed=41)
    bot = HeuristicAgent()
    for step in range(180):
        if game.is_over:
            break
        if step % 7 == 0:
            native = reconstruct(game, 1)
            for seat in range(4):
                f = features(game, seat)
                for key in WEIGHTS:
                    if key in ("reachable_production_0", "longest_road"):
                        continue
                    weights = dict.fromkeys(WEIGHTS, 0.0)
                    weights[key] = 1.0
                    reference = base_fn(weights)(native, tuple(Color)[seat]) - base_fn(
                        dict.fromkeys(WEIGHTS, 0.0)
                    )(native, tuple(Color)[seat])
                    assert f[key] == pytest.approx(reference), (step, seat, key)
        game.step(bot.act(game, game.decision_player, game.legal_actions()))


def test_optional_wrapped_f_completes():
    pytest.importorskip("catanatron")
    result = play_game(
        [make_agent("catanatron-f"), *[HeuristicAgent() for _ in range(3)]],
        12,
        EngineConfig(max_turns=100),
    )
    assert result.winner is not None


def test_optional_port_ratios_match_on_every_corner():
    pytest.importorskip("catanatron")
    from catanatron.models.player import Color

    from carla.agents.catanatron.codec import reconstruct
    from carla.engine.rules import trade_ratios

    for corner in range(54):
        game = Game(seed=42)
        game.state.corner_owner[corner] = 0
        game.state.corner_building[corner] = 1
        game.state.settlements_left[0] -= 1
        native = reconstruct(game, 42)
        ports = native.state.board.get_player_port_resources(next(iter(Color)))
        ratios = tuple(
            2 if r in ports else 3 if None in ports else 4
            for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
        )
        assert tuple(trade_ratios(game.state, 0)) == ratios
