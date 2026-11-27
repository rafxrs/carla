"""Tests execute original Java strategies when the pinned optional runtime is configured."""

import os

import pytest

from carla.agents.api import AgentView
from carla.agents.heuristic import HeuristicAgent
from carla.agents.jsettlers import JSettlersAgent
from carla.engine import Game
from carla.engine.config import EngineConfig
from carla.eval.runner import play_game
from carla.planning.jsettlers_eta import building_estimates, rolls_per_resource

pytestmark = pytest.mark.skipif(
    not os.environ.get("JSETTLERS_JAR") or not os.environ.get("CARLA_JSETTLERS_CLASSES"),
    reason="optional JSettlers JAR and compiled sidecar required",
)


def test_native_eta_matches_original_java():
    game = Game(seed=41)
    agent = JSettlersAgent("fast")
    agent.reset(game, 0, 42)
    bot = HeuristicAgent()
    try:
        for step in range(180):
            if game.is_over:
                break
            if step % 7 == 0:
                reply = agent._request("eta", view=AgentView.from_game(game, 0).to_json())
                assert rolls_per_resource(game.state, 0) == tuple(
                    reply["periods"][r] for r in (5, 1, 3, 4, 2)
                )
                assert building_estimates(game.state, 0) == tuple(reply["estimates"][:4])
            events = game.step(bot.act(game, game.decision_player, game.legal_actions()))
            agent.observe([e.visible_to(0) for e in events], game.num_steps)
    finally:
        agent.close()


@pytest.mark.parametrize("strategy", ["fast", "smart"])
@pytest.mark.parametrize("trading", [False, True])
def test_java_strategies_complete_game(strategy, trading):
    result = play_game(
        [JSettlersAgent(strategy), *[HeuristicAgent() for _ in range(3)]],
        12,
        EngineConfig(max_turns=160, allow_domestic_trade=trading),
    )
    assert result.winner is not None


def test_smart_revalidates_committed_roads_during_multi_robot_trading():
    # Regression: a retained SMART plan attempted to rebuild owned edge 58.
    result = play_game(
        [HeuristicAgent(), *[JSettlersAgent("smart") for _ in range(3)]],
        4123140968738647690,
        EngineConfig(max_turns=300, allow_domestic_trade=True),
    )
    assert result.winner is not None
    assert sum(result.trade_offers[1:]) > 0


def test_bank_trade_the_bank_cannot_pay_ends_the_turn():
    # Regression: JSettlers' bank is unlimited, so FAST asked for wheat -> ore with no ore left
    # in CARLA's bank, and the referee aborted the game.
    agents = [JSettlersAgent("fast") for _ in range(3)]
    agents.insert(3, HeuristicAgent())
    result = play_game(agents, 7370695838722492483, EngineConfig(max_turns=300))
    assert result.turns > 76
