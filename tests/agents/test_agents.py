import pytest

from carla.agents import AGENTS, HeuristicAgent, RandomAgent, make_agent
from carla.engine.game import Game
from carla.eval.runner import play_game


def test_registry():
    assert {"random", "heuristic"} <= set(AGENTS)
    agent = make_agent("random_flat")
    assert isinstance(agent, RandomAgent) and agent.name == "random_flat" and not agent.by_type
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("alphazero")


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agents_only_choose_legal_actions(name):
    game = Game(seed=4)
    agents = [make_agent(name) for _ in range(4)]
    for seat, agent in enumerate(agents):
        agent.reset(game, seat, seed=seat)
    while not game.is_over:
        seat = game.decision_player
        action = agents[seat].act(game, seat, game.legal_actions())
        assert game.is_legal(action)
        game.step(action)
    assert game.winner is not None


def test_agents_are_reproducible():
    a = play_game([RandomAgent(), HeuristicAgent(), RandomAgent(), HeuristicAgent()], seed=9)
    b = play_game([RandomAgent(), HeuristicAgent(), RandomAgent(), HeuristicAgent()], seed=9)
    assert a == b


def test_heuristic_beats_random():
    wins = sum(
        play_game([HeuristicAgent(), RandomAgent(), RandomAgent(), RandomAgent()], seed=s).winner
        == 0
        for s in range(12)
    )
    assert wins >= 10
