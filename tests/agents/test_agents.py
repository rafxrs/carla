import pytest

from carla.agents import AGENTS, HeuristicAgent, NoisyHeuristicAgent, RandomAgent, make_agent
from carla.engine.game import Game
from carla.eval.runner import GameResult, play_game


def assert_same_game(a: GameResult, b: GameResult) -> None:
    assert a.winner == b.winner
    assert a.victory_points == b.victory_points
    assert a.turns == b.turns
    assert a.steps == b.steps


def test_registry():
    assert {
        "random",
        "heuristic05",
        "heuristic10",
        "heuristic15",
        "heuristic25",
        "heuristic50",
        "heuristic75",
        "heuristic",
    } <= set(AGENTS)
    assert "noisy_heuristic" not in AGENTS
    agent = make_agent("random_flat")
    assert isinstance(agent, RandomAgent) and agent.name == "random_flat" and not agent.by_type
    for name, probability in (
        ("heuristic05", 0.05),
        ("heuristic10", 0.10),
        ("heuristic15", 0.15),
        ("heuristic25", 0.25),
        ("heuristic50", 0.5),
        ("heuristic75", 0.75),
    ):
        noisy = make_agent(name)
        assert isinstance(noisy, NoisyHeuristicAgent) and noisy.name == name
        assert noisy.heuristic_probability == probability
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


def test_noisy_heuristic_is_reproducible():
    agents_a = [
        NoisyHeuristicAgent(0.25),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]
    agents_b = [
        NoisyHeuristicAgent(0.25),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    a = play_game(agents_a, seed=17)
    b = play_game(agents_b, seed=17)

    assert a == b


def test_noisy_heuristic_probability_endpoints():
    for seed in range(5):
        random_game = play_game(
            [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()],
            seed=seed,
        )
        noisy_random_game = play_game(
            [
                NoisyHeuristicAgent(0.0),
                RandomAgent(),
                RandomAgent(),
                RandomAgent(),
            ],
            seed=seed,
        )

        assert_same_game(random_game, noisy_random_game)

        heuristic_game = play_game(
            [HeuristicAgent(), RandomAgent(), RandomAgent(), RandomAgent()],
            seed=seed,
        )
        noisy_heuristic_game = play_game(
            [
                NoisyHeuristicAgent(1.0),
                RandomAgent(),
                RandomAgent(),
                RandomAgent(),
            ],
            seed=seed,
        )

        assert_same_game(heuristic_game, noisy_heuristic_game)
