"""Strategy-styled heuristic teachers: legal, deterministic, competent and distinct."""

import pytest

from carla.agents import TEACHERS, make_agent
from carla.agents.strategic import GOAL_COSTS, StrategicHeuristic
from carla.engine import actions as A
from carla.engine.config import EngineConfig
from carla.eval.behaviour import behaviour_profile, play_tracked
from tests.engine.scenario import give, post_setup_game


@pytest.mark.parametrize("name", TEACHERS)
def test_teachers_play_legal_deterministic_games_with_domestic_trade(name):
    engine = EngineConfig(allow_domestic_trade=True)
    results = []
    for _ in range(2):
        agents = [make_agent(name), make_agent("heuristic"), make_agent(name), make_agent("random")]
        game, _ = play_tracked(agents, seed=5, engine=engine)
        results.append((game.winner, game.turn, game.state_hash()))
    assert results[0] == results[1]


def test_teacher_styles_are_distinct_and_beat_random_bots():
    rows = {
        r["agent"]: r
        for r in behaviour_profile(
            [lambda n=n: make_agent(n) for n in (*TEACHERS, "random")], games=3, seed=1
        )
    }
    city, expansion, devcard = (rows[n] for n in TEACHERS)
    assert city["mean_cities"] > expansion["mean_cities"]
    assert expansion["mean_roads"] > city["mean_roads"]
    assert expansion["mean_settlements"] > city["mean_settlements"]
    assert devcard["mean_dev_bought"] > max(city["mean_dev_bought"], expansion["mean_dev_bought"])
    assert devcard["mean_knights"] > city["mean_knights"]
    assert rows["random"]["win_rate"] < min(r["win_rate"] for r in (city, expansion, devcard))


def test_reserved_cards_are_kept_when_discarding_and_buying():
    game = post_setup_game()
    agent = StrategicHeuristic("city")
    agent.reset(game, 0, 0)
    agent.goal = "city"
    give(game, 0, (0, 0, 4, 2, 2))
    s = game.state
    assert agent.reserved(s, 0) == [0, 0, 0, 2, 2]
    # A development card costs wool, grain and ore: it would eat into the city's grain and ore.
    assert not agent._keeps_reserve(s, 0, GOAL_COSTS["dev"])
    discards = [A.DISCARD_OF[r] for r in range(5) if s.hands[0][r]]
    assert agent._discard(s, 0, discards).resource == 2  # wool: the only surplus


def test_plan_keeps_its_goal_unless_clearly_beaten():
    game = post_setup_game()
    agent = StrategicHeuristic("expansion")
    agent.reset(game, 0, 0)
    first = agent.plan(game.state, 0)
    assert agent.plan(game.state, 0) == first
    assert agent.goal_changes == 1
