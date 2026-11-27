"""Agents by name, so configs and worker processes can build them (`make_agent("heuristic")`)."""

from collections.abc import Callable

from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.random_agent import RandomAgent
from carla.agents.strategic import CityHeuristic, DevCardHeuristic, ExpansionHeuristic

AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "random_flat": lambda: RandomAgent(by_type=False),
    "heuristic": HeuristicAgent,
    "heuristic_city": CityHeuristic,
    "heuristic_expansion": ExpansionHeuristic,
    "heuristic_devcard": DevCardHeuristic,
}
TEACHERS = ("heuristic_city", "heuristic_expansion", "heuristic_devcard")
"""Strategy-styled teachers (`carla.agents.strategic`); their archetypes are the suffixes."""
"""Trained checkpoints are agents too: "ppo:<checkpoint path>" (see `policy_agent`)."""


SAMPLED_PREFIX = "ppo-sampled:"
"""Like "ppo:", but the policy samples its actions instead of playing greedily."""


def is_agent_name(name: str) -> bool:
    return (
        name in AGENTS
        or (name.startswith("ppo:") and len(name) > 4)
        or (name.startswith(SAMPLED_PREFIX) and len(name) > len(SAMPLED_PREFIX))
    )


def make_agent(name: str) -> Agent:
    if name.startswith(("ppo:", SAMPLED_PREFIX)):
        from carla.agents.policy_agent import PolicyAgent  # Imports torch; bots stay light.

        sampled = name.startswith(SAMPLED_PREFIX)
        path = name.removeprefix(SAMPLED_PREFIX if sampled else "ppo:")
        agent = PolicyAgent(path, deterministic=not sampled)
        agent.name = name
        return agent
    try:
        factory = AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(AGENTS)}") from None
    agent = factory()
    agent.name = name
    return agent
