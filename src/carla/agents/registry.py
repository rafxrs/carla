"""Agents by name, so configs and worker processes can build them (`make_agent("heuristic")`)."""

from collections.abc import Callable

from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.random_agent import RandomAgent

AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "random_flat": lambda: RandomAgent(by_type=False),
    "heuristic": HeuristicAgent,
}
"""TODO(eval): register trained policies, e.g. "ppo:<checkpoint path>"."""


def make_agent(name: str) -> Agent:
    try:
        factory = AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(AGENTS)}") from None
    agent = factory()
    agent.name = name
    return agent
