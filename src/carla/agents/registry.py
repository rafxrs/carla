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
"""Trained checkpoints are agents too: "ppo:<checkpoint path>" (see `policy_agent`)."""


def is_agent_name(name: str) -> bool:
    return name in AGENTS or (name.startswith("ppo:") and len(name) > 4)


def make_agent(name: str) -> Agent:
    if name.startswith("ppo:"):
        from carla.agents.policy_agent import PolicyAgent  # Imports torch; bots stay light.

        return PolicyAgent(name.removeprefix("ppo:"))
    try:
        factory = AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; known: {sorted(AGENTS)}") from None
    agent = factory()
    agent.name = name
    return agent
