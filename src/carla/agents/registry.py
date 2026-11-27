"""Agents by name, so configs and worker processes can build them (`make_agent("heuristic")`)."""

from collections.abc import Callable

from carla.agents.api import AgentSpec
from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.random_agent import RandomAgent

BENCHMARKS = (
    "catanatron-f",
    "catanatron-ab2",
    "native-f",
    "native-ab2",
    "native-f-upstream",
    "native-ab2-upstream",
    "jsettlers-smart-adapted",
    "jsettlers-fast-adapted",
    "jsettlers-smart-native",
    "jsettlers-fast-native",
)

AGENTS: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "random_flat": lambda: RandomAgent(by_type=False),
    "heuristic": HeuristicAgent,
}
"""Trained checkpoints are agents too: "ppo:<checkpoint path>" (see `policy_agent`)."""


def is_agent_name(name: str) -> bool:
    return name in AGENTS or name in BENCHMARKS or (name.startswith("ppo:") and len(name) > 4)


def make_agent(name: str | AgentSpec) -> Agent:
    parameters = {}
    if isinstance(name, AgentSpec):
        parameters = name.parameters
        name = name.implementation
    if name == "external":
        from carla.agents.external import ExternalAgent

        return ExternalAgent(**parameters)
    if name in BENCHMARKS:
        if name.startswith("jsettlers-"):
            if name.endswith("-native"):
                from carla.agents.jsettlers_native.agent import NativeJSettlersAgent

                agent = NativeJSettlersAgent(
                    strategy="fast" if "fast" in name else "smart", **parameters
                )
            else:
                from carla.agents.jsettlers import JSettlersAgent

                agent = JSettlersAgent(strategy="fast" if "fast" in name else "smart", **parameters)
        elif name.startswith("native-"):
            from carla.agents.catanatron.native import NativeCatanatronAgent

            upstream = name.endswith("-upstream")
            if upstream:
                parameters = {"chance": "upstream", **parameters}
            agent = NativeCatanatronAgent(depth=2 if "ab2" in name else 1, **parameters)
        else:
            from carla.agents.catanatron.agent import CatanatronAgent

            agent = CatanatronAgent(algorithm="AB" if name.endswith("ab2") else "F", **parameters)
        agent.name = name
        return agent
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
