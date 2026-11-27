"""Players: the `Agent` interface, baseline bots and the registry used by tournaments."""

from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.random_agent import RandomAgent
from carla.agents.registry import AGENTS, TEACHERS, is_agent_name, make_agent
from carla.agents.strategic import StrategicHeuristic

__all__ = [
    "AGENTS",
    "TEACHERS",
    "Agent",
    "HeuristicAgent",
    "RandomAgent",
    "StrategicHeuristic",
    "is_agent_name",
    "make_agent",
]
