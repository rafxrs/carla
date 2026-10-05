"""Players: the `Agent` interface, baseline bots and the registry used by tournaments."""

from carla.agents.base import Agent
from carla.agents.heuristic import HeuristicAgent
from carla.agents.random_agent import RandomAgent
from carla.agents.registry import AGENTS, make_agent

__all__ = ["AGENTS", "Agent", "HeuristicAgent", "RandomAgent", "make_agent"]
