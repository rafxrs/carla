"""Players: the `Agent` interface, baseline bots and the registry used by tournaments."""

from carla.agents.base import Agent
from carla.agents.random_agent import RandomAgent

__all__ = ["Agent", "RandomAgent"]
