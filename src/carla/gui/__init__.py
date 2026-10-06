"""Arcade GUI: play a seat against bots, watch bots, or replay a game record (`carla gui -h`).

Only opening a window needs the `gui` extra; every module here imports without Arcade.
"""

from carla.gui.app import GuiOptions, HumanAgent, main

__all__ = ["GuiOptions", "HumanAgent", "main"]
