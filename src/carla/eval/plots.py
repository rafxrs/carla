"""Figures for the report.

TODO(eval): win-rate bars with confidence intervals per agent/ablation, learning curves (win rate
vs. environment steps against fixed baselines), game-length histograms.
"""

from pathlib import Path

from carla.eval.metrics import AgentSummary


def plot_win_rates(summary: dict[str, AgentSummary], path: str | Path) -> Path:
    raise NotImplementedError("eval workstream: plot win rates with their confidence intervals")
