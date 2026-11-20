"""Figures for tournaments and reproducible experiment analysis."""

from pathlib import Path

import numpy as np

from carla.eval.metrics import AgentSummary


def plot_win_rates(summary: dict[str, AgentSummary], path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    values = list(summary.values())
    fig, ax = plt.subplots()
    rates = np.asarray([s.win_rate for s in values])
    ax.bar(
        [s.name for s in values],
        rates,
        yerr=[rates - [s.ci_low for s in values], [s.ci_high for s in values] - rates],
        capsize=4,
    )
    ax.set(ylabel="Win rate", ylim=(0, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def draw_curves(ax, groups: dict, *, xlabel: str, ylabel: str) -> None:
    """Thin per-seed curves plus mean/range on common support; no extrapolation.

    Linear interpolation only aligns aggregate curves; the individual measured points
    remain visible. Shading is the observed seed range, not a confidence interval.
    """
    import matplotlib.pyplot as plt

    for index, (label, curves) in enumerate(sorted(groups.items())):
        color = plt.get_cmap("tab10")(index % 10)
        usable = []
        for x, y in curves:
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            if not len(x):
                continue
            ax.plot(x, y, ".-", color=color, alpha=0.25, linewidth=0.8, markersize=3)
            usable.append((x, y))
        if not usable:
            continue
        low = max(x[0] for x, _ in usable)
        high = min(x[-1] for x, _ in usable)
        if low > high:
            continue
        grid = np.unique(np.concatenate([x[(x >= low) & (x <= high)] for x, _ in usable]))
        if not len(grid):
            continue
        values = np.stack([np.interp(grid, x, y) for x, y in usable])
        ax.plot(grid, values.mean(0), ".-", color=color, label=label, linewidth=1.7)
        if len(usable) > 1:
            ax.fill_between(grid, values.min(0), values.max(0), color=color, alpha=0.12)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2)
