"""Win-rate statistics for tournaments."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

from carla.eval.runner import GameResult


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well behaved near 0 and 1)."""
    if trials == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass(frozen=True)
class AgentSummary:
    """Per-agent results. A game counts once per seat the agent occupies (chance level: 25%)."""

    name: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    mean_vp: float


def summarize(results: Sequence[GameResult], confidence: float = 0.95) -> dict[str, AgentSummary]:
    stats: dict[str, list[float]] = {}
    for r in results:
        for seat, name in enumerate(r.agents):
            games, wins, vp = stats.setdefault(name, [0, 0, 0.0])
            stats[name] = [games + 1, wins + (r.winner == seat), vp + r.victory_points[seat]]
    out = {}
    for name, (games, wins, vp) in sorted(stats.items()):
        low, high = wilson_interval(int(wins), int(games), confidence)
        out[name] = AgentSummary(name, int(games), int(wins), wins / games, low, high, vp / games)
    return out


def format_summary(summary: dict[str, AgentSummary], confidence: float = 0.95) -> str:
    level = f"{round(confidence * 100)}% CI"
    lines = [f"{'agent':<16}{'seats':>7}{'wins':>7}{'win rate':>10}  {level:<17}{'mean VP':>8}"]
    for s in summary.values():
        ci = f"[{s.ci_low:.3f}, {s.ci_high:.3f}]"
        lines.append(
            f"{s.name:<16}{s.games:>7}{s.wins:>7}{s.win_rate:>10.3f}  {ci:<17}{s.mean_vp:>8.2f}"
        )
    return "\n".join(lines)


def trade_metrics(results: Sequence[GameResult]) -> dict[str, float]:
    """Domestic-trade metrics: proposal feasibility rate and acceptance rate per agent.

    TODO(eval): implement once the engine supports domestic trade (count TRADE_* events from
    each result's record).
    """
    raise NotImplementedError("domestic trade is not implemented in the engine yet")
