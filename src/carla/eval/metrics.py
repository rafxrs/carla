"""Win-rate statistics for tournaments."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

from carla.engine.events import EventKind
from carla.engine.rules import can_afford
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
    """Offer feasibility, acceptance and execution rates from recorded games.

    Feasibility uses hidden state for retrospective evaluation only: a proposal is feasible if
    at least one opponent held all requested cards when it was made. It must never be exposed
    to the proposer through action masks or observations.
    """
    proposals = invitations = counteroffers = feasible = counter_feasible = accepted = executed = 0
    for result in results:
        if result.record is None:
            raise ValueError("trade metrics require recorded games")
        had_acceptance = False
        original_round = False
        for game, events in result.record.iter_steps():
            for event in events:
                if event.kind == EventKind.TRADE_PROPOSED:
                    proposals += 1
                    original_round = True
                    had_acceptance = False
                    if any(
                        q != event.player and can_afford(game.state.hands[q], event.receive)
                        for q in range(len(game.state.hands))
                    ):
                        feasible += 1
                elif event.kind == EventKind.TRADE_INVITED:
                    invitations += 1
                    original_round = False
                    had_acceptance = False
                elif event.kind == EventKind.TRADE_COUNTERED:
                    counteroffers += 1
                    counter_feasible += int(
                        can_afford(game.state.hands[event.target], event.receive)
                    )
                elif event.kind == EventKind.TRADE_RESPONDED and event.value == 1:
                    had_acceptance = True
                elif event.kind in (EventKind.TRADE_EXECUTED, EventKind.TRADE_CANCELLED):
                    accepted += int(original_round and had_acceptance)
                    executed += int(event.kind == EventKind.TRADE_EXECUTED)
                    had_acceptance = False
    return {
        "proposals": float(proposals),
        "invitations": float(invitations),
        "counteroffers": float(counteroffers),
        "feasibility_rate": feasible / proposals if proposals else 0.0,
        "acceptance_rate": accepted / proposals if proposals else 0.0,
        "counter_feasibility_rate": counter_feasible / counteroffers if counteroffers else 0.0,
        "execution_rate": executed / (proposals + invitations) if proposals + invitations else 0.0,
    }
