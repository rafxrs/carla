"""Text and JSON summary of a league run (`carla league-report <run>`)."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from carla.envs.strategy import STAT_NAMES


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def league_report(run: Path, recent: int = 400) -> str:
    run = Path(run)
    roster = json.loads((run / "roster.json").read_text(encoding="utf-8"))
    updates = _rows(run / "updates.csv")
    episodes = _rows(run / "episodes.csv")
    events = _rows(run / "events.csv")
    evaluation = _rows(run / "evaluation_games.csv")
    payoff = _rows(run / "payoff.csv")
    lines = [f"league run {run}", f"total learner decisions {roster['total_steps']}"]
    report: dict = {"total_steps": roster["total_steps"], "learners": {}}

    lines.append(
        "\nlearners (steps, generation, table win rate overall / with a main agent, strength)"
    )
    learners = [p for p in roster["players"] if p["kind"] == "learner"]
    for player in learners:
        pid = player["id"]
        own = [u for u in updates if u["learner"] == pid]
        strength = _float(own[-1]["strength"]) if own else None
        rates = roster["table_win_rates"].get(pid, {})
        lines.append(
            f"  {pid:22} {roster['learner_steps'].get(pid, 0):>10} gen {player['generation']:>2} "
            f"variant {player['variant'] or '-':10} table {rates.get('all', 0):.3f} / "
            f"{rates.get('main', 0):.3f} strength {strength if strength is not None else 0:+.2f}"
        )
        report["learners"][pid] = {
            "steps": roster["learner_steps"].get(pid, 0),
            "generation": player["generation"],
            "table_win_rates": rates,
            "strength": strength,
        }

    counts = Counter((e["event"], e["role"]) for e in events)
    lines.append(
        "\nleague events: " + ", ".join(f"{k[0]} {k[1]}: {v}" for k, v in sorted(counts.items()))
    )
    frozen = [p for p in roster["players"] if p["kind"] != "learner"]
    lines.append(f"frozen players now: {len(frozen)} ({', '.join(p['id'] for p in frozen)})")

    if evaluation:
        lines.append("\nmain-agent validation (win rate by evaluation point and opponent)")
        grouped = defaultdict(list)
        for r in evaluation:
            grouped[(r["learner"], int(r["env_steps"]), r["opponent"])].append(
                r["outcome"] == "win"
            )
        report["validation"] = []
        for (pid, steps, opponent), wins in sorted(grouped.items()):
            rate = float(np.mean(wins))
            lines.append(f"  {pid} {steps:>10} vs {opponent:55} {rate:.3f} ({len(wins)} games)")
            report["validation"].append(
                {"learner": pid, "steps": steps, "opponent": opponent, "win_rate": rate}
            )

    main = [p["id"] for p in learners if p["role"] == "main"]
    for pid in main:
        own = [e for e in episodes if e["learner"] == pid][-recent:]
        if not own:
            continue
        lines.append(f"\n{pid}: last {len(own)} training episodes by strategy variant")
        by_variant = defaultdict(list)
        for e in own:
            by_variant[e["strategy"] or "-"].append(e)
        report.setdefault("variants", {})[pid] = {}
        for variant, rows in sorted(by_variant.items()):
            wins = np.mean([r["outcome"] == "win" for r in rows])
            adherence = [_float(r["strategy_adherence"]) for r in rows]
            adherence = [a for a in adherence if a is not None]
            stats = {
                name: float(np.mean([_float(r[f"stat_{name}"]) or 0 for r in rows]))
                for name in STAT_NAMES
            }
            switches = [_float(r["strategy_switches"]) for r in rows]
            switches = [s for s in switches if s is not None]
            lines.append(
                f"  {variant:10} {len(rows):>4} games win {wins:.3f} adherence "
                f"{np.mean(adherence) if adherence else float('nan'):.3f} switches "
                f"{np.mean(switches) if switches else 0:.2f} | "
                + " ".join(f"{k} {v:.2f}" for k, v in stats.items())
            )
            report["variants"][pid][variant] = {
                "games": len(rows),
                "win_rate": float(wins),
                "adherence": float(np.mean(adherence)) if adherence else None,
                "stats": stats,
            }
        choices = Counter(c for e in own for c in e["strategy_choices"].split("-") if c)
        if choices:
            total = sum(choices.values())
            lines.append(
                "  selector choices: "
                + ", ".join(f"{k}: {v / total:.2f}" for k, v in sorted(choices.items()))
            )
        hardest = sorted(
            (r for r in payoff if r["learner"] == pid and float(r["games"]) >= 5),
            key=lambda r: float(r["win_rate"]),
        )[:8]
        if hardest:
            lines.append(f"  hardest opponents for {pid} (pairwise rate, decayed games):")
            for r in hardest:
                lines.append(f"    {r['opponent']:40} {float(r['win_rate']):.3f} ({r['games']})")

    behaviour = roster.get("behaviour", {})
    if behaviour:
        lines.append("\nbehaviour of players (running mean of normalized build statistics)")
        lines.append("  " + " ".join(f"{n[:10]:>10}" for n in STAT_NAMES))
        for pid, values in sorted(behaviour.items()):
            lines.append(f"  {pid:28}" + " ".join(f"{v:10.3f}" for v in values))
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return "\n".join(lines)
