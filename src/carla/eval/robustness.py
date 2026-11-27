"""Robustness matrix: one subject against several opponent tables (`carla robustness`).

Every agent is built by name (`carla.agents.make_agent`), so bots and any trained checkpoint
(`ppo:<path>[@variant]`, recurrent and strategy-conditioned ones included) can be the subject
or an opponent. Each table is played on `boards` seeded boards in all four seat rotations of the
subject. Reported per table: the subject's win rate with a Wilson interval, mean victory
points, and its mean build statistics; overall: the mean and the worst table win rate (a
max-min view of robustness, the quantity PFSP's f_hard targets).
"""

import csv
import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from carla.agents import make_agent
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_PLAYERS
from carla.engine.rng import derive_seed
from carla.envs.strategy import STAT_NAMES
from carla.eval.behaviour import play_tracked
from carla.eval.metrics import wilson_interval

DEFAULT_TABLES = (
    ("heuristic", "heuristic", "heuristic"),
    ("heuristic_city", "heuristic_city", "heuristic_city"),
    ("heuristic_expansion", "heuristic_expansion", "heuristic_expansion"),
    ("heuristic_devcard", "heuristic_devcard", "heuristic_devcard"),
    ("heuristic_city", "heuristic_expansion", "heuristic_devcard"),
)


def _play(job: tuple) -> dict:
    subject, table, board, rotation, seed, engine = job
    import torch

    torch.set_num_threads(1)
    names = [subject, *table]
    seats = [(slot + rotation) % NUM_PLAYERS for slot in range(NUM_PLAYERS)]
    agents = [None] * NUM_PLAYERS
    for slot, name in enumerate(names):
        agents[seats[slot]] = make_agent(name)
    game_seed = derive_seed(seed, "robustness-board", board)
    game, stats = play_tracked(agents, game_seed, engine)
    seat = seats[0]
    return {
        "table": ",".join(table),
        "board": board,
        "rotation": rotation,
        "seat": seat,
        "outcome": "truncated" if game.is_truncated else ("win" if game.winner == seat else "loss"),
        "victory_points": game.victory_points(seat),
        "turns": game.turn,
        **{f"stat_{k}": float(v) for k, v in zip(STAT_NAMES, stats[seat].counts, strict=True)},
    }


def robustness_matrix(
    subject: str,
    tables: Sequence[Sequence[str]] = DEFAULT_TABLES,
    *,
    boards: int = 10,
    seed: int = 30_000,
    engine: EngineConfig | None = None,
    workers: int = 1,
    csv_path: Path | None = None,
) -> dict:
    jobs = [
        (subject, tuple(table), board, rotation, seed, engine)
        for table in tables
        for board in range(boards)
        for rotation in range(NUM_PLAYERS)
    ]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            records = pool.map(_play, jobs, chunksize=1)
    else:
        records = [_play(job) for job in jobs]
    rows = []
    for table in tables:
        key = ",".join(table)
        games = [r for r in records if r["table"] == key]
        wins = sum(r["outcome"] == "win" for r in games)
        low, high = wilson_interval(wins, len(games))
        row = {
            "table": key,
            "games": len(games),
            "wins": wins,
            "win_rate": wins / len(games),
            "ci_low": low,
            "ci_high": high,
            "mean_vp": float(np.mean([r["victory_points"] for r in games])),
        }
        for name in STAT_NAMES:
            row[f"mean_{name}"] = float(np.mean([r[f"stat_{name}"] for r in games]))
        rows.append(row)
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    rates = [r["win_rate"] for r in rows]
    return {
        "subject": subject,
        "tables": rows,
        "mean_win_rate": float(np.mean(rates)),
        "worst_win_rate": float(np.min(rates)),
        "worst_table": rows[int(np.argmin(rates))]["table"],
    }


def format_matrix(result: dict) -> str:
    lines = [f"subject {result['subject']}"]
    lines.append(
        f"{'table':62} {'games':>5} {'win':>6} {'95% CI':>15} {'VP':>5} {'cities':>6} "
        f"{'setl':>5} {'roads':>5} {'devs':>5}"
    )
    for r in result["tables"]:
        lines.append(
            f"{r['table']:62} {r['games']:5d} {r['win_rate']:6.3f} "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['mean_vp']:5.2f} "
            f"{r['mean_cities']:6.2f} {r['mean_settlements']:5.2f} {r['mean_roads']:5.2f} "
            f"{r['mean_dev_bought']:5.2f}"
        )
    lines.append(
        f"mean win rate {result['mean_win_rate']:.3f}; worst {result['worst_win_rate']:.3f} "
        f"({result['worst_table']})"
    )
    return "\n".join(lines)
