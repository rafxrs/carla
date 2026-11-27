"""Rebuild the training, cloning and tournament figures for a report."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_bc import analyze_bc

from carla.eval.analysis import analyze_runs
from carla.eval.metrics import AgentSummary
from carla.eval.plots import plot_win_rates


def tournament_summary(path: Path, output: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if (
        not rows
        or not {
            "player",
            "board",
            "game_seed",
            "rotation",
            "slot",
            "seat",
            "outcome",
            "victory_points",
        }
        <= rows[0].keys()
    ):
        raise ValueError(f"missing tournament records or columns: {path}")
    by_board_slot = defaultdict(list)
    for row in rows:
        if row["outcome"] not in ("win", "loss", "truncated"):
            raise ValueError(f"invalid tournament outcome: {path}")
        by_board_slot[(row["game_seed"], row["board"], row["slot"])].append(row)
    for records in by_board_slot.values():
        if len(records) != 4 or {r["seat"] for r in records} != {"0", "1", "2", "3"}:
            raise ValueError(f"tournament is missing a seat rotation: {path}")
    board_slots = defaultdict(set)
    for game_seed, board, slot in by_board_slot:
        board_slots[(game_seed, board)].add(slot)
    if any(slots != {"0", "1", "2", "3"} for slots in board_slots.values()):
        raise ValueError(f"tournament is missing a player slot: {path}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["player"]].append(row)
    summary = {}
    for player, records in sorted(grouped.items()):
        games = len(records)
        wins = sum(row["outcome"] == "win" for row in records)
        boards = defaultdict(list)
        for row in records:
            boards[(row["game_seed"], row["board"])].append(row["outcome"] == "win")
        board_rates = np.asarray([np.mean(outcomes) for outcomes in boards.values()])
        rng = np.random.default_rng(0)
        samples = rng.choice(board_rates, size=(2000, len(board_rates)), replace=True).mean(axis=1)
        low, high = np.percentile(samples, (2.5, 97.5))
        summary[player] = AgentSummary(
            player,
            games,
            wins,
            wins / games,
            low,
            high,
            sum(float(row["victory_points"]) for row in records) / games,
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tournament_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("player", "games", "wins", "win_rate", "mean_vp"))
        writer.writerows((s.name, s.games, s.wins, s.win_rate, s.mean_vp) for s in summary.values())
    display_names = {}
    chart = {}
    for player, item in summary.items():
        label = player if player in ("random", "heuristic") else f"Policy {len(display_names) + 1}"
        display_names[label] = player
        chart[label] = AgentSummary(
            label, item.games, item.wins, item.win_rate, item.ci_low, item.ci_high, item.mean_vp
        )
    color_keys = {}
    for label, player in display_names.items():
        checkpoint = Path(player)
        if checkpoint.suffix == ".pt":
            run = (
                checkpoint.parent.parent
                if checkpoint.parent.name == "checkpoints"
                else checkpoint.parent
            )
            color_keys[label] = run.name
        else:
            color_keys[label] = player
    plot_win_rates(chart, output / "tournament_win_rates.png", color_keys=color_keys)
    return display_names


def build_report_figures(
    runs: list[Path],
    output: Path,
    *,
    split: str = "validation",
    lineage: dict[str, int] | None = None,
    smooth_updates: int = 25,
    include_incomplete: bool = False,
    bc: Path | None = None,
    bc_boards: int = 25,
    bc_seed: int = 10_000,
    tournament: Path | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result = analyze_runs(
        runs,
        output / "training",
        split=split,
        lineage=lineage,
        smooth_updates=smooth_updates,
        include_incomplete=include_incomplete,
    )
    lines = ["# Report figures", "", "Training: [results](training/report.md)."]
    if bc is not None:
        analyze_bc(bc, output / "bc", boards=bc_boards, seed=bc_seed)
        lines += ["", "Behaviour cloning: [results](bc/bc_report.md)."]
    if tournament is not None:
        display_names = tournament_summary(tournament, output / "tournament")
        lines += [
            "",
            "Tournament: [summary](tournament/tournament_summary.csv).",
            "",
            "![Tournament win rates](tournament/tournament_win_rates.png) "
            "[PDF](tournament/tournament_win_rates.pdf)",
            "",
            "Intervals resample boards with their seat rotations together. With few boards they remain exploratory.",
            "",
        ]
        lines.extend(
            f"- {label}: `{player}`" for label, player in display_names.items() if label != player
        )
    lines += [
        "",
        "All figures are available as PNG and PDF. Validation plots are for model selection; use a separate test split for final results.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--lineage", type=Path, help="JSON mapping moved run names to their starting decisions"
    )
    parser.add_argument("--smooth-updates", type=int, default=25)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--bc", type=Path, help="BC directory or bc.pt checkpoint")
    parser.add_argument("--bc-boards", type=int, default=25)
    parser.add_argument("--bc-seed", type=int, default=10_000)
    parser.add_argument("--tournament", type=Path, help="existing policy-tournament game CSV")
    args = parser.parse_args()
    lineage = json.loads(args.lineage.read_text(encoding="utf-8")) if args.lineage else None
    if lineage is not None and not isinstance(lineage, dict):
        parser.error("--lineage must contain a JSON object")
    result = build_report_figures(
        args.runs,
        args.output,
        split=args.split,
        lineage=lineage,
        smooth_updates=args.smooth_updates,
        include_incomplete=args.include_incomplete,
        bc=args.bc,
        bc_boards=args.bc_boards,
        bc_seed=args.bc_seed,
        tournament=args.tournament,
    )
    print(f"Built report figures for {result['runs']} training runs: {args.output}")
    for notice in result["warnings"]:
        print(f"Warning: {notice}")


if __name__ == "__main__":
    main()
