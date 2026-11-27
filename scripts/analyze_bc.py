"""Summarize a behavioural-cloning run's before/after evaluations."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from carla.eval.style import AFTER_BC, BEFORE_BC, REPORT_WIDTH, save_report_figure, use_report_style
from carla.rl.evaluate import evaluate_checkpoint


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze_bc(run: Path, output: Path, *, boards: int = 25, seed: int = 10_000) -> list[dict]:
    use_report_style()
    checkpoint = run if run.suffix == ".pt" else run / "bc.pt"
    run = checkpoint.parent
    games_path = run / "bc_evaluation_games.csv"
    epochs_path = run / "bc_metrics.csv"
    epochs = read_rows(epochs_path) if epochs_path.exists() else []
    if games_path.exists():
        games = read_rows(games_path)
    else:
        if not checkpoint.exists() or boards <= 0:
            raise ValueError("a BC checkpoint and positive --boards are required")
        games = [
            {"stage": "after_bc", **record}
            for opponent in ("random", "heuristic")
            for record in evaluate_checkpoint(
                checkpoint, opponent=opponent, boards=boards, seed=seed
            ).records
        ]
    if not games:
        raise ValueError("BC evaluation has no game records")
    groups = defaultdict(list)
    for row in games:
        groups[(row["opponent"], row["stage"])].append(row)
    opponents = sorted({key[0] for key in groups})
    if len({bool(groups[(name, "before_bc")]) for name in opponents}) != 1:
        raise ValueError("before-BC records must exist for every opponent or none")
    summaries = []
    for opponent in opponents:
        before = groups[(opponent, "before_bc")]
        after = groups[(opponent, "after_bc")]
        if not after:
            raise ValueError(f"missing trained BC evaluation against {opponent}")
        if before:
            before_keys = {(r["eval_seed"], r["board"], r["seat"], r["game_seed"]) for r in before}
            after_keys = {(r["eval_seed"], r["board"], r["seat"], r["game_seed"]) for r in after}
            if before_keys != after_keys or len(before_keys) != len(before):
                raise ValueError(f"BC evaluations are not paired by board and seat: {opponent}")
        if any(
            {int(r["seat"]) for r in after if r["board"] == board} != {0, 1, 2, 3}
            for board in {r["board"] for r in after}
        ):
            raise ValueError(f"BC evaluation is missing seat rotations: {opponent}")
        for stage, records in (("before_bc", before), ("after_bc", after)):
            if not records:
                continue
            summaries.append(
                {
                    "opponent": opponent,
                    "stage": stage,
                    "boards": len({r["board"] for r in records}),
                    "games": len(records),
                    "wins": sum(r["outcome"] == "win" for r in records),
                    "win_rate": np.mean([r["outcome"] == "win" for r in records]),
                    "mean_vp": np.mean([float(r["victory_points"]) for r in records]),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    if not games_path.exists():
        with (output / "bc_evaluation_games.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(games[0]))
            writer.writeheader()
            writer.writerows(games)
    with (output / "bc_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 3.2))
    x = np.arange(len(opponents))
    stages = (
        ("before_bc", "after_bc")
        if any(r["stage"] == "before_bc" for r in summaries)
        else ("after_bc",)
    )
    for index, stage in enumerate(stages):
        rows = [
            next(r for r in summaries if r["opponent"] == name and r["stage"] == stage)
            for name in opponents
        ]
        for ax, field in zip(axes, ("win_rate", "mean_vp"), strict=True):
            values = [r[field] for r in rows]
            bars = ax.bar(
                x + (index - (len(stages) - 1) / 2) * 0.36,
                values,
                width=0.36,
                label="Before BC" if stage == "before_bc" else "After BC",
                color=BEFORE_BC if stage == "before_bc" else AFTER_BC,
            )
            ax.bar_label(
                bars,
                labels=[f"{v:.0%}" if field == "win_rate" else f"{v:.1f}" for v in values],
                padding=3,
                fontsize=8,
            )
    for ax in axes:
        ax.set_xticks(x, opponents)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(ylabel="Win rate", ylim=(0, 1.08))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(ylabel="Mean VP", ylim=(0, max(2.5, max(r["mean_vp"] for r in summaries) * 1.2)))
    axes[0].legend(frameon=False)
    fig.suptitle("Behaviour cloning on held-out boards")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_report_figure(fig, output / "bc_before_after.png")
    plt.close(fig)

    if epochs:
        fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 3.2))
        for ax, field, ylabel in zip(
            axes,
            ("nll", "type_accuracy"),
            ("Expert action NLL", "Action type accuracy"),
            strict=True,
        ):
            ax.plot(
                [int(row["epoch"]) for row in epochs],
                [float(row[field]) for row in epochs],
                "o-",
                color=AFTER_BC,
            )
            ax.set(xlabel="Epoch", ylabel=ylabel)
            ax.grid(alpha=0.2)
        fig.suptitle("Imitation fit on the training demonstrations")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_report_figure(fig, output / "bc_fit.png")
        plt.close(fig)
    lines = [
        "# Behaviour cloning check",
        "",
        (
            "Before and after BC were evaluated on the same boards in all four seats."
            if len(stages) == 2
            else "Only the trained checkpoint was available. Before-BC performance cannot be recovered from its weights."
        ),
        "Treat small board counts as a pipeline check, not evidence of playing strength.",
        "",
        "| Opponent | Stage | Games | Wins | Win rate | Mean VP |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        stage = "Before BC" if row["stage"] == "before_bc" else "After BC"
        lines.append(
            f"| {row['opponent']} | {stage} | {row['games']} | "
            f"{row['wins']} | {row['win_rate']:.1%} | {row['mean_vp']:.2f} |"
        )
    lines += [
        "",
        "![BC evaluation](bc_before_after.png) [PDF](bc_before_after.pdf)",
    ]
    if epochs:
        lines += [
            "",
            "Training fit does not by itself prove stronger play.",
            "",
            "![Imitation fit](bc_fit.png) [PDF](bc_fit.pdf)",
        ]
    (output / "bc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--boards", type=int, default=25, help="boards per opponent when only bc.pt is available"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="held-out board seed for checkpoint-only evaluation",
    )
    args = parser.parse_args()
    analyze_bc(args.run, args.output, boards=args.boards, seed=args.seed)
    print(f"BC figures and summary: {args.output}")


if __name__ == "__main__":
    main()
