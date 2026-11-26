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


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def analyze_bc(run: Path, output: Path) -> list[dict]:
    games = read_rows(run / "bc_evaluation_games.csv")
    epochs = read_rows(run / "bc_metrics.csv")
    if not games or not epochs:
        raise ValueError("BC metrics and before/after evaluation games are required")
    groups = defaultdict(list)
    for row in games:
        groups[(row["opponent"], row["stage"])].append(row)
    opponents = sorted({key[0] for key in groups})
    summaries = []
    for opponent in opponents:
        before = groups[(opponent, "before_bc")]
        after = groups[(opponent, "after_bc")]
        if not before or not after:
            raise ValueError(f"missing paired BC evaluations against {opponent}")
        before_keys = {(r["eval_seed"], r["board"], r["seat"], r["game_seed"]) for r in before}
        after_keys = {(r["eval_seed"], r["board"], r["seat"], r["game_seed"]) for r in after}
        if before_keys != after_keys or len(before_keys) != len(before):
            raise ValueError(f"BC evaluations are not paired by board and seat: {opponent}")
        if any(
            {r["seat"] for r in before if r["board"] == board} != {"0", "1", "2", "3"}
            for board in {r["board"] for r in before}
        ):
            raise ValueError(f"BC evaluation is missing seat rotations: {opponent}")
        for stage, records in (("before_bc", before), ("after_bc", after)):
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
    with (output / "bc_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(opponents))
    for index, stage in enumerate(("before_bc", "after_bc")):
        rows = [
            next(r for r in summaries if r["opponent"] == name and r["stage"] == stage)
            for name in opponents
        ]
        for ax, field in zip(axes, ("win_rate", "mean_vp"), strict=True):
            values = [r[field] for r in rows]
            bars = ax.bar(
                x + (index - 0.5) * 0.36,
                values,
                width=0.36,
                label="Before BC" if stage == "before_bc" else "After BC",
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
    fig.suptitle("Behaviour cloning on the same boards and seats")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "bc_before_after.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, ylabel in zip(
        axes, ("nll", "type_accuracy"), ("Expert action NLL", "Action type accuracy"), strict=True
    ):
        ax.plot(
            [int(row["epoch"]) for row in epochs],
            [float(row[field]) for row in epochs],
            "o-",
            color="#247A73",
        )
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.2)
    fig.suptitle("Imitation fit on the training demonstrations")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "bc_fit.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Behaviour cloning check",
        "",
        "The before/after policy played the same held-out boards in all four seats. "
        "This is a pipeline check until enough boards and training seeds are run.",
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
        "![Before and after BC](bc_before_after.png)",
        "",
        "`bc_fit.png` shows training fit; lower NLL alone does not prove stronger play.",
        "",
        "![Imitation fit](bc_fit.png)",
    ]
    (output / "bc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze_bc(args.run, args.output)
    print(f"BC figures and summary: {args.output}")


if __name__ == "__main__":
    main()
