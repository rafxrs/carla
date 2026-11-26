"""Validate run journals and produce seed-aware CSV summaries and static figures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from carla.config import config_hash, load_config, to_dict
from carla.eval.plots import draw_curves
from carla.rl.metrics import ACTION_FIELDS
from carla.utils.logging import write_json


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"missing experiment journal: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames is None or any(None in r or None in r.values() for r in rows):
            raise ValueError(f"incomplete or malformed CSV: {path}")
    return rows


def _unique(rows: list[dict], keys: tuple[str, ...], path: Path) -> None:
    identities = [tuple(row[key] for key in keys) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate records in {path}: {keys}")


def load_run(path: Path, include_incomplete: bool = False) -> dict:
    config = load_config(path / "config.yaml")
    updates = read_csv(path / "updates.csv")
    episodes = read_csv(path / "episodes.csv")
    evaluation = read_csv(path / "evaluation_games.csv")
    for export in sorted(path.glob("evaluation_*_games.csv")):
        evaluation.extend(read_csv(export))
    _unique(updates, ("update",), path / "updates.csv")
    _unique(episodes, ("env_id", "episode_id"), path / "episodes.csv")
    _unique(
        evaluation,
        ("env_steps", "checkpoint", "split", "opponent", "mode", "eval_seed", "board", "seat"),
        path / "evaluation_games.csv",
    )
    steps = [int(row["env_steps"]) for row in updates]
    versions = [int(row["update"]) for row in updates]
    if steps != sorted(set(steps)) or versions != sorted(versions):
        raise ValueError(f"non-monotonic update journal: {path}")
    for row in updates:
        for key, value in row.items():
            if value != "" and not np.isfinite(float(value)):
                raise ValueError(f"non-finite {key} in {path}")
    last_step = steps[-1] if steps else 0
    if any(int(row["env_steps"]) > last_step for row in episodes + evaluation):
        raise ValueError(f"episode/evaluation journal is ahead of updates: {path}")
    if any(row["outcome"] not in ("win", "loss", "truncated") for row in episodes + evaluation):
        raise ValueError(f"invalid outcome in {path}")
    metadata_path = path / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    complete = metadata.get("status") == "complete" and last_step == config.ppo.total_env_steps
    if not complete and not include_incomplete:
        raise ValueError(f"incomplete run (use --include-incomplete for exploratory plots): {path}")
    canonical = to_dict(config)
    canonical.pop("seed")
    warm_start = bool(metadata.get("sessions", [{}])[0].get("init_from"))
    canonical["warm_start"] = warm_start
    for key in ("output_dir", "checkpoint_every", "log_every"):
        canonical["runtime"].pop(key)
    identity = config_hash(canonical)
    label = (
        f"{config.name} (gamma={config.ppo.gamma:g}, lambda={config.ppo.gae_lambda:g}, "
        f"T={config.ppo.rollout_steps}, N={config.runtime.num_workers * config.runtime.envs_per_worker}"
        f"{', warm start' if warm_start else ''})"
    )
    notices = []
    if not complete:
        notices.append(f"Incomplete run included: {path}")
    if not evaluation:
        notices.append(
            f"No policy evaluations; optimizer diagnostics cannot rank playing strength: {path}"
        )
    missing = {
        r["checkpoint"]
        for r in evaluation
        if r["checkpoint"] and not (path / r["checkpoint"]).exists()
    }
    if missing:
        notices.append(f"{len(missing)} evaluation checkpoints unavailable for replay: {path}")
    return {
        "path": path,
        "config": config,
        "group": identity,
        "label": label,
        "updates": updates,
        "episodes": episodes,
        "evaluation": evaluation,
        "complete": complete,
        "metadata": metadata,
        "notices": notices,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_points(run: dict, split: str) -> dict:
    points = defaultdict(list)
    for row in run["evaluation"]:
        if row["split"] == split:
            points[(row["opponent"], row["mode"], int(row["env_steps"]))].append(row)
    return points


def _cluster_interval(groups: list[list[dict]], repetitions: int = 2000) -> tuple[float, float]:
    """Resample training runs, then boards (keeping their four seats together).

    This is statistical resampling, not RL value bootstrapping. Few training seeds and
    unanimous outcomes can yield misleadingly narrow intervals; always show seed ranges.
    """
    board_scores = []
    for records in groups:
        boards = defaultdict(list)
        for row in records:
            boards[(row["eval_seed"], row["board"])].append(row["outcome"] == "win")
        board_scores.append(np.asarray([np.mean(v) for v in boards.values()]))
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(repetitions):
        selected = rng.integers(len(board_scores), size=len(board_scores))
        samples.append(
            np.mean(
                [
                    rng.choice(board_scores[i], size=len(board_scores[i]), replace=True).mean()
                    for i in selected
                ]
            )
        )
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def analyze_runs(
    inputs: list[Path], output: Path, *, include_incomplete: bool = False, split: str = "validation"
) -> dict:
    paths = set()
    for root in inputs:
        if (root / "updates.csv").exists():
            paths.add(root.resolve())
        else:
            paths.update(p.parent.resolve() for p in root.rglob("updates.csv"))
    if not paths:
        raise ValueError("no experiment runs found")
    runs, notices = [], []
    for path in sorted(paths):
        try:
            run = load_run(path, include_incomplete)
        except ValueError as exc:
            if str(exc).startswith("incomplete run"):
                notices.append(str(exc))
                continue
            raise
        notices.extend(run["notices"])
        runs.append(run)
    if not runs:
        raise ValueError("no completed runs found; inspect failures or use --include-incomplete")
    identities = [(r["group"], r["config"].seed) for r in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate configuration/training seed; select one attempt per seed")
    machines = {
        (session.get("device"), session.get("gpu"), session.get("processor"))
        for run in runs
        for session in run["metadata"].get("sessions", [])
    }
    if len(machines) > 1:
        notices.append(
            "Runs span different hardware/devices; wall-time comparisons are not controlled."
        )
    output.mkdir(parents=True, exist_ok=True)
    final_rows, grouped = [], defaultdict(list)
    evaluation_points = []
    for run in runs:
        cfg = run["config"]
        by_opponent = defaultdict(list)
        for (opponent, mode, step), records in sorted(_evaluation_points(run, split).items()):
            for row in records:
                if row["seat"] not in ("0", "1", "2", "3"):
                    raise ValueError("evaluation has an invalid seat")
            boards = {(r["eval_seed"], r["board"]) for r in records}
            if len(records) != len(boards) * 4:
                raise ValueError(f"evaluation is missing seat rotations: {run['path']}, {step}")
            point = {
                "run": str(run["path"]),
                "group": run["group"],
                "label": run["label"],
                "seed": cfg.seed,
                "gamma": cfg.ppo.gamma,
                "lambda": cfg.ppo.gae_lambda,
                "rollout_steps": cfg.ppo.rollout_steps,
                "envs": cfg.runtime.num_workers * cfg.runtime.envs_per_worker,
                "env_steps": step,
                "training_seconds": float(records[0]["training_seconds"]),
                "wall_seconds": float(records[0]["wall_seconds"]),
                "split": split,
                "opponent": opponent,
                "mode": mode,
                "games": len(records),
                "wins": sum(r["outcome"] == "win" for r in records),
                "win_rate": np.mean([r["outcome"] == "win" for r in records]),
                "truncation_rate": np.mean([r["outcome"] == "truncated" for r in records]),
                "mean_vp": np.mean([float(r["victory_points"]) for r in records]),
                "mean_turns": np.mean([float(r["turns"]) for r in records]),
            }
            if all(all(row.get(field, "") != "" for field in ACTION_FIELDS) for row in records):
                decisions = sum(int(row["learner_decisions"]) for row in records)
                if decisions:
                    point.update(
                        {
                            f"{field}_per_100_decisions": 100
                            * sum(int(r[field]) for r in records)
                            / decisions
                            for field in ACTION_FIELDS
                        }
                    )
            evaluation_points.append(point)
            by_opponent[(opponent, mode)].append((point, records))
        for (opponent, mode), points in by_opponent.items():
            point, records = max(points, key=lambda pair: pair[0]["env_steps"])
            start = (
                next((p for p, _ in points if p["env_steps"] == 0), None)
                if split == "validation"
                else None
            )
            point["start_win_rate"] = start["win_rate"] if start else ""
            point["win_rate_change"] = point["win_rate"] - start["win_rate"] if start else ""
            point["start_mean_vp"] = start["mean_vp"] if start else ""
            point["mean_vp_change"] = point["mean_vp"] - start["mean_vp"] if start else ""
            final_rows.append(point)
            grouped[(run["group"], opponent, mode)].append((point, records))
    summaries = []
    for (group, opponent, mode), pairs in sorted(grouped.items()):
        rates = np.asarray([p[0]["win_rate"] for p in pairs])
        low, high = _cluster_interval([p[1] for p in pairs])
        point = pairs[0][0]
        summaries.append(
            {
                "group": group,
                "label": point["label"],
                "split": split,
                "opponent": opponent,
                "mode": mode,
                "training_seeds": len(pairs),
                "games": sum(p[0]["games"] for p in pairs),
                "mean_win_rate": float(rates.mean()),
                "seed_std": float(rates.std(ddof=1)) if len(rates) > 1 else "",
                "seed_min": float(rates.min()),
                "seed_max": float(rates.max()),
                "cluster_ci_low": low,
                "cluster_ci_high": high,
                "ci_degenerate": int(low == high),
                "mean_vp": float(np.mean([p[0]["mean_vp"] for p in pairs])),
                "mean_win_rate_change": (
                    float(np.mean([p[0]["win_rate_change"] for p in pairs]))
                    if all(p[0]["win_rate_change"] != "" for p in pairs)
                    else ""
                ),
                "mean_vp_change": (
                    float(np.mean([p[0]["mean_vp_change"] for p in pairs]))
                    if all(p[0]["mean_vp_change"] != "" for p in pairs)
                    else ""
                ),
                "mean_truncation_rate": float(np.mean([p[0]["truncation_rate"] for p in pairs])),
                "min_env_steps": min(p[0]["env_steps"] for p in pairs),
                "max_env_steps": max(p[0]["env_steps"] for p in pairs),
                "mean_wall_seconds": float(np.mean([p[0]["wall_seconds"] for p in pairs])),
            }
        )
        if len(pairs) < 3:
            notices.append(f"Fewer than three training seeds: {point['label']} vs {opponent}")
        if len({p[0]["env_steps"] for p in pairs}) > 1:
            notices.append(f"Unequal final evaluation budgets: {point['label']} vs {opponent}")
    summaries.sort(key=lambda row: (row["opponent"], row["mode"], -row["mean_win_rate"]))
    _write_csv(output / "evaluation_points.csv", evaluation_points)
    _write_csv(output / "run_summary.csv", final_rows)
    _write_csv(output / "configuration_summary.csv", summaries)
    _make_figures(runs, evaluation_points, output)
    report = {
        "runs": len(runs),
        "warnings": notices,
        "notes": [
            "Thin curves show individual seeds; shading is the observed seed range, not a CI.",
            "Aggregate curves interpolate only within the common support of the seed curves.",
            "Cluster intervals resample training runs and boards; four seats stay together.",
            "Few seeds and unanimous outcomes limit bootstrap intervals; a zero-width CI does not prove certainty.",
            "Configuration summaries use the last evaluation, not the best validation checkpoint.",
            "Longer rollouts also change collection batch size and policy refresh frequency.",
            "Critic loss/explained variance use lambda-dependent targets, not true state values.",
            "Use separate test boards after selecting configurations on validation results.",
        ],
    }
    write_json(output / "report.json", report)
    _write_report(output / "report.md", summaries, report, split, runs)
    return report


def _write_report(
    path: Path, summaries: list[dict], report: dict, split: str, runs: list[dict]
) -> None:
    """Index the measured results without treating a smoke run as a strength claim."""
    lines = [
        f"# {split.title()} results",
        "",
        f"{report['runs']} training run{'s' if report['runs'] != 1 else ''}. "
        "Win rates use the last evaluation of each run. "
        "Each board is played in all four seats.",
        "",
        "| Configuration | Opponent | Seeds | Games | Win rate | Change from step 0 | Seed range | Mean VP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        change = f"{row['mean_win_rate_change']:+.1%}" if row["mean_win_rate_change"] != "" else "—"
        lines.append(
            f"| {row['label']} | {row['opponent']} ({row['mode']}) | "
            f"{row['training_seeds']} | {row['games']} | {row['mean_win_rate']:.1%} | "
            f"{change} | "
            f"{row['seed_min']:.1%}–{row['seed_max']:.1%} | {row['mean_vp']:.2f} |"
        )
    lines += [
        "",
        "## Run design",
        "",
        "| Name | Policy | Reward | Starting weights | Training opponents |",
        "| --- | --- | --- | --- | --- |",
    ]
    designs = set()
    for run in runs:
        cfg = run["config"]
        model = cfg.model
        policy = ", ".join(
            part
            for part in (
                model.encoder,
                "set attention" if model.set_attention else "",
                f"{model.belief} belief" if model.belief != "none" else "",
                f"{model.trade_decoder} trade decoder" if model.trade_decoder != "none" else "",
            )
            if part
        )
        reward = f"{cfg.env.shaping} VP potential" if cfg.env.shaping != "none" else "terminal"
        start = (
            "checkpoint"
            if run["metadata"].get("sessions", [{}])[0].get("init_from")
            else "random initialization"
        )
        opponents = ", ".join(f"{spec.kind} ({spec.weight:g})" for spec in cfg.opponents.pool)
        designs.add((cfg.name, policy, reward, start, opponents))
    lines.extend(f"| {' | '.join(design)} |" for design in sorted(designs))
    lines += [
        "",
        "## Figures",
        "",
        "- `evaluation_*_env_steps.png`: win rate, victory points and truncated games "
        "against training decisions. The wall-time versions show compute cost.",
        "- `evaluation_strategy_*.png`: actions per 100 learner decisions in held-out "
        "games; offers are attempts, not completed trades.",
        "- `training_diagnostics.png`: PPO health and nonzero reward frequency. "
        "Training episode plots are diagnostics, not held-out performance.",
        "",
        "![PPO diagnostics](training_diagnostics.png)",
        "",
        "## Reading the results",
        "",
        "Thin lines are individual training seeds; shaded bands show their observed range, "
        "not a confidence interval. A small smoke run checks the pipeline only. "
        "Use separate test board seeds after choosing settings on validation games.",
    ]
    if report["warnings"]:
        lines += ["", "## Checks", ""]
        lines.extend(f"- {warning}" for warning in report["warnings"])
    for opponent, mode in sorted({(row["opponent"], row["mode"]) for row in summaries}):
        figure = path.parent / f"evaluation_{opponent}_{mode}_env_steps.png"
        lines += ["", f"### Against {opponent} ({mode})", "", f"![Evaluation]({figure.name})"]
        strategy = path.parent / f"evaluation_strategy_{opponent}_{mode}.png"
        if strategy.exists():
            lines += ["", f"![Decision mix]({strategy.name})"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_figures(runs: list[dict], points: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    metric_labels = {
        "policy_loss": "Policy loss",
        "value_loss": "Value loss",
        "entropy": "Policy entropy",
        "main_type_entropy": "Action type entropy",
        "approx_kl": "Approximate KL",
        "clip_fraction": "Clipped fraction",
        "grad_norm": "Gradient norm",
        "explained_variance": "Explained variance",
        "advantage_std": "Advantage spread",
        "main_end_turn_rate_when_buildable": "End turn when buildable",
        "return_std": "Return spread",
        "decisions_per_second": "Decisions / second",
        "nonzero_reward_fraction": "Decisions with nonzero reward",
        "win_rate": "Win rate",
        "mean_vp": "Mean victory points",
        "truncation_rate": "Truncated games",
    }

    diagnostics = (
        "policy_loss",
        "value_loss",
        "entropy",
        "main_type_entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "explained_variance",
        "advantage_std",
        "main_end_turn_rate_when_buildable",
        "return_std",
        "decisions_per_second",
        "nonzero_reward_fraction",
    )
    available = [
        field
        for field in diagnostics
        if field == "nonzero_reward_fraction"
        or any(row.get(field, "") != "" for run in runs for row in run["updates"])
    ]
    rows_count = (len(available) + 2) // 3
    fig, axes = plt.subplots(rows_count, 3, figsize=(16, rows_count * 3.5), squeeze=False)
    for ax, field in zip(axes.flat, available, strict=False):
        curves = defaultdict(list)
        for run in runs:
            if field == "nonzero_reward_fraction":
                rows = [r for r in run["updates"] if int(r["samples"]) > 0]
                values = [float(r["nonzero_rewards"]) / float(r["samples"]) for r in rows]
            else:
                rows = [r for r in run["updates"] if r.get(field, "") != ""]
                values = [float(r[field]) for r in rows]
            curves[run["label"]].append(([float(r["env_steps"]) for r in rows], values))
        draw_curves(ax, curves, xlabel="Learner decisions", ylabel=metric_labels[field])
        if field in ("nonzero_reward_fraction", "main_end_turn_rate_when_buildable"):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(-0.02, 1.02)
    for ax in list(axes.flat)[len(available) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output / "training_diagnostics.png", dpi=150)
    plt.close(fig)

    head_fields = sorted(
        {
            key
            for run in runs
            for row in run["updates"]
            for key, value in row.items()
            if key.startswith("head_") and value != ""
        }
    )
    if head_fields:
        fig, axes = plt.subplots((len(head_fields) + 2) // 3, 3, figsize=(16, 11), squeeze=False)
        for ax, field in zip(axes.flat, head_fields, strict=False):
            curves = defaultdict(list)
            for run in runs:
                rows = [r for r in run["updates"] if r.get(field, "") != ""]
                curves[run["label"]].append(
                    ([float(r["env_steps"]) for r in rows], [float(r[field]) for r in rows])
                )
            draw_curves(ax, curves, xlabel="Learner decisions", ylabel=field)
        for ax in list(axes.flat)[len(head_fields) :]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", fontsize=7)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(output / "head_entropies.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, field in zip(
        axes.flat, ("return", "victory_points", "turns", "truncated"), strict=True
    ):
        curves = defaultdict(list)
        for run in runs:
            rows = run["episodes"]
            if not rows:
                continue
            values = np.asarray(
                [
                    float(r[field]) if field != "truncated" else float(r["outcome"] == "truncated")
                    for r in rows
                ]
            )
            window = min(25, len(values))
            means = np.convolve(values, np.ones(window) / window, mode="valid")
            curves[run["label"]].append(
                ([float(r["env_steps"]) for r in rows][window - 1 :], means)
            )
        draw_curves(ax, curves, xlabel="Learner decisions", ylabel=f"{field} (up to 25 episodes)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / "training_episodes.png", dpi=150)
    plt.close(fig)

    for opponent, mode in sorted({(p["opponent"], p["mode"]) for p in points}):
        for axis, xlabel in (("env_steps", "Learner decisions"), ("wall_seconds", "Wall seconds")):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, field in zip(axes, ("win_rate", "mean_vp", "truncation_rate"), strict=True):
                by_run = defaultdict(list)
                for point in points:
                    if point["opponent"] == opponent and point["mode"] == mode:
                        by_run[(point["label"], point["run"])].append(point)
                curves = defaultdict(list)
                for (label, _), rows in by_run.items():
                    rows.sort(key=lambda r: r[axis])
                    curves[label].append(([r[axis] for r in rows], [r[field] for r in rows]))
                draw_curves(ax, curves, xlabel=xlabel, ylabel=metric_labels[field])
                if field != "mean_vp":
                    ax.set_ylim(-0.02, 1.02)
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
                else:
                    ax.set_ylim(bottom=0)
            fig.suptitle(f"vs {opponent}, {mode}")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", fontsize=7)
            fig.tight_layout(rect=(0, 0.12, 1, 0.95))
            fig.savefig(output / f"evaluation_{opponent}_{mode}_{axis}.png", dpi=150)
            plt.close(fig)

        strategy = (
            "action_build_road",
            "action_build_settlement",
            "action_build_city",
            "action_buy_dev_card",
            "action_propose_trade",
            "action_multi_card_offer",
        )
        if any(f"{field}_per_100_decisions" in point for field in strategy for point in points):
            local = [
                point for point in points if point["opponent"] == opponent and point["mode"] == mode
            ]
            active = [
                field
                for field in strategy
                if field in ("action_propose_trade", "action_multi_card_offer")
                or any(point.get(f"{field}_per_100_decisions", 0) > 0 for point in local)
            ]
            columns = 2 if len(active) <= 4 else 3
            rows_count = (len(active) + columns - 1) // columns
            fig, axes = plt.subplots(
                rows_count, columns, figsize=(5 * columns, rows_count * 3.5), squeeze=False
            )
            for ax, field in zip(axes.flat, active, strict=False):
                key = f"{field}_per_100_decisions"
                by_run = defaultdict(list)
                for point in points:
                    if point["opponent"] == opponent and point["mode"] == mode and key in point:
                        by_run[(point["label"], point["run"])].append(point)
                curves = defaultdict(list)
                for (label, _), rows in by_run.items():
                    rows.sort(key=lambda r: r["env_steps"])
                    curves[label].append(([r["env_steps"] for r in rows], [r[key] for r in rows]))
                name = field.removeprefix("action_").replace("_", " ")
                draw_curves(
                    ax, curves, xlabel="Learner decisions", ylabel=f"{name} / 100 decisions"
                )
                if not any(point.get(key, 0) > 0 for point in local):
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(bottom=0)
            for ax in list(axes.flat)[len(active) :]:
                ax.set_visible(False)
            fig.suptitle(f"Decision mix vs {opponent} ({mode})")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", fontsize=7)
            fig.tight_layout(rect=(0, 0.10, 1, 0.96))
            fig.savefig(output / f"evaluation_strategy_{opponent}_{mode}.png", dpi=150)
            plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(argv)
    report = analyze_runs(
        args.inputs, args.output, include_incomplete=args.include_incomplete, split=args.split
    )
    print(f"Analyzed {report['runs']} runs; figures and summaries: {args.output}")
    for notice in report["warnings"]:
        print(f"Warning: {notice}")
