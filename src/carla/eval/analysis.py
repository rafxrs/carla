"""Validate run journals and produce seed-aware CSV summaries and static figures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from carla.config import config_hash, load_config, to_dict
from carla.eval.plots import draw_curves
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
    canonical.pop("name")
    for key in ("output_dir", "checkpoint_every", "log_every"):
        canonical["runtime"].pop(key)
    identity = config_hash(canonical)
    label = (
        f"g={config.ppo.gamma:g}, l={config.ppo.gae_lambda:g}, "
        f"T={config.ppo.rollout_steps}, N={config.runtime.num_workers * config.runtime.envs_per_worker} [{identity}]"
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
        writer = csv.DictWriter(stream, list(rows[0]))
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
            evaluation_points.append(point)
            by_opponent[(opponent, mode)].append((point, records))
        for (opponent, mode), points in by_opponent.items():
            point, records = max(points, key=lambda pair: pair[0]["env_steps"])
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
    return report


def _make_figures(runs: list[dict], points: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    )
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    for ax, field in zip(axes.flat, diagnostics, strict=True):
        curves = defaultdict(list)
        for run in runs:
            rows = [r for r in run["updates"] if r.get(field, "") != ""]
            curves[run["label"]].append(
                ([float(r["env_steps"]) for r in rows], [float(r[field]) for r in rows])
            )
        draw_curves(ax, curves, xlabel="Learner decisions", ylabel=field)
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
                draw_curves(ax, curves, xlabel=xlabel, ylabel=field)
                if field != "mean_vp":
                    ax.set_ylim(-0.02, 1.02)
            fig.suptitle(f"vs {opponent}, {mode}")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", fontsize=7)
            fig.tight_layout(rect=(0, 0.12, 1, 0.95))
            fig.savefig(output / f"evaluation_{opponent}_{mode}_{axis}.png", dpi=150)
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
