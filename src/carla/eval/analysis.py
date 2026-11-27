"""Validate run journals and produce seed-aware CSV summaries and static figures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from carla.config import config_hash, load_config, to_dict
from carla.eval.plots import draw_curves
from carla.eval.style import REPORT_WIDTH, moving_average, save_report_figure, use_report_style
from carla.rl.metrics import PHASE_ACTION_FIELDS
from carla.utils.logging import write_json
from carla.utils.runs import run_lineage_start


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
    elif any(row.get("main_decisions", "") == "" for row in evaluation):
        notices.append(f"Older evaluation rows lack phase-specific action counts: {path}")
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
        "lineage_start": run_lineage_start(path),
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
    inputs: list[Path],
    output: Path,
    *,
    include_incomplete: bool = False,
    split: str = "validation",
    lineage: dict[str, int] | None = None,
    smooth_updates: int = 25,
) -> dict:
    if smooth_updates <= 0:
        raise ValueError("smooth_updates must be positive")
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
        supplied = (lineage or {}).get(str(path), (lineage or {}).get(path.name))
        if supplied is not None:
            if int(supplied) < 0:
                raise ValueError(f"lineage offset must be nonnegative: {path}")
            run["lineage_start"] = int(supplied)
        if run["lineage_start"] is None:
            raise ValueError(
                f"cannot locate the parent checkpoint for {path}; supply --lineage "
                "with this run's starting decision count"
            )
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
                "global_env_steps": run["lineage_start"] + step,
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
            if all(
                all(row.get(field, "") != "" for field in PHASE_ACTION_FIELDS) for row in records
            ):
                decisions = sum(int(row["main_decisions"]) for row in records)
                if decisions:
                    point.update(
                        {
                            f"{field}_per_100_main_decisions": 100
                            * sum(int(r[field]) for r in records)
                            / decisions
                            for field in PHASE_ACTION_FIELDS
                            if field.startswith("main_") and field != "main_decisions"
                        }
                    )
                point["main_decisions"] = decisions
                for field in ("road_building_roads", "counteroffers", "multi_card_counteroffers"):
                    point[f"{field}_per_game"] = sum(int(r[field]) for r in records) / len(records)
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
                "min_global_env_steps": min(p[0]["global_env_steps"] for p in pairs),
                "max_global_env_steps": max(p[0]["global_env_steps"] for p in pairs),
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
    _make_figures(runs, evaluation_points, output, smooth_updates)
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
            f"PPO diagnostic curves use a trailing mean of {smooth_updates} updates; raw values remain in updates.csv.",
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
        "| Configuration | Opponent | Seeds | Games | Win rate | Change within stage | Seed range | Mean VP |",
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
    lines += ["", "## Run design", ""]
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
        reward = (
            f"{cfg.env.shaping} VP potential (coef {cfg.env.shaping_coef:g}, "
            f"gamma {cfg.env.shaping_gamma:g})"
            if cfg.env.shaping != "none"
            else "terminal reward"
        )
        if cfg.env.shaping == "production":
            reward += f", production weight {cfg.env.production_weight:g}"
        if cfg.env.vp_reward:
            reward += f", {cfg.env.vp_reward:g} per public VP"
        source = run["metadata"].get("sessions", [{}])[0].get("init_from")
        start = (
            f"checkpoint at {run['lineage_start']:,} cumulative decisions"
            if source
            else "random initialization"
        )
        opponent_parts = []
        for spec in cfg.opponents.pool:
            name = spec.kind
            if spec.kind == "checkpoint" and spec.path:
                name = spec.name or Path(spec.path).stem
            opponent_parts.append(f"{name} ({spec.weight:g})")
        opponents = ", ".join(opponent_parts)
        if any(spec.kind == "snapshot" for spec in cfg.opponents.pool):
            opponents += f"; snapshots every {cfg.opponents.snapshot_every} updates, max {cfg.opponents.max_snapshots}"
        kl = (
            f"KL early-stop target {cfg.ppo.target_kl:g}"
            if cfg.ppo.target_kl is not None
            else "no KL early-stop target"
        )
        if cfg.ppo.reference_kl_coef:
            kl += f"; reference action-type KL penalty {cfg.ppo.reference_kl_coef:g}"
            if cfg.ppo.reference_checkpoint:
                kl += f" toward {Path(cfg.ppo.reference_checkpoint).stem}"
        else:
            kl += "; no reference KL penalty"
        designs.add((cfg.name, policy, reward, start, opponents, kl))
    lines.extend(
        f"- **{name}:** {policy}; {reward}; {start}; opponents: {opponents}; {kl}."
        for name, policy, reward, start, opponents, kl in sorted(designs)
    )
    lines += [
        "",
        "## Figures",
        "",
        "- Evaluation figures show win rate, VP and truncation against cumulative learner "
        "decisions. Wall-time versions show each run's own elapsed time.",
        "- Strategy figures count only MAIN-phase decisions. Trade figures separate new offers, "
        "counteroffers and free Road Building roads; offers are attempts, not completed trades.",
        "- PPO diagnostics use a trailing update mean. Raw per-update values, including throughput "
        "and reward density, remain in `updates.csv`. Training episodes are diagnostics, not held-out results.",
        "- Every figure is saved as PNG and PDF at the report width.",
        "",
        "![PPO diagnostics](training_diagnostics.png)",
        "[PDF](training_diagnostics.pdf)",
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
    if any(any(row.get("reference_kl", "") != "" for row in run["updates"]) for run in runs):
        lines += [
            "",
            "### Reference policy KL",
            "",
            "![Reference KL](reference_kl.png) [PDF](reference_kl.pdf)",
        ]
    for opponent, mode in sorted({(row["opponent"], row["mode"]) for row in summaries}):
        figure = path.parent / f"evaluation_{opponent}_{mode}_env_steps.png"
        lines += [
            "",
            f"### Against {opponent} ({mode})",
            "",
            f"![Evaluation]({figure.name}) [PDF]({figure.with_suffix('.pdf').name})",
        ]
        for kind in ("strategy", "trade"):
            figure = path.parent / f"evaluation_{kind}_{opponent}_{mode}.png"
            if figure.exists():
                lines += [
                    "",
                    f"![{kind.title()} choices]({figure.name}) "
                    f"[PDF]({figure.with_suffix('.pdf').name})",
                ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_figures(runs: list[dict], points: list[dict], output: Path, smooth_updates: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    use_report_style()
    metric_labels = {
        "value_loss": "Value loss",
        "entropy": "Policy entropy",
        "approx_kl": "Approximate KL",
        "clip_fraction": "Clipped fraction",
        "win_rate": "Win rate",
        "mean_vp": "Mean victory points",
        "truncation_rate": "Truncated games",
    }

    diagnostics = ("value_loss", "approx_kl", "entropy", "clip_fraction")
    fig, axes = plt.subplots(2, 2, figsize=(REPORT_WIDTH, 5.2))
    for ax, field in zip(axes.flat, diagnostics, strict=True):
        curves = defaultdict(list)
        for run in runs:
            rows = [r for r in run["updates"] if r.get(field, "") != ""]
            values = moving_average([float(r[field]) for r in rows], smooth_updates)
            curves[run["label"]].append(
                ([run["lineage_start"] + float(r["env_steps"]) for r in rows], values)
            )
        draw_curves(ax, curves, xlabel="Cumulative learner decisions", ylabel=metric_labels[field])
        if field == "clip_fraction":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(bottom=0)
        if field == "approx_kl":
            limits = {run["config"].ppo.target_kl for run in runs}
            if len(limits) == 1 and None not in limits:
                ax.axhline(limits.pop(), color="#666666", linestyle="--", linewidth=1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"PPO diagnostics (trailing {smooth_updates}-update mean)")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    save_report_figure(fig, output / "training_diagnostics.png")
    plt.close(fig)

    if any(any(row.get("reference_kl", "") != "" for row in run["updates"]) for run in runs):
        fig, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 3.1))
        for ax, weighted in zip(axes, (False, True), strict=True):
            curves = defaultdict(list)
            for run in runs:
                rows = [row for row in run["updates"] if row.get("reference_kl", "") != ""]
                if not rows:
                    continue
                scale = run["config"].ppo.reference_kl_coef if weighted else 1.0
                values = moving_average(
                    [float(row["reference_kl"]) * scale for row in rows], smooth_updates
                )
                curves[run["label"]].append(
                    ([run["lineage_start"] + float(row["env_steps"]) for row in rows], values)
                )
            draw_curves(
                ax,
                curves,
                xlabel="Cumulative learner decisions",
                ylabel="Weighted loss term" if weighted else "Reference action-type KL",
            )
            ax.set_ylim(bottom=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        save_report_figure(fig, output / "reference_kl.png")
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
        rows_count = (len(head_fields) + 1) // 2
        fig, axes = plt.subplots(
            rows_count, 2, figsize=(REPORT_WIDTH, rows_count * 2.3), squeeze=False
        )
        for ax, field in zip(axes.flat, head_fields, strict=False):
            curves = defaultdict(list)
            for run in runs:
                rows = [r for r in run["updates"] if r.get(field, "") != ""]
                curves[run["label"]].append(
                    (
                        [run["lineage_start"] + float(r["env_steps"]) for r in rows],
                        moving_average([float(r[field]) for r in rows], smooth_updates),
                    )
                )
            draw_curves(ax, curves, xlabel="Cumulative learner decisions", ylabel=field)
        for ax in list(axes.flat)[len(head_fields) :]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        save_report_figure(fig, output / "head_entropies.png")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(REPORT_WIDTH, 5.2))
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
                (
                    [run["lineage_start"] + float(r["env_steps"]) for r in rows][window - 1 :],
                    means,
                )
            )
        draw_curves(
            ax, curves, xlabel="Cumulative learner decisions", ylabel=f"{field} (up to 25 episodes)"
        )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_report_figure(fig, output / "training_episodes.png")
    plt.close(fig)

    for opponent, mode in sorted({(p["opponent"], p["mode"]) for p in points}):
        for axis, filename_axis, xlabel in (
            ("global_env_steps", "env_steps", "Cumulative learner decisions"),
            ("wall_seconds", "wall_seconds", "Run wall seconds"),
        ):
            fig, axes = plt.subplot_mosaic(
                [["win_rate", "mean_vp"], ["truncation_rate", "truncation_rate"]],
                figsize=(REPORT_WIDTH, 5.2),
            )
            for field, ax in axes.items():
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
            fig.suptitle(f"Evaluation vs {opponent} ({mode})")
            handles, labels = axes["win_rate"].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=2)
            fig.tight_layout(rect=(0, 0.10, 1, 0.95))
            save_report_figure(fig, output / f"evaluation_{opponent}_{mode}_{filename_axis}.png")
            plt.close(fig)

        local = [
            point for point in points if point["opponent"] == opponent and point["mode"] == mode
        ]
        if any("main_build_road_per_100_main_decisions" in point for point in local):
            panels = {
                "strategy": (
                    (
                        "main_build_road_per_100_main_decisions",
                        "Road builds",
                        "per 100 main decisions",
                    ),
                    (
                        "main_build_settlement_per_100_main_decisions",
                        "Settlement builds",
                        "per 100 main decisions",
                    ),
                    (
                        "main_build_city_per_100_main_decisions",
                        "City upgrades",
                        "per 100 main decisions",
                    ),
                    (
                        "main_buy_dev_card_per_100_main_decisions",
                        "Development cards",
                        "per 100 main decisions",
                    ),
                ),
                "trade": (
                    (
                        "main_propose_trade_per_100_main_decisions",
                        "New offers",
                        "per 100 main decisions",
                    ),
                    ("counteroffers_per_game", "Counteroffers", "per game"),
                    (
                        "main_multi_card_offer_per_100_main_decisions",
                        "Multi-card offers",
                        "per 100 main decisions",
                    ),
                    ("road_building_roads_per_game", "Free Road Building roads", "per game"),
                ),
            }
            for kind, fields in panels.items():
                fig, axes = plt.subplots(2, 2, figsize=(REPORT_WIDTH, 5.2))
                for index, (ax, (key, title, unit)) in enumerate(
                    zip(axes.flat, fields, strict=True)
                ):
                    by_run = defaultdict(list)
                    for point in local:
                        if key in point:
                            by_run[(point["label"], point["run"])].append(point)
                    curves = defaultdict(list)
                    for (label, _), rows in by_run.items():
                        rows.sort(key=lambda r: r["global_env_steps"])
                        curves[label].append(
                            ([r["global_env_steps"] for r in rows], [r[key] for r in rows])
                        )
                    draw_curves(
                        ax,
                        curves,
                        xlabel="Cumulative learner decisions",
                        ylabel=unit if index % 2 == 0 else "",
                    )
                    ax.set_title(f"{title} / game" if index % 2 and kind == "trade" else title)
                    ax.set_ylim(0, max(1, ax.get_ylim()[1]))
                fig.suptitle(f"{kind.title()} choices vs {opponent} ({mode})")
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2)
                fig.tight_layout(rect=(0, 0.10, 1, 0.96))
                save_report_figure(fig, output / f"evaluation_{kind}_{opponent}_{mode}.png")
                plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--lineage",
        type=Path,
        help="JSON mapping old run folder names to global starting decisions",
    )
    args = parser.parse_args(argv)
    lineage = json.loads(args.lineage.read_text(encoding="utf-8")) if args.lineage else None
    if lineage is not None and not isinstance(lineage, dict):
        parser.error("--lineage must contain a JSON object")
    report = analyze_runs(
        args.inputs,
        args.output,
        include_incomplete=args.include_incomplete,
        split=args.split,
        lineage=lineage,
    )
    print(f"Analyzed {report['runs']} runs; figures and summaries: {args.output}")
    for notice in report["warnings"]:
        print(f"Warning: {notice}")
