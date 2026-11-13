"""Sequential, restartable gamma/lambda/rollout sweeps; one job owns the device."""

import argparse
import json
import logging
from itertools import product
from pathlib import Path

from carla.config import config_hash, load_config, to_dict
from carla.rl.train import _validate, train
from carla.utils.logging import write_json
from carla.utils.runs import create_run_dir


def run_sweep(
    config_path: Path,
    output: Path,
    *,
    gammas=(0.999,),
    lambdas=(0.95, 0.99, 1.0),
    rollout_steps=(256, 1024),
    seeds=(0, 1, 2),
    overrides=(),
    resume: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    output = output.resolve()
    if any(
        not choices or len(set(choices)) != len(choices)
        for choices in (gammas, lambdas, rollout_steps, seeds)
    ):
        raise ValueError("sweep dimensions must be nonempty and contain no duplicates")
    cases, configs = [], []
    base_gamma = load_config(config_path, overrides).ppo.gamma
    for gamma, lam, steps, seed in product(gammas, lambdas, rollout_steps, seeds):
        cfg = load_config(
            config_path,
            [
                *overrides,
                f"ppo.gamma={gamma}",
                f"ppo.gae_lambda={lam}",
                f"ppo.rollout_steps={steps}",
                f"seed={seed}",
            ],
        )
        _validate(cfg)
        canonical = to_dict(cfg)
        canonical["runtime"]["output_dir"] = "<sweep>"
        gamma_tag = "" if gamma == base_gamma else f"gamma-{gamma:g}-"
        case_id = f"{gamma_tag}lambda-{lam:g}-rollout-{steps}-seed-{seed}-{config_hash(canonical)}"
        cfg = load_config(
            config_path,
            [
                *overrides,
                f"ppo.gamma={gamma}",
                f"ppo.gae_lambda={lam}",
                f"ppo.rollout_steps={steps}",
                f"seed={seed}",
                f"runtime.output_dir={json.dumps(str(output / case_id))}",
            ],
        )
        cases.append({"id": case_id, "config": canonical, "status": "planned", "run": None})
        configs.append(cfg)
    if dry_run:
        for case in cases:
            print(case["id"])
        print(f"{len(cases)} runs; output: {output}")
        return []
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("sweep already exists; use --resume or a new output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if [(c["id"], c["config"]) for c in manifest["cases"]] != [
            (c["id"], c["config"]) for c in cases
        ]:
            raise ValueError("sweep plan differs from the existing manifest")
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 1, "source_config": str(config_path), "cases": cases}
        write_json(manifest_path, manifest)
    results = []
    for case, cfg in zip(manifest["cases"], configs, strict=True):
        if case["status"] == "complete":
            completed_run = output / case["run"]
            if not (completed_run / "latest.pt").exists():
                raise ValueError(f"completed sweep case is missing its checkpoint: {completed_run}")
            results.append(completed_run)
            continue
        run_dir = output / case["run"] if case["run"] else create_run_dir(cfg)
        # Store relative paths so a sweep directory can be moved between laptops.
        case["run"] = run_dir.relative_to(output).as_posix()
        case["status"] = "running"
        write_json(manifest_path, manifest)
        checkpoint = run_dir / "latest.pt"
        try:
            train(cfg, resume=checkpoint if checkpoint.exists() else None, run_dir=run_dir)
        except BaseException as exc:
            case["status"], case["error"] = "failed", str(exc)
            write_json(manifest_path, manifest)
            raise
        case["status"] = "complete"
        case.pop("error", None)
        write_json(manifest_path, manifest)
        results.append(run_dir)
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.999])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.95, 0.99, 1.0])
    parser.add_argument("--rollout-steps", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--set", dest="overrides", nargs="+", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_sweep(
        args.config,
        args.output,
        gammas=args.gammas,
        lambdas=args.lambdas,
        rollout_steps=args.rollout_steps,
        seeds=args.seeds,
        overrides=args.overrides,
        resume=args.resume,
        dry_run=args.dry_run,
    )
