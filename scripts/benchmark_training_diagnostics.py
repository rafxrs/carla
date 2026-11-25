"""Paired, reproducible diagnostics on/off benchmarks; never runs the 300k experiments.

Example: uv run python scripts/benchmark_training_diagnostics.py --output runs/training_diagnostics_benchmark.json
All tiny training runs are isolated in a temporary directory and removed afterwards.
"""

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import pickle
import statistics
import sys
import tempfile
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import torch

from carla.agents import RandomAgent, make_agent
from carla.config import load_config, to_dict
from carla.engine.rng import derive_seed
from carla.envs import CatanEnv
from carla.rl.train import train


def deep_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if is_dataclass(value):
        size += sum(deep_size(getattr(value, item.name), seen) for item in fields(value))
    elif isinstance(value, list | tuple):
        size += sum(deep_size(item, seen) for item in value)
    return size


def environment_case(config, enabled, steps, seed):
    env = CatanEnv(
        config.engine,
        config.env,
        [make_agent(kind) for kind in ("heuristic25", "heuristic25", "heuristic50")],
        collect_training_diagnostics=enabled,
    )
    policy, episodes = RandomAgent(), 0
    env.reset(seed=seed)
    policy.reset(env.game, env.learner, derive_seed(seed, "benchmark-policy", episodes))
    gc.collect()
    started = time.perf_counter()
    cpu_started = time.process_time()
    for _ in range(steps):
        action = policy.act(env.game, env.learner, env.game.legal_actions())
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            if enabled:
                env.training_diagnostics.record()  # Include the once-per-episode snapshot cost.
            episodes += 1
            env.reset()
            policy.reset(env.game, env.learner, derive_seed(seed, "benchmark-policy", episodes))
    elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    result = {
        "seconds": elapsed,
        "decisions_per_second": steps / elapsed,
        "cpu_seconds": cpu_seconds,
        "cpu_decisions_per_second": steps / cpu_seconds,
        "episodes": episodes,
        "final_state": env.game.state_hash(),
        "counter_heap_bytes": deep_size(env.training_diagnostics) if enabled else 0,
        "counter_pickle_bytes": len(pickle.dumps(env.training_diagnostics)) if enabled else 0,
    }
    env.close()
    return result


def training_case(config, enabled, output):
    cfg = replace(
        config,
        runtime=replace(
            config.runtime, output_dir=str(output), collect_training_diagnostics=enabled
        ),
    )
    gc.collect()
    started = time.perf_counter()
    cpu_started = time.process_time()
    run = train(cfg)
    elapsed = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    checkpoint = run / "latest.pt"
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name, tensor in data["policy"].items():
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    path = run / "training_diagnostics.csv"
    records = []
    if path.exists():
        with path.open() as stream:
            records = list(csv.DictReader(stream))
    size = path.stat().st_size if path.exists() else 0
    header_bytes = len(path.read_bytes().splitlines(keepends=True)[0]) if path.exists() else 0
    with (run / "updates.csv").open() as stream:
        updates = list(csv.DictReader(stream))
    collection_seconds = sum(float(row["collection_seconds"]) for row in updates)
    optimizer_seconds = sum(float(row["update_seconds"]) for row in updates)
    return {
        "seconds": elapsed,
        "decisions_per_second": cfg.ppo.total_env_steps / elapsed,
        "cpu_seconds": cpu_seconds,
        "cpu_decisions_per_second": cfg.ppo.total_env_steps / cpu_seconds,
        "policy_digest": digest.hexdigest(),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "diagnostic_rows": len(records),
        "diagnostic_csv_bytes": size,
        "diagnostic_header_bytes": header_bytes,
        "bytes_per_episode": (size - header_bytes) / len(records) if records else None,
        "collection_seconds": collection_seconds,
        "optimizer_seconds": optimizer_seconds,
        "steady_state_seconds": collection_seconds + optimizer_seconds,
        "steady_state_decisions_per_second": cfg.ppo.total_env_steps
        / (collection_seconds + optimizer_seconds),
    }


def summary(pairs, rate="decisions_per_second"):
    baseline = statistics.median(pair["disabled"][rate] for pair in pairs)
    enabled = statistics.median(pair["enabled"][rate] for pair in pairs)
    differences = [100 * (pair["enabled"][rate] / pair["disabled"][rate] - 1) for pair in pairs]
    return {
        "disabled_median_decisions_per_second": baseline,
        "enabled_median_decisions_per_second": enabled,
        "median_paired_throughput_difference_percent": statistics.median(differences),
        "paired_throughput_difference_percent": differences,
        "median_paired_elapsed_overhead_percent": statistics.median(
            100 * (pair["disabled"][rate] / pair["enabled"][rate] - 1) for pair in pairs
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-steps", type=int, default=5000)
    parser.add_argument("--training-steps", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu",
        default="auto",
        help="pin to an allowed CPU (auto: first allowed; none: no affinity)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.env_steps <= 0 or args.training_steps <= 0 or args.repeats < 2:
        parser.error("positive budgets and at least two paired repeats are required")
    if args.output is not None and args.output.exists():
        parser.error("output already exists; use a new path")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.cpu != "none" and hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        selected = min(allowed) if args.cpu == "auto" else int(args.cpu)
        if selected not in allowed:
            parser.error("requested CPU is outside this process's allowed affinity")
        os.sched_setaffinity(0, {selected})
    logging.getLogger("carla.rl.train").propagate = False
    base = load_config("configs/experiments/full_trade_strategic_economy_05.yaml")
    config = replace(
        base,
        engine=replace(base.engine, max_turns=40),
        env=replace(base.env, learner_seat=0, max_events=16),
        model=replace(base.model, hidden_dim=64, gnn_layers=2, belief_dim=32),
        ppo=replace(
            base.ppo,
            total_env_steps=args.training_steps,
            rollout_steps=64,
            epochs=1,
            minibatch_size=256,
        ),
        runtime=replace(
            base.runtime,
            num_workers=1,
            envs_per_worker=4,
            torch_threads_per_worker=1,
            device=args.device,
            checkpoint_every=999999,
            log_every=999999,
        ),
    )
    env_pairs, training_pairs = [], []
    with tempfile.TemporaryDirectory(prefix="carla-training-diagnostics-") as temp:
        root = Path(temp)
        # Untimed warmup for imports, kernels, allocator and filesystem caches, in both modes.
        for enabled in (False, True):
            environment_case(config, enabled, min(300, args.env_steps), 999)
            training_case(
                replace(
                    config, ppo=replace(config.ppo, total_env_steps=min(1024, args.training_steps))
                ),
                enabled,
                root / f"warmup-{enabled}",
            )
        for repeat in range(args.repeats):
            cfg = replace(config, seed=100 + repeat)
            pair_env = {"seed": cfg.seed}
            for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                label = "enabled" if enabled else "disabled"
                pair_env[label] = environment_case(cfg, enabled, args.env_steps, cfg.seed)
            assert pair_env["disabled"]["final_state"] == pair_env["enabled"]["final_state"]
            assert pair_env["disabled"]["episodes"] == pair_env["enabled"]["episodes"]
            env_pairs.append(pair_env)
            print(
                f"completed environment pair {repeat + 1}/{args.repeats}",
                file=sys.stderr,
                flush=True,
            )
        for repeat in range(args.repeats):
            cfg = replace(config, seed=100 + repeat)
            pair_train = {"seed": cfg.seed}
            for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                label = "enabled" if enabled else "disabled"
                pair_train[label] = training_case(cfg, enabled, root / f"trial-{repeat}-{label}")
            assert pair_train["disabled"]["policy_digest"] == pair_train["enabled"]["policy_digest"]
            training_pairs.append(pair_train)
            print(
                f"completed training pair {repeat + 1}/{args.repeats}", file=sys.stderr, flush=True
            )
    result = {
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "settings": to_dict(config),
        "env_steps_per_trial": args.env_steps,
        "training_steps_per_trial": args.training_steps,
        "repeats": args.repeats,
        "environment": {
            **summary(env_pairs),
            "cpu_time": summary(env_pairs, "cpu_decisions_per_second"),
            "trials": env_pairs,
        },
        "training": {
            **summary(training_pairs),
            "cpu_time": summary(training_pairs, "cpu_decisions_per_second"),
            "steady_state": summary(training_pairs, "steady_state_decisions_per_second"),
            "trials": training_pairs,
        },
        "median_counter_heap_bytes": statistics.median(
            pair["enabled"]["counter_heap_bytes"] for pair in env_pairs
        ),
        "median_counter_pickle_bytes": statistics.median(
            pair["enabled"]["counter_pickle_bytes"] for pair in env_pairs
        ),
        "median_diagnostic_bytes_per_episode": (
            statistics.median(
                pair["enabled"]["bytes_per_episode"]
                for pair in training_pairs
                if pair["enabled"]["bytes_per_episode"] is not None
            )
            if any(pair["enabled"]["bytes_per_episode"] is not None for pair in training_pairs)
            else None
        ),
        "median_additional_checkpoint_bytes": statistics.median(
            pair["enabled"]["checkpoint_bytes"] - pair["disabled"]["checkpoint_bytes"]
            for pair in training_pairs
        ),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
