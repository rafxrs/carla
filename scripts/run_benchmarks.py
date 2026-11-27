"""Paired-board smoke/benchmark runs with explicit failures and provenance.

Example: python scripts/run_benchmarks.py --opponents native-f catanatron-f --boards 5
Each opponent gets three seats, the CARLA heuristic rotates through all four.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from carla.agents.registry import make_agent
from carla.engine.config import EngineConfig
from carla.engine.rng import derive_seed
from carla.eval.metrics import wilson_interval
from carla.eval.provenance import archive_sources
from carla.eval.runner import play_game
from carla.utils.seeding import fixed_hash_seed, rerun_with_fixed_hash_seed


def run(opponents, boards, seed, output, trading=False, max_turns=300):
    if boards < 1:
        raise ValueError("boards must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = EngineConfig(max_turns=max_turns, allow_domestic_trade=trading)
    manifest = {
        "schema": 1,
        "referee": "CARLA",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "python": platform.python_version(),
        "python_hash_seed": fixed_hash_seed(),
        "platform": platform.platform(),
        "engine": asdict(config),
        "opponents": opponents,
        "agent_conformance": {
            name: "experimental: whole-policy equivalence not established"
            for name in opponents
            if name.startswith("jsettlers-") and name.endswith("-native")
        },
        "boards": boards,
        "seed": seed,
        "failure_policy": "failed games are recorded, never replaced; win fraction uses all scheduled games",
        "information": {
            "catanatron": "public-inventory-sampled-world",
            "jsettlers": "own-exact-others-unknown",
            "native": "public-inventory-sampled-world",
        },
        "dependencies": {},
        "source_hashes": {},
    }
    for package in ("catanatron", "carla"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            manifest["dependencies"][package] = importlib.metadata.version(package)
    for root in (
        Path("src/carla/agents"),
        Path("src/carla/planning"),
        Path("src/carla/engine"),
        Path("src/carla/eval"),
        Path("adapters/jsettlers/src"),
    ):
        for path in sorted(root.rglob("*")):
            if path.suffix in (".py", ".java"):
                manifest["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if jar := os.environ.get("JSETTLERS_JAR"):
        manifest["jsettlers_sha256"] = hashlib.sha256(Path(jar).read_bytes()).hexdigest()
    manifest.update(archive_sources(output))
    manifest["references"] = {
        "catanatron": "ecf931181b9a65bb4116a2153fb78c16f1438e00",
        "jsettlers": "2.6.10",
        "lanzenbrecher": "aabaa1912bd971265109e3eab15f9a340878142a",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (output / "games.jsonl").open("w", encoding="utf-8") as journal:
        for opponent in opponents:
            for board in range(boards):
                game_seed = derive_seed(seed, "paired-board", board)
                for seat in range(4):
                    names = [opponent] * 4
                    names[seat] = "heuristic"
                    row = {"opponent": opponent, "board": board, "seat": seat, "seed": game_seed}
                    start = time.perf_counter()
                    try:
                        result = play_game(
                            [make_agent(n) for n in names], game_seed, config, record=True
                        )
                        row.update(
                            winner=result.winner,
                            heuristic_win=result.winner == seat,
                            vp=result.victory_points[seat],
                            turns=result.turns,
                            steps=result.steps,
                            truncated=result.winner is None,
                            failure=None,
                        )
                        result.record.save(output / f"{opponent}-{board}-{seat}.json.gz")
                    except Exception as error:
                        row.update(failure=f"{type(error).__name__}: {error}", heuristic_win=False)
                    row["seconds"] = time.perf_counter() - start
                    journal.write(json.dumps(row) + "\n")
                    journal.flush()
                    records.append(row)
                    print(json.dumps(row), flush=True)
    summary = {}
    for opponent in opponents:
        rows = [r for r in records if r["opponent"] == opponent]
        wins = sum(r["heuristic_win"] for r in rows)
        failures = sum(r["failure"] is not None for r in rows)
        summary[opponent] = {
            "scheduled": len(rows),
            "failures": failures,
            "wins": wins,
            "truncations": sum(r.get("truncated", False) for r in rows),
            "win_fraction_scheduled": wins / len(rows),
            "wilson_95_descriptive": wilson_interval(wins, len(rows)),
            "mean_vp_completed": (
                sum(r.get("vp", 0) for r in rows) / (len(rows) - failures)
                if failures < len(rows)
                else None
            ),
            "valid_run": failures == 0,
        }
    summary["note"] = (
        "Initial paired-board results; rotations share boards. Wilson intervals are descriptive, not board-cluster-adjusted."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    rerun_with_fixed_hash_seed()  # Catanatron decisions depend on string hashing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["catanatron-f", "native-f", "catanatron-ab2", "native-ab2"],
    )
    parser.add_argument("--boards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5522601)
    parser.add_argument("--output", default="outputs/benchmarks/initial")
    parser.add_argument("--trading", action="store_true")
    parser.add_argument("--max-turns", type=int, default=300)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.opponents, args.boards, args.seed, args.output, args.trading, args.max_turns),
            indent=2,
        )
    )
