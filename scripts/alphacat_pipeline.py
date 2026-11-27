"""Run the whole AlphaCat pipeline unattended and resumably (docs/alphacat.md).

Stages (each skipped when its outputs exist, so rerunning the same command continues):
  1. strategy behaviour cloning of the three teachers -> <root>/bc/bc.pt (+ z library)
  2. league training (main agent, main exploiter, league exploiter), resumed from
     league_state.pt after any crash, up to --restarts times
  3. final evaluation -> <root>/final/: league report, robustness matrices of the main agent,
     the behaviour clone and the earlier full-model checkpoint on the same tables, and the
     main agent's behaviour under every strategy variant

Usage (from the repository root, inside the venv or with `uv run --no-sync`):
  python scripts/alphacat_pipeline.py
  python scripts/alphacat_pipeline.py --root runs/alphacat --league-steps 12000000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASELINE = (
    "outputs/full_trade_production_heuristic-20261124-210028-ce6fcc76-20261126T211130Z-1-001/"
    "full_trade_production_heuristic-20261124-210028-ce6fcc76/checkpoints/step_000007003136.pt"
)


def run(command: list[str], log: Path) -> int:
    print(time.strftime("%H:%M:%S"), " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as stream:
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)


def carla(*args: str) -> list[str]:
    return [sys.executable, "-m", "carla.cli", *args]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("runs/alphacat"))
    parser.add_argument("--base", default="configs/experiments/alphacat.yaml")
    parser.add_argument("--league", default="configs/experiments/alphacat_league.yaml")
    parser.add_argument("--bc-games", type=int, default=2000)
    parser.add_argument("--bc-epochs", type=int, default=16)
    parser.add_argument("--league-steps", type=int, default=12_000_000)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--eval-boards", type=int, default=25, help="robustness boards per table")
    parser.add_argument("--bc-eval-boards", type=int, default=5)
    parser.add_argument("--profile-games", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--baseline", default=BASELINE, help="earlier checkpoint for comparison")
    parser.add_argument("overrides", nargs="*", help="extra dotted overrides for the league")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    log = root / "pipeline_commands.log"

    bc = root / "bc" / "bc.pt"
    if not bc.exists():
        code = run(
            carla(
                "strategy-bc",
                args.base,
                "--output",
                str(root / "bc"),
                "--games",
                str(args.bc_games),
                "--epochs",
                str(args.bc_epochs),
                "--workers",
                str(args.workers),
                "--eval-boards",
                str(args.bc_eval_boards),
                "runtime.device=auto",
            ),
            log,
        )
        if code or not bc.exists():
            print("behaviour cloning failed; see", log, flush=True)
            return 1

    overrides = [
        f"league.init_checkpoint={bc}",
        f"ppo.reference_checkpoint={bc}",
        f"league.total_env_steps={args.league_steps}",
        f"runtime.num_workers={args.workers}",
        f"runtime.output_dir={root / 'league'}",
        *args.overrides,
    ]
    marker = root / "league_run.txt"
    for attempt in range(args.restarts + 1):
        run_dir = Path(marker.read_text().strip()) if marker.exists() else None
        if run_dir is not None and (run_dir / "metadata.json").exists():
            status = json.loads((run_dir / "metadata.json").read_text()).get("status")
            if status == "complete":
                break
        if run_dir is not None and (run_dir / "league_state.pt").exists():
            code = run(carla("league", args.league, "--resume", str(run_dir), *overrides), log)
        else:
            before = set((root / "league").glob("*")) if (root / "league").exists() else set()
            code = run(carla("league", args.league, *overrides), log)
            created = sorted(set((root / "league").glob("*")) - before)
            if created:
                marker.write_text(str(created[-1]))
        if code == 0:
            break
        print(f"league exited with {code}; restart {attempt + 1}/{args.restarts}", flush=True)
        time.sleep(30)
    run_dir = Path(marker.read_text().strip())
    final = root / "final"
    final.mkdir(exist_ok=True)
    with (final / "league_report.txt").open("w", encoding="utf-8") as stream:
        subprocess.call(
            carla("league-report", str(run_dir)), stdout=stream, stderr=subprocess.STDOUT
        )
    main_agent = run_dir / "learners" / "main_0.pt"
    tables = [
        "heuristic,heuristic,heuristic",
        "heuristic_city,heuristic_city,heuristic_city",
        "heuristic_expansion,heuristic_expansion,heuristic_expansion",
        "heuristic_devcard,heuristic_devcard,heuristic_devcard",
        "heuristic_city,heuristic_expansion,heuristic_devcard",
    ]
    if Path(args.baseline).exists():
        tables.append(f"ppo-sampled:{args.baseline},heuristic,heuristic_city")
    subjects = {
        "main_select": f"ppo-sampled:{main_agent}@select",
        "main_none": f"ppo-sampled:{main_agent}@none",
        "bc_select": f"ppo-sampled:{bc}@select",
    }
    if Path(args.baseline).exists():
        subjects["baseline"] = f"ppo-sampled:{args.baseline}"
    for name, subject in subjects.items():
        command = carla(
            "robustness",
            subject,
            "--boards",
            str(args.eval_boards),
            "--workers",
            str(args.workers),
            "--config",
            args.base,
            "--json",
            str(final / f"robustness_{name}.json"),
            "--csv",
            str(final / f"robustness_{name}.csv"),
        )
        for table in tables:
            command += ["--table", table]
        with (final / f"robustness_{name}.txt").open("w", encoding="utf-8") as stream:
            subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
    for variant in ("city", "expansion", "devcard", "none", "select"):
        with (final / f"profile_{variant}.txt").open("w", encoding="utf-8") as stream:
            subprocess.call(
                carla(
                    "profile",
                    f"ppo-sampled:{main_agent}@{variant}",
                    "heuristic",
                    "heuristic_city",
                    "heuristic_expansion",
                    "--games",
                    str(args.profile_games),
                    "--config",
                    args.base,
                    "--csv",
                    str(final / f"profile_{variant}.csv"),
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    print("pipeline finished; results in", final, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
