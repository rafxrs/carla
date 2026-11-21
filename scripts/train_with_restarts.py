"""Run `carla train` and resume from latest.pt whenever the process dies (e.g. a GPU reset).

Run from the repository root:

    python scripts/train_with_restarts.py configs/experiments/full_trade.yaml \
        runtime.output_dir=runs/full_trade [more overrides]

`runtime.output_dir` is required; the newest run directory under it is resumed with the same
overrides. Stops after `--max-restarts` failures in a row without progress.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _newest_run(output_dir: Path, after: float) -> Path | None:
    runs = [p for p in output_dir.iterdir() if p.is_dir() and p.stat().st_mtime >= after]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and resume after crashes.")
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--init-from", help="first launch only: start from these policy weights")
    parser.add_argument("--resume", help="continue an existing run directory (or its latest.pt)")
    args = parser.parse_args()
    output = [o.split("=", 1)[1] for o in args.overrides if o.startswith("runtime.output_dir=")]
    if not output:
        parser.error("pass runtime.output_dir=... so the run directory can be found again")
    output_dir = Path(output[-1])
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "carla.cli", "train", args.config]
    started = time.time()
    run_dir: Path | None = None
    if args.resume:
        run_dir = Path(args.resume)
        run_dir = run_dir.parent if run_dir.is_file() else run_dir
        if not (run_dir / "latest.pt").exists():
            parser.error(f"no latest.pt in {run_dir}")
    failures, last_size = 0, -1
    while True:
        latest = run_dir / "latest.pt" if run_dir is not None else None
        if latest is not None and latest.exists():
            resume = ["--resume", str(latest)]
        else:
            resume = ["--init-from", args.init_from] if args.init_from else []
        code = subprocess.call([*command, *resume, *args.overrides])
        if code == 0:
            return 0
        run_dir = run_dir or _newest_run(output_dir, started)
        progress = (run_dir / "updates.csv").stat().st_size if run_dir else 0
        failures = 0 if progress > last_size else failures + 1
        last_size = progress
        print(f"train exited with code {code}; restart {failures}/{args.max_restarts}", flush=True)
        if failures >= args.max_restarts:
            return code
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
