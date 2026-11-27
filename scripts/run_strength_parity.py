"""Port-versus-wrapper strength parity, measured with the CARLA heuristic.

Each condition seats the heuristic against three copies of one bot, on CARLA's referee. Every
board is played with the heuristic in all four seats, and port and wrapper conditions use the
same game seeds, so their games are paired by (board, seat). Two questions are answered:

1. How strong is the heuristic against each bot? Win rate with a board-clustered bootstrap
   interval; 0.25 is parity in a four-player game.
2. Is each port as strong as its wrapper? The paired difference in the heuristic's win rate
   (wrapper minus port), with a board-clustered interval and an exact McNemar test. A pair is
   `equivalent` when the interval lies inside +/- margin, `different` when it excludes zero,
   and `inconclusive` otherwise.

Pairs: catanatron-f / native-f-upstream, catanatron-ab2 / native-ab2-upstream (no domestic
trading; Catanatron never trades), and jsettlers-{fast,smart}-adapted / -native (with and
without domestic trading; native in `java-adapter` mode by default, the wrapper's own boundary
policy). Failed games are recorded and excluded from pairing. `--retry-failures` replays them
after a fix: the failure stays in the journal, the retry is appended, and analysis uses each
game's latest row.

Example (about 50 minutes on 16 cores):
    python scripts/run_strength_parity.py --boards 160 --output outputs/benchmarks/parity
    python scripts/run_strength_parity.py --retry-failures --output outputs/benchmarks/parity
    python scripts/run_strength_parity.py --analyze-only --output outputs/benchmarks/parity
"""

import argparse
import json
import math
import os
import platform
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PAIRS = {
    "catanatron-f": ("catanatron-f", "native-f-upstream", (False,)),
    "catanatron-ab2": ("catanatron-ab2", "native-ab2-upstream", (False,)),
    "jsettlers-fast": ("jsettlers-fast-adapted", "jsettlers-fast-native", (False, True)),
    "jsettlers-smart": ("jsettlers-smart-adapted", "jsettlers-smart-native", (False, True)),
}
PROCESS_BACKED = ("jsettlers-fast-adapted", "jsettlers-smart-adapted")


def conditions(pairs, trading_modes):
    out = []
    for pair in pairs:
        wrapper, port, modes = PAIRS[pair]
        for trading in modes:
            if len(modes) > 1 and trading_modes != "both" and trading != (trading_modes == "on"):
                continue
            for role, name in (("wrapper", wrapper), ("port", port)):
                out.append({"pair": pair, "role": role, "bot": name, "trading": trading})
    return out


def condition_key(c):
    return f"{c['bot']}|trading={int(c['trading'])}"


def play_one(condition, board, seat, game_seed, max_turns, jsettlers_mode):
    """One game: the heuristic in `seat`, three copies of the condition's bot elsewhere."""
    from carla.agents.api import AgentSpec
    from carla.agents.registry import make_agent
    from carla.engine.config import EngineConfig
    from carla.eval.runner import play_game

    name = condition["bot"]
    parameters = (
        {"compatibility": jsettlers_mode}
        if name.startswith("jsettlers-") and name.endswith("-native")
        else {}
    )
    row = {
        "condition": condition_key(condition),
        **condition,
        "board": board,
        "seat": seat,
        "seed": game_seed,
        "failure": None,
    }
    started = time.perf_counter()
    try:
        agents = [make_agent(AgentSpec(name, parameters)) for _ in range(3)]
        agents.insert(seat, make_agent("heuristic"))
        config = EngineConfig(max_turns=max_turns, allow_domestic_trade=condition["trading"])
        result = play_game(agents, game_seed, config)
        row.update(
            winner=result.winner,
            heuristic_win=result.winner == seat,
            truncated=result.winner is None,
            heuristic_vp=result.victory_points[seat],
            bot_vp=max(v for p, v in enumerate(result.victory_points) if p != seat),
            turns=result.turns,
            trade_offers=sum(result.trade_offers),
        )
    except Exception as error:
        row["failure"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc(limit=8)
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


# ------------------------------------------------------------------------------------ analysis


def bootstrap(by_board, statistic, rng, draws=2000):
    """Percentile interval of `statistic` over boards resampled with replacement."""
    boards = sorted(by_board)
    values = []
    for _ in range(draws):
        sample = [by_board[rng.choice(boards)] for _ in boards]
        values.append(statistic([x for group in sample for x in group]))
    values.sort()
    return values[int(0.025 * draws)], values[int(0.975 * draws) - 1]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def latest(rows):
    """Each game's last journaled row, and how many games a retry turned from failed to done."""
    last, failed_before = {}, set()
    for r in rows:
        key = (r["condition"], r["board"], r["seat"])
        if key in last and last[key]["failure"] is not None:
            failed_before.add(key)
        last[key] = r
    recovered = sum(last[k]["failure"] is None for k in failed_before)
    return list(last.values()), recovered


def analyze(rows, margin, seed=0, min_boards=30):
    rng = random.Random(seed)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    rows, recovered = latest(rows)
    report = {
        "conditions": {},
        "pairs": {},
        "margin": margin,
        "min_boards": min_boards,
        "retried_failures_recovered": recovered,
    }
    by_condition = {}
    for r in rows:
        by_condition.setdefault(r["condition"], []).append(r)
    for key, group in sorted(by_condition.items()):
        done = [r for r in group if r["failure"] is None]
        boards = {}
        for r in done:
            boards.setdefault(r["board"], []).append(float(r["heuristic_win"]))
        wins = sum(r["heuristic_win"] for r in done)
        report["conditions"][key] = {
            "scheduled": len(group),
            "failures": len(group) - len(done),
            "truncations": sum(r["truncated"] for r in done),
            "heuristic_wins": wins,
            "heuristic_win_rate": wins / len(done) if done else None,
            "heuristic_win_rate_ci95": bootstrap(boards, mean, rng) if boards else None,
            "heuristic_mean_vp": mean([r["heuristic_vp"] for r in done]),
            "best_bot_mean_vp": mean([r["bot_vp"] for r in done]),
            "mean_seconds": mean([r["seconds"] for r in group]),
        }
    for pair, (wrapper, port, modes) in PAIRS.items():
        for trading in modes:
            w_key, p_key = f"{wrapper}|trading={int(trading)}", f"{port}|trading={int(trading)}"
            if w_key not in by_condition or p_key not in by_condition:
                continue
            w = {(r["board"], r["seat"]): r for r in by_condition[w_key] if r["failure"] is None}
            p = {(r["board"], r["seat"]): r for r in by_condition[p_key] if r["failure"] is None}
            common = sorted(set(w) & set(p))
            if not common:
                continue
            diffs = {}
            for game in common:
                d = float(w[game]["heuristic_win"]) - float(p[game]["heuristic_win"])
                diffs.setdefault(game[0], []).append(d)
            b = sum(w[g]["heuristic_win"] and not p[g]["heuristic_win"] for g in common)
            c = sum(p[g]["heuristic_win"] and not w[g]["heuristic_win"] for g in common)
            low, high = bootstrap(diffs, mean, rng)
            if len(diffs) < min_boards:
                verdict = "insufficient boards"  # a bootstrap over few boards is unreliable
            elif -margin < low and high < margin:
                verdict = "equivalent"
            elif low > 0 or high < 0:
                verdict = "different"
            else:
                verdict = "inconclusive"
            report["pairs"][f"{pair}|trading={int(trading)}"] = {
                "paired_games": len(common),
                "heuristic_win_rate_vs_wrapper": mean(
                    [float(w[g]["heuristic_win"]) for g in common]
                ),
                "heuristic_win_rate_vs_port": mean([float(p[g]["heuristic_win"]) for g in common]),
                "difference_wrapper_minus_port": mean([d for ds in diffs.values() for d in ds]),
                "difference_ci95_board_bootstrap": [low, high],
                "mcnemar_discordant": [b, c],
                "mcnemar_exact_p": mcnemar_exact(b, c),
                "verdict": verdict,
            }
    report["note"] = (
        "Heuristic against three copies of one bot, CARLA referee; 0.25 is parity. Intervals "
        "resample boards (each board is played in all four seats). Port and wrapper games share "
        "seeds but diverge after their first different decision."
    )
    return report


def table(report):
    lines = [
        "| Bot | Trading | Games | Heuristic win rate (95% CI) | Failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, c in report["conditions"].items():
        bot, trading = key.split("|")
        ci = c["heuristic_win_rate_ci95"]
        rate = (
            "n/a"
            if c["heuristic_win_rate"] is None
            else f"{c['heuristic_win_rate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        )
        lines.append(
            f"| {bot} | {(trading[-1] == '1' and 'on') or 'off'} | {c['scheduled']} | {rate} | {c['failures']} |"
        )
    lines += [
        "",
        "| Pair | Trading | Paired games | Wrapper minus port (95% CI) | McNemar p | Verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, q in report["pairs"].items():
        pair, trading = key.split("|")
        low, high = q["difference_ci95_board_bootstrap"]
        lines.append(
            f"| {pair} | {(trading[-1] == '1' and 'on') or 'off'} | {q['paired_games']} | "
            f"{q['difference_wrapper_minus_port']:+.3f} [{low:+.3f}, {high:+.3f}] | "
            f"{q['mcnemar_exact_p']:.3f} | {q['verdict']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------- main


def main():
    from carla.engine.rng import derive_seed
    from carla.eval.provenance import archive_sources
    from carla.utils.seeding import fixed_hash_seed

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", nargs="+", choices=tuple(PAIRS), default=list(PAIRS))
    parser.add_argument(
        "--trading", choices=("both", "off", "on"), default="both", help="JSettlers pairs"
    )
    parser.add_argument("--boards", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7_000_552)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument(
        "--jsettlers-mode", choices=("java-adapter", "native"), default="java-adapter"
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--java-workers", type=int, default=4, help="concurrent games with JVMs")
    parser.add_argument("--margin", type=float, default=0.05, help="equivalence margin")
    parser.add_argument("--min-boards", type=int, default=30, help="boards needed for a verdict")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/parity"))
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--retry-failures", action="store_true", help="replay games whose latest row failed"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / "games.jsonl"
    rows = [json.loads(line) for line in journal.open(encoding="utf-8")] if journal.exists() else []

    if not args.analyze_only:
        if any(c["bot"] in PROCESS_BACKED for c in conditions(args.pairs, args.trading)):
            for variable in ("JSETTLERS_JAR", "CARLA_JSETTLERS_CLASSES"):
                if not os.environ.get(variable):
                    parser.error(f"{variable} is required for the Java wrappers")
        # Each pass (first run, resume, retry) keeps its own manifest and source archive.
        if (args.output / "manifest.json").exists():
            k = 1
            while (args.output / f"manifest-{k}.json").exists():
                k += 1
            for name, suffix in (("manifest", ".json"), ("source", ".zip")):
                old = args.output / f"{name}{suffix}"
                if old.exists():
                    old.rename(args.output / f"{name}-{k}{suffix}")
        manifest = {
            "schema": 1,
            "referee": "CARLA (official rules)",
            "design": "heuristic vs three copies of one bot; each board in all four seats",
            "pairs": {p: PAIRS[p][:2] for p in args.pairs},
            "boards": args.boards,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "jsettlers_native_mode": args.jsettlers_mode,
            "margin": args.margin,
            "retry_failures": args.retry_failures,
            "python": platform.python_version(),
            "python_hash_seed": fixed_hash_seed(),
            "failure_policy": (
                "recorded and kept; resumed runs skip journaled games; --retry-failures appends "
                "a replay and analysis uses each game's latest row"
            ),
            **archive_sources(args.output),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        done = {
            (r["condition"], r["board"], r["seat"])
            for r in latest(rows)[0]
            if not (args.retry_failures and r["failure"] is not None)
        }
        jobs = [
            (c, board, seat, derive_seed(args.seed, "parity-board", board))
            for board in range(args.boards)  # board-major: partial runs stay balanced
            for c in conditions(args.pairs, args.trading)
            for seat in range(4)
            if (condition_key(c), board, seat) not in done
        ]
        print(f"{len(jobs)} games to play ({len(done)} already journaled)", flush=True)
        started = time.time()
        with (
            ProcessPoolExecutor(args.workers) as pool,
            ProcessPoolExecutor(args.java_workers) as java_pool,
            journal.open("a", encoding="utf-8") as out,
        ):
            futures = [
                (java_pool if c["bot"] in PROCESS_BACKED else pool).submit(
                    play_one, c, board, seat, seed, args.max_turns, args.jsettlers_mode
                )
                for c, board, seat, seed in jobs
            ]
            for count, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
                if count % 50 == 0 or row["failure"]:
                    rate = count / max(time.time() - started, 1e-9)
                    eta = (len(jobs) - count) / rate / 60
                    note = (
                        f" FAILURE {row['condition']}: {row['failure']}" if row["failure"] else ""
                    )
                    print(
                        f"{count}/{len(jobs)} games, {rate:.2f}/s, ~{eta:.0f} min left{note}",
                        flush=True,
                    )

    report = analyze(rows, args.margin, min_boards=args.min_boards)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "report.md").write_text(table(report), encoding="utf-8")
    print(table(report))


if __name__ == "__main__":
    from carla.utils.seeding import rerun_with_fixed_hash_seed

    rerun_with_fixed_hash_seed()  # Catanatron decisions depend on string hashing.
    main()
