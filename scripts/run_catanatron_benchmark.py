"""Play any CARLA agent against unmodified Catanatron bots on Catanatron's own engine.

The candidate (a registry name such as `heuristic`, `native-f` or `ppo:<checkpoint>`) is seated
through `carla.agents.catanatron.host.CarlaPlayer` with a CARLA mirror, so history- and
event-based agents (PPO checkpoints, native JSettlers) play as under CARLA's referee while
Catanatron owns every transition. Each Catanatron seed is played with the candidate in every
seat against three originals (`--opponents f` or `ab2`).

The candidate sees only its own cards and public counts (`view`). PPO checkpoints trained with
domestic trading need `--allow-trading-override`: Catanatron has no player trades. Failed games
stay in the denominator; they are recorded, never replaced.

Example:
    python scripts/run_catanatron_benchmark.py --agent ppo:runs/x/latest.pt --opponents f --boards 25
"""

import argparse
import json
import platform
import time
import traceback
from pathlib import Path

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.player import Color
from catanatron.players import minimax
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from carla.agents.catanatron.host import CarlaPlayer
from carla.agents.registry import is_agent_name, make_agent
from carla.engine.rng import derive_seed
from carla.eval.metrics import wilson_interval
from carla.eval.provenance import archive_sources
from carla.utils.seeding import fixed_hash_seed, rerun_with_fixed_hash_seed


def original(kind, color):
    if kind == "f":
        return ValueFunctionPlayer(color)
    return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=2))


def play(agent_name, opponents, seed, seat, allow_trading_override):
    colors = list(Color)
    probe = Game([original(opponents, c) for c in colors], seed=seed)
    target = probe.state.colors[seat]  # Catanatron seats players by seed.
    agent = make_agent(agent_name)
    if allow_trading_override:
        agent.allow_trading_override = True
    seated = CarlaPlayer(target, agent, seed=derive_seed(seed, "candidate", seat))
    players = [seated if c == target else original(opponents, c) for c in colors]
    game = Game(players, seed=seed)
    row = {"seed": seed, "seat": seat, "failure": None}
    started = time.perf_counter()
    try:
        winner = game.play()
        row.update(
            winner_seat=None if winner is None else list(game.state.colors).index(winner),
            candidate_won=winner == target,
            truncated=winner is None and game.state.num_turns >= TURNS_LIMIT,
            turns=game.state.num_turns,
            road_resyncs=seated.road_resyncs,
            fallbacks=getattr(agent, "fallbacks", 0),
        )
    except Exception as error:
        row["failure"] = f"{type(error).__name__}: {error}"
        row["traceback"] = traceback.format_exc(limit=6)
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="heuristic", help="candidate agent registry name")
    parser.add_argument("--opponents", choices=("f", "ab2"), default="f")
    parser.add_argument("--boards", type=int, default=10, help="Catanatron seeds to play")
    parser.add_argument("--seed", type=int, default=56000, help="first Catanatron seed")
    parser.add_argument("--allow-trading-override", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/catanatron-engine"))
    args = parser.parse_args()
    if not is_agent_name(args.agent) or args.boards < 1:
        parser.error("a registered candidate agent and positive boards are required")
    # Upstream AB2 searches until a 20-second wall-clock deadline; lift it for reproducibility.
    minimax.MAX_SEARCH_TIME_SECS = 1e9
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "referee": "Catanatron ecf9311 (CARLA mirror, strict state checks)",
        "candidate": args.agent,
        "opponents": {"f": "ValueFunctionPlayer", "ab2": "AlphaBetaPlayer(depth=2)"}[
            args.opponents
        ],
        "information": "view",
        "python": platform.python_version(),
        "python_hash_seed": fixed_hash_seed(),
        "allow_trading_override": args.allow_trading_override,
        **archive_sources(args.output),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (args.output / "games.jsonl").open("w", encoding="utf-8") as journal:
        for board in range(args.boards):
            for seat in range(4):
                row = play(
                    args.agent, args.opponents, args.seed + board, seat, args.allow_trading_override
                )
                rows.append(row)
                journal.write(json.dumps(row) + "\n")
                journal.flush()
                print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)
    wins = sum(bool(r.get("candidate_won")) for r in rows)
    low, high = wilson_interval(wins, len(rows))
    summary = {
        "games": len(rows),
        "candidate_wins": wins,
        "win_rate": wins / len(rows),
        "wilson_95": [low, high],
        "failures": sum(r["failure"] is not None for r in rows),
        "truncations": sum(bool(r.get("truncated")) for r in rows),
        "note": "Seats share boards; intervals are descriptive, not clustered by board.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(int(summary["failures"] > 0))


if __name__ == "__main__":
    rerun_with_fixed_hash_seed()  # Catanatron decisions depend on string hashing.
    main()
