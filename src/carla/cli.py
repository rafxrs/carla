"""Command line interface: `carla <command> --help`."""

import argparse
import logging
import sys
import time
from pathlib import Path

from carla.config import load_config, to_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carla", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("simulate", help="play games between agents and save their logs")
    p.add_argument("--agents", nargs=4, default=["random"] * 4, metavar="AGENT")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--save", type=Path, help="directory for .json.gz records and .txt logs")

    p = commands.add_parser("replay", help="show a recorded game or the position at a step")
    p.add_argument("record", type=Path)
    p.add_argument("--step", type=int, help="position after this many steps (default: end)")
    p.add_argument("--text", action="store_true", help="print the whole human-readable log")

    p = commands.add_parser("tournament", help="win rates with confidence intervals")
    p.add_argument("--config", type=Path, help="experiment config providing eval/engine settings")
    p.add_argument("--agents", nargs=4, metavar="AGENT")
    p.add_argument("--games", type=int, help="games per seed")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--workers", type=int)
    p.add_argument("--records", type=Path, help="save every game record here")

    p = commands.add_parser("train", help="train a policy (dry run until PPO lands)")
    p.add_argument("config", type=Path)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. ppo.learning_rate=1e-4")

    p = commands.add_parser("config", help="print a fully resolved config")
    p.add_argument("config", type=Path, nargs="?")
    p.add_argument("overrides", nargs="*")

    p = commands.add_parser("gui", help="Arcade viewer (not implemented yet)")
    p.add_argument("record", type=Path, nargs="?")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return {
        "simulate": _simulate,
        "replay": _replay,
        "tournament": _tournament,
        "train": _train,
        "config": _config,
        "gui": _gui,
    }[args.command](args)


def _simulate(args) -> int:
    from carla.agents import make_agent
    from carla.engine import EngineConfig
    from carla.engine.rng import derive_seed
    from carla.eval import play_game

    config = EngineConfig(max_turns=args.max_turns)
    start, steps = time.perf_counter(), 0
    for i in range(args.games):
        seed = derive_seed(args.seed, "game", i)
        result = play_game(
            [make_agent(a) for a in args.agents], seed, config, record=bool(args.save)
        )
        steps += result.steps
        winner = "none" if result.winner is None else f"P{result.winner} ({result.winner_name})"
        print(f"game {i}: seed {seed} winner {winner} turns {result.turns} steps {result.steps}")
        if args.save and result.record is not None:
            result.record.save(args.save / f"game_{i:04d}.json.gz")
            result.record.save_text(args.save / f"game_{i:04d}.txt")
    elapsed = time.perf_counter() - start
    print(
        f"{args.games} games in {elapsed:.2f}s: {args.games / elapsed:.1f} games/s, {steps / elapsed:.0f} steps/s"
    )
    return 0


def _replay(args) -> int:
    from carla.engine.record import GameRecord
    from carla.engine.text import describe_state

    record = GameRecord.load(args.record)
    if args.text:
        print(record.to_text(), end="")
    else:
        print(describe_state(record.replay(args.step)))
    return 0


def _tournament(args) -> int:
    from carla.eval import format_summary, run_tournament

    cfg = load_config(args.config)
    ev = cfg.eval
    result = run_tournament(
        agents=args.agents or ev.agents,
        num_games=args.games or ev.num_games,
        seeds=args.seeds or ev.seeds,
        engine_config=cfg.engine,
        rotate_seats=ev.rotate_seats,
        num_workers=args.workers or ev.num_workers,
        confidence=ev.confidence,
        records_dir=args.records,
    )
    for seed, summary in result.per_seed.items():
        rates = ", ".join(f"{s.name} {s.win_rate:.3f}" for s in summary.values())
        print(f"seed {seed}: {rates}")
    print(format_summary(result.summary, ev.confidence))
    return 0


def _train(args) -> int:
    from carla.rl.train import train

    train(load_config(args.config, args.overrides))
    return 0


def _config(args) -> int:
    import yaml

    print(
        yaml.safe_dump(to_dict(load_config(args.config, args.overrides)), sort_keys=False), end=""
    )
    return 0


def _gui(args) -> int:
    from carla.gui.app import main as gui_main

    gui_main(str(args.record) if args.record else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
