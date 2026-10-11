"""Command line interface: `carla <command> --help`."""

import argparse
import logging
import random
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
    p.add_argument("--config", type=Path, help="experiment config providing engine rules")
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

    p = commands.add_parser("gui", help="play against bots, watch them, or replay a record")
    p.add_argument("record", type=Path, nargs="?", help="game record (.json or .json.gz) to replay")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--play", action="store_true", help="play one seat against bots (default)")
    mode.add_argument("--watch", action="store_true", help="watch a live game between bots")
    p.add_argument(
        "--seat",
        type=int,
        default=random.randint(0, 3),
        choices=range(4),
        help="your seat in --play",
    )
    p.add_argument(
        "--opponents",
        type=_agent_names,
        default=(),
        metavar="A,B,...",
        help="bots for the other seats (all four with --watch), cycled "
        "(default: heuristic; eval.agents with --watch)",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="config seed; game i uses derive_seed(seed, 'game', i) "
        "(default: the config's seed, or a random one without --config)",
    )
    p.add_argument("--delay", type=float, default=0.6, help="seconds between bot moves at 1x")
    p.add_argument(
        "--no-trading",
        dest="trading",
        action="store_false",
        help="no trades between players (bank and port trades remain)",
    )
    p.add_argument(
        "--config", type=Path, help="experiment config (engine rules, seed, eval.agents)"
    )
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. engine.vp_to_win=8",
    )

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
    from dataclasses import replace

    from carla.agents import make_agent
    from carla.engine import EngineConfig
    from carla.engine.rng import derive_seed
    from carla.eval import play_game

    config = load_config(args.config).engine if args.config else EngineConfig()
    if args.max_turns is not None:
        config = replace(config, max_turns=args.max_turns)
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
    from carla.eval.metrics import trade_metrics

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
    if args.records and cfg.engine.allow_domestic_trade:
        trade = trade_metrics(result.results)
        print(
            f"trades: {int(trade['proposals'])} proposals, "
            f"{int(trade['invitations'])} invitations, "
            f"{int(trade['counteroffers'])} counteroffers; "
            f"{trade['feasibility_rate']:.3f} proposal feasibility, "
            f"{trade['acceptance_rate']:.3f} acceptance, "
            f"{trade['counter_feasibility_rate']:.3f} counter feasibility, "
            f"{trade['execution_rate']:.3f} execution"
        )
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


def _agent_names(text: str) -> tuple[str, ...]:
    from carla.agents import AGENTS

    names = tuple(name.strip() for name in text.split(",") if name.strip())
    unknown = [name for name in names if name not in AGENTS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"unknown agents {unknown}; known: {sorted(AGENTS)}")
    return names


def _gui(args) -> int:
    from carla.gui.app import GuiOptions
    from carla.gui.app import main as gui_main

    if args.record is not None and (args.play or args.watch):
        print(
            "carla gui: a record is replayed; --play and --watch start new games", file=sys.stderr
        )
        return 2
    if args.record is not None and not args.record.exists():
        print(f"carla gui: no such record {args.record}", file=sys.stderr)
        return 2
    seed = args.seed
    if seed is None and args.config is None:  # a new board each launch, still reproducible
        seed = random.randrange(2**31)
        print(f"carla gui: seed {seed} (pass --seed {seed} to play this board again)")
    trading = f"engine.allow_domestic_trade={str(args.trading).lower()}"  # --set may override
    overrides = [trading, *args.overrides, *([f"seed={seed}"] if seed is not None else [])]
    config = load_config(args.config, overrides)
    mode = "replay" if args.record is not None else "watch" if args.watch else "play"
    default = config.eval.agents if mode == "watch" else ("heuristic",)
    options = GuiOptions(
        mode=mode,
        record=args.record,
        seat=args.seat,
        opponents=args.opponents or default,
        delay=args.delay,
        seed=config.seed,
        engine=config.engine,
        output_dir=Path(config.runtime.output_dir),
    )
    try:
        gui_main(options)
    except ImportError as exc:
        if exc.name not in ("arcade", "pyglet"):
            raise
        print(
            "carla gui needs Arcade: uv sync --extra gui (or pip install -e '.[gui]')",
            file=sys.stderr,
        )
        return 1
    except NotImplementedError as exc:
        print(f"carla gui: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
