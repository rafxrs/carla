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

    p = commands.add_parser("train", help="train a synchronous MLP/GNN PPO baseline")
    p.add_argument("config", type=Path)
    p.add_argument("--resume", type=Path, help="resume from a training checkpoint")
    p.add_argument(
        "--init-from",
        type=Path,
        help="start a new run from another run's policy weights (fresh optimizer and counters)",
    )
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. ppo.learning_rate=1e-4")

    p = commands.add_parser("bc", help="behaviour-clone the heuristic bot as a PPO warm start")
    p.add_argument("config", type=Path, help="experiment config (model, env and opponents)")
    p.add_argument("--output", type=Path, required=True, help="directory for bc.pt")
    p.add_argument("--games", type=int, default=600, help="expert games to record")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=1, help="processes playing expert games")
    p.add_argument("--eval-boards", type=int, default=10, help="0 skips the final evaluation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. runtime.device=cuda")

    p = commands.add_parser(
        "strategy-bc",
        help="strategy-conditioned behaviour cloning of the styled heuristic teachers",
    )
    p.add_argument("config", type=Path, help="experiment config (model, env, strategy)")
    p.add_argument(
        "--output", type=Path, required=True, help="directory for bc.pt and the z library"
    )
    p.add_argument("--games", type=int, default=900, help="teacher games (balanced over styles)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=1, help="processes playing teacher games")
    p.add_argument("--eval-boards", type=int, default=5, help="boards per z variant; 0 skips")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. runtime.device=cuda")

    p = commands.add_parser("league", help="AlphaStar-style league training (main, exploiters)")
    p.add_argument("config", type=Path)
    p.add_argument("--resume", type=Path, help="league run directory to resume")
    p.add_argument("overrides", nargs="*", help="dotted overrides")

    p = commands.add_parser("profile", help="behaviour profile: what each agent builds")
    p.add_argument("agents", nargs=4, help="four agents (bots or ppo:<checkpoint>[@variant])")
    p.add_argument("--games", type=int, default=25, help="boards, each in four rotations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, help="experiment config providing engine rules")
    p.add_argument("--csv", type=Path)

    p = commands.add_parser(
        "robustness", help="one agent against several opponent tables (worst-case win rate)"
    )
    p.add_argument("subject", help="a bot or ppo:<checkpoint>[@variant]")
    p.add_argument(
        "--table",
        action="append",
        help="three comma-separated opponents (repeatable; default: heuristic and teacher tables)",
    )
    p.add_argument("--boards", type=int, default=10, help="boards per table, each in 4 rotations")
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--config", type=Path, help="experiment config providing engine rules")
    p.add_argument("--csv", type=Path, help="per-game records")
    p.add_argument("--json", type=Path, help="summary")

    p = commands.add_parser("league-report", help="summarize a league run")
    p.add_argument("run", type=Path, help="league run directory")

    p = commands.add_parser("evaluate", help="evaluate a checkpoint with seat rotation")
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--opponent",
        default="heuristic",
        help="a bot for every seat, or a comma-separated table of three bots",
    )
    p.add_argument(
        "--strategy",
        help="strategy-conditioned policies: select, none or an archetype (default: select "
        "with a selector, else none)",
    )
    p.add_argument("--boards", type=int, default=10, help="held-out boards, each played in 4 seats")
    p.add_argument("--seed", type=int, default=10_000)

    # Final test exports stay separate from the training journals and their recovery counts.
    p.add_argument("--sampled", action="store_true", help="sample actions instead of greedy play")
    p.add_argument("--csv", type=Path, help="append per-game records to a separate evaluation CSV")
    p.add_argument("--split", choices=("validation", "test"), default="test")

    p = commands.add_parser("policy-tournament", help="compare checkpoints on paired boards")
    p.add_argument("players", nargs=4, help="four checkpoint paths or random/heuristic bots")
    p.add_argument("--boards", type=int, default=10, help="boards, each played in four rotations")
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--sampled", action="store_true", help="sample policy actions")
    p.add_argument("--csv", type=Path, help="write per-game results")

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
        help="bots for the other seats (all four with --watch), cycled; ppo:<checkpoint.pt> "
        "plays a trained policy (default: heuristic; eval.agents with --watch)",
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

    # argparse on Python 3.11 leaves positional overrides after --resume unparsed.
    # Accept only KEY=VALUE tokens there; retain normal errors for unknown options.
    args, extra = parser.parse_known_args(argv)
    if extra:
        if args.command in ("train", "bc", "strategy-bc", "league") and all(
            "=" in item and not item.startswith("-") for item in extra
        ):
            args.overrides.extend(extra)
        else:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return {
        "simulate": _simulate,
        "replay": _replay,
        "tournament": _tournament,
        "train": _train,
        "bc": _bc,
        "strategy-bc": _strategy_bc,
        "league": _league,
        "profile": _profile,
        "robustness": _robustness,
        "league-report": _league_report,
        "evaluate": _evaluate,
        "policy-tournament": _policy_tournament,
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

    train(load_config(args.config, args.overrides), resume=args.resume, init_from=args.init_from)
    return 0


def _bc(args) -> int:
    from carla.rl.imitation import behaviour_clone

    behaviour_clone(
        load_config(args.config, args.overrides),
        args.output,
        games=args.games,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        workers=args.workers,
        eval_boards=args.eval_boards,
        seed=args.seed,
    )
    return 0


def _strategy_bc(args) -> int:
    from carla.rl.strategy_imitation import strategy_behaviour_clone

    strategy_behaviour_clone(
        load_config(args.config, args.overrides),
        args.output,
        games=args.games,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        minibatch_size=args.minibatch_size,
        workers=args.workers,
        eval_boards=args.eval_boards,
        seed=args.seed,
    )
    return 0


def _league(args) -> int:
    from carla.rl.league.trainer import train_league

    train_league(load_config(args.config, args.overrides), resume=args.resume)
    return 0


def _profile(args) -> int:
    import csv

    from carla.agents import make_agent
    from carla.engine import EngineConfig
    from carla.eval.behaviour import behaviour_profile, format_profiles

    engine = load_config(args.config).engine if args.config else EngineConfig()
    rows = behaviour_profile(
        [lambda name=name: make_agent(name) for name in args.agents],
        args.games,
        args.seed,
        engine,
    )
    print(format_profiles(rows))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


def _robustness(args) -> int:
    import json

    from carla.engine import EngineConfig
    from carla.eval.robustness import DEFAULT_TABLES, format_matrix, robustness_matrix

    if args.config:
        engine = load_config(args.config).engine
    elif args.subject.startswith(("ppo:", "ppo-sampled:")):
        from carla.agents.policy_agent import load_checkpoint, split_variant

        engine = load_checkpoint(split_variant(args.subject.split(":", 1)[1])[0])[0].engine
    else:
        engine = EngineConfig()
    tables = DEFAULT_TABLES
    if args.table:
        tables = [tuple(part.strip() for part in t.split(",")) for t in args.table]
        if any(len(t) != 3 for t in tables):
            print("carla robustness: each --table needs three opponents", file=sys.stderr)
            return 2
    result = robustness_matrix(
        args.subject,
        tables,
        boards=args.boards,
        seed=args.seed,
        engine=engine,
        workers=args.workers,
        csv_path=args.csv,
    )
    print(format_matrix(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def _league_report(args) -> int:
    from carla.rl.league.report import league_report

    print(league_report(args.run))
    return 0


def _evaluate(args) -> int:
    from carla.rl.evaluate import evaluate_checkpoint

    result = evaluate_checkpoint(
        args.checkpoint,
        opponent=args.opponent,
        boards=args.boards,
        seed=args.seed,
        deterministic=not args.sampled,
        strategy=args.strategy,
    )
    if args.csv is not None:
        from carla.rl.evaluate import export_evaluation

        export_evaluation(args.checkpoint, result, args.csv, split=args.split)
    print(
        f"policy vs {result.opponent}: {result.wins}/{result.games} wins "
        f"({result.win_rate:.3f}, 95% CI [{result.ci_low:.3f}, {result.ci_high:.3f}]); "
        f"mean VP {result.mean_vp:.2f}"
    )
    return 0


def _policy_tournament(args) -> int:
    from carla.rl.policy_tournament import run_policy_tournament

    result = run_policy_tournament(
        args.players,
        boards=args.boards,
        seed=args.seed,
        deterministic=not args.sampled,
        csv_path=args.csv,
    )
    for entry in result.entries:
        print(
            f"slot {entry.slot} {entry.player}: {entry.wins}/{entry.games} wins "
            f"({entry.win_rate:.3f}, 95% CI [{entry.ci_low:.3f}, {entry.ci_high:.3f}]); "
            f"mean VP {entry.mean_vp:.2f}"
        )
    return 0


def _config(args) -> int:
    import yaml

    print(
        yaml.safe_dump(to_dict(load_config(args.config, args.overrides)), sort_keys=False), end=""
    )
    return 0


def _agent_names(text: str) -> tuple[str, ...]:
    from carla.agents import AGENTS, is_agent_name

    names = tuple(name.strip() for name in text.split(",") if name.strip())
    unknown = [name for name in names if not is_agent_name(name)]
    if not names or unknown:
        known = [*sorted(AGENTS), "ppo:<checkpoint.pt>"]
        raise argparse.ArgumentTypeError(f"unknown agents {unknown}; known: {known}")
    from carla.agents.policy_agent import split_variant

    missing = [
        n
        for n in names
        if n.startswith(("ppo:", "ppo-sampled:"))
        and not Path(split_variant(n.split(":", 1)[1])[0]).is_file()
    ]
    if missing:
        raise argparse.ArgumentTypeError(f"checkpoint not found: {missing}")
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
    opponents = args.opponents or default
    engine = config.engine
    checkpoints = [name[4:] for name in opponents if name.startswith("ppo:")]
    if checkpoints and mode != "replay":
        from carla.agents.policy_agent import load_checkpoint

        trained = [load_checkpoint(path)[0].engine for path in checkpoints]
        if args.config is None and not args.overrides and args.trading:
            engine = trained[0]  # Play by the rules the policy was trained with.
        if any(rules != engine for rules in trained):
            print(
                "carla gui: the checkpoint was trained with other engine rules; pass its "
                "experiment config with --config (and no rule overrides)",
                file=sys.stderr,
            )
            return 2
    options = GuiOptions(
        mode=mode,
        record=args.record,
        seat=args.seat,
        opponents=opponents,
        delay=args.delay,
        seed=config.seed,
        engine=engine,
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
