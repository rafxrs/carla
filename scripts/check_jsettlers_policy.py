"""Compare native decisions with the Java adapter on a Java-driven history.

This is a policy disagreement diagnostic, not an equivalence certificate. Once
decisions diverge, retained plans may differ on subsequent steps. Records retain
the first divergence and distinguish decisions before/after it. Views are redacted.
"""

import argparse
import json
from pathlib import Path

from carla.agents.api import AgentView
from carla.agents.heuristic import HeuristicAgent
from carla.agents.jsettlers import JSettlersAgent
from carla.agents.jsettlers_native.agent import NativeJSettlersAgent
from carla.engine import Game
from carla.engine.config import EngineConfig
from carla.eval.provenance import archive_sources


def compare(strategy, seed, seat, trading, limit, output, compatibility="native"):
    game = Game(EngineConfig(max_turns=220, allow_domestic_trade=trading), seed)
    native, reference = NativeJSettlersAgent(strategy, compatibility=compatibility), JSettlersAgent(
        strategy
    )
    others = [HeuristicAgent() for _ in range(4)]
    prefix = f"{strategy}-{seed}-{seat}-trade{int(trading)}"
    row = {
        "strategy": strategy,
        "seed": seed,
        "seat": seat,
        "trading": trading,
        "decisions": 0,
        "agreements": 0,
        "first_divergence": None,
        "failure": None,
        "compatibility": compatibility,
    }
    try:
        native.reset(game, seat, 42)
        reference.reset(game, seat, 42)
        with (output / f"{prefix}.jsonl").open("w", encoding="utf-8") as journal:
            while not game.is_over and row["decisions"] < limit:
                actor = game.decision_player
                legal = game.legal_actions()
                if actor == seat:
                    actual = native.act(game, seat, legal)
                    expected = reference.act(game, seat, legal)
                    if not game.is_legal(actual) or not game.is_legal(expected):
                        raise ValueError(f"illegal decision: native={actual}, Java={expected}")
                    same = actual == expected
                    entry = {
                        "step": game.num_steps,
                        "phase": game.state.phase.name,
                        "native": actual.to_json(),
                        "java": expected.to_json(),
                        "agreement": same,
                    }
                    if not same:
                        entry["view"] = AgentView.from_game(game, seat).to_json()
                        entry["native_plan"] = [p.action().to_json() for p in native.plan]
                        entry["native_knowledge"] = native.knowledge.snapshot()
                        entry["native_negotiation"] = {
                            "selling": native.memory.selling,
                            "win_times": [t.win_eta for t in native.trackers],
                            "targets": [
                                p.action().to_json() if p is not None else None
                                for p in native.memory.targets
                            ],
                        }
                        entry["java_diagnostic"] = reference._request(
                            "inspect", view=entry["view"], operation="snapshot"
                        )["diagnostic"]
                        if row["first_divergence"] is None:
                            row["first_divergence"] = {
                                k: entry[k] for k in ("step", "phase", "native", "java")
                            }
                    journal.write(json.dumps(entry) + "\n")
                    journal.flush()
                    row["decisions"] += 1
                    row["agreements"] += int(same)
                    action = expected
                else:
                    action = others[actor].act(game, actor, legal)
                events = game.step(action)
                visible = [event.visible_to(seat) for event in events]
                native.observe(visible, game.num_steps)
                reference.observe(visible, game.num_steps)
        row.update(winner=game.winner, steps=game.num_steps, completed=game.is_over)
    except Exception as error:
        row["failure"] = f"{type(error).__name__}: {error}"
    finally:
        native.close()
        reference.close()
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", default=["fast", "smart"])
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--seats", nargs="+", type=int, default=[0])
    parser.add_argument("--trading", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--compatibility", choices=("native", "java-adapter"), default="native")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/native-jsettlers-policy")
    )
    args = parser.parse_args()
    if args.max_decisions < 1 or any(seat not in range(4) for seat in args.seats):
        parser.error("positive decision limit and seats 0..3 required")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "whole_policy_equivalence": False,
        "artifacts": archive_sources(args.output),
        "cases": [],
    }
    for strategy in args.strategies:
        for seat in args.seats:
            row = compare(
                strategy,
                args.seed,
                seat,
                args.trading,
                args.max_decisions,
                args.output,
                args.compatibility,
            )
            report["cases"].append(row)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(row), flush=True)
    raise SystemExit(int(any(row["failure"] is not None for row in report["cases"])))
