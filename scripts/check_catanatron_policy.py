"""Compare CARLA's native Catanatron ports with the originals on Catanatron's own engine.

compare: Catanatron referees a game of four original bots. At each decision of the target seat,
the native port (seated through `carla.agents.catanatron.host`) also decides on the same state.
The original's move is applied, so the history is the original's. Disagreements are
classified as native ties (the native port scores both moves equally, so only the order
decides) or real differences.

play: the native port plays each seat in turn against three original bots, on boards Catanatron
draws from each seed.

This is a policy disagreement diagnostic, not an equivalence certificate. `--information open`
gives the native port the true hands and deck, which the originals read. `view` keeps CARLA's
hidden-information boundary. Upstream F samples chance by executing a copy of the game; the
native port takes exact expectations, so some disagreement is a declared design difference.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players import minimax
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from carla.agents.catanatron.host import CarlaPlayer, carla_action
from carla.agents.catanatron.native import NativeCatanatronAgent
from carla.eval.provenance import archive_sources
from carla.utils.seeding import rerun_with_fixed_hash_seed

TURNS_LIMIT = 1000

# Upstream AB2 stops searching at a 20-second wall-clock deadline, so its decisions depend on
# machine load. Lift it here so comparisons measure the algorithm (deadline hits are rare).
minimax.MAX_SEARCH_TIME_SECS = 1e9


def original(bot, color):
    if bot == "f":
        return ValueFunctionPlayer(color)
    return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=2))


def native(bot, color, information, chance="exact", share_rng=True):
    agent = NativeCatanatronAgent(
        depth=1 if bot == "f" else 2,
        information="open" if information == "open" else "sampled",
        chance=chance,
    )
    return CarlaPlayer(color, agent, information=information, share_rng=share_rng)


def phase(game):
    state = game.state
    if state.is_road_building and state.free_roads_available > 0:
        return "ROAD_BUILDING"
    prompt = state.current_prompt.name
    if prompt == "PLAY_TURN":
        rolled = state.player_state[f"P{state.current_turn_index}_HAS_ROLLED"]
        return "MAIN" if rolled else "PRE_ROLL"
    return prompt


def compare(bot, seed, seat, information, limit, chance="exact"):
    players = [original(bot, color) for color in Color]
    game = Game(players, seed=seed)
    target = game.state.colors[seat]
    # The port draws from a copy of Catanatron's RNG, so the original's own draws are unchanged.
    port = native(bot, target, information, chance, share_rng=False)
    row = {
        "bot": bot,
        "chance": chance,
        "seed": seed,
        "seat": seat,
        "information": information,
        "decisions": 0,
        "agreements": 0,
        "native_ties": 0,
        "by_phase": {},
        "first_divergence": None,
        "disagreements": [],
        "failure": None,
    }
    by_phase = Counter()
    agreed = Counter()

    def decide(player, game, playable):
        if player.color != target or row["decisions"] >= limit:
            return player.decide(game, playable)
        name = phase(game)
        actual = port.decide(game, playable)
        expected = player.decide(game, playable)
        same = actual == expected
        row["decisions"] += 1
        by_phase[name] += 1
        if same:
            row["agreements"] += 1
            agreed[name] += 1
        else:
            colors = list(game.state.colors)
            scores = {
                json.dumps(a, sort_keys=True): v for a, v in getattr(port.agent, "last_scores", ())
            }
            want = carla_action(expected, colors).to_json()
            best = max(scores.values(), default=None)
            score = scores.get(json.dumps(want, sort_keys=True))
            tie = score is not None and abs(score - best) < 1e-9
            row["native_ties"] += int(tie)
            entry = {
                "turn": game.state.num_turns,
                "phase": name,
                "native": carla_action(actual, colors).to_json(),
                "original": want,
                "native_tie": tie,
            }
            if row["first_divergence"] is None:
                row["first_divergence"] = entry
            if len(row["disagreements"]) < 50:
                row["disagreements"].append(entry)
        return expected

    try:
        while game.winning_color() is None and game.state.num_turns < TURNS_LIMIT:
            game.play_tick(decide_fn=decide)
            if row["decisions"] >= limit:
                break
        row["completed"] = game.winning_color() is not None
    except Exception as error:
        row["failure"] = f"{type(error).__name__}: {error}"
    row["by_phase"] = {k: {"compared": by_phase[k], "agreed": agreed[k]} for k in sorted(by_phase)}
    return row


def play(bot, seed, seat, information, chance="exact"):
    colors = list(Color)
    players = [original(bot, color) for color in colors]
    game = Game(players, seed=seed)
    target = game.state.colors[seat]
    players[colors.index(target)] = native(bot, target, information, chance)
    game = Game(players, seed=seed)
    if game.state.colors[seat] != target:
        raise RuntimeError("Catanatron seating changed between identical seeds")
    row = {"bot": bot, "seed": seed, "seat": seat, "information": information, "chance": chance}
    row["failure"] = None
    try:
        winner = game.play()
        row.update(
            winner_seat=None if winner is None else list(game.state.colors).index(winner),
            native_won=winner == target,
            turns=game.state.num_turns,
        )
    except Exception as error:
        row["failure"] = f"{type(error).__name__}: {error}"
    return row


if __name__ == "__main__":
    rerun_with_fixed_hash_seed()  # Catanatron decisions depend on string hashing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "play"))
    parser.add_argument("--bots", nargs="+", choices=("f", "ab2"), default=["f"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[12])
    parser.add_argument("--seats", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--information", choices=("view", "open"), default="open")
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument(
        "--chance",
        choices=("exact", "upstream"),
        default="exact",
        help="native chance handling: exact expectations, or upstream's (see native.py)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/native-catanatron-policy")
    )
    args = parser.parse_args()
    if args.max_decisions < 1 or any(seat not in range(4) for seat in args.seats):
        parser.error("positive decision limit and seats 0..3 required")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "mode": args.mode,
        "whole_policy_equivalence": False,
        "artifacts": archive_sources(args.output),
        "cases": [],
    }
    for bot in args.bots:
        for seed in args.seeds:
            for seat in args.seats:
                row = (
                    compare(bot, seed, seat, args.information, args.max_decisions, args.chance)
                    if args.mode == "compare"
                    else play(bot, seed, seat, args.information, args.chance)
                )
                report["cases"].append(row)
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(row), flush=True)
    raise SystemExit(int(any(row["failure"] is not None for row in report["cases"])))
