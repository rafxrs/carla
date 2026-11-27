"""Measure feature and action agreement; a passing feature test is not policy equivalence."""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

from carla.agents.api import AgentView
from carla.agents.catanatron.agent import CatanatronAgent
from carla.agents.catanatron.codec import reconstruct
from carla.agents.catanatron.native import WEIGHTS, NativeCatanatronAgent, features
from carla.agents.heuristic import HeuristicAgent
from carla.engine import Game
from carla.engine.config import EngineConfig
from carla.planning.jsettlers_eta import building_estimates, rolls_per_resource
from carla.utils.seeding import rerun_with_fixed_hash_seed


def run(games, output):
    from catanatron.models.player import Color
    from catanatron.players.value import base_fn

    feature_count = feature_mismatches = eta_count = eta_mismatches = 0
    decisions = Counter()
    agreements = Counter()
    errors = []
    jsettlers = None
    if os.environ.get("JSETTLERS_JAR") and os.environ.get("CARLA_JSETTLERS_CLASSES"):
        from carla.agents.jsettlers import JSettlersAgent

        jsettlers = JSettlersAgent("fast")
    for seed in range(games):
        game = Game(EngineConfig(max_turns=200), 100000 + seed)
        heuristic = HeuristicAgent()
        wrappers = [CatanatronAgent() for _ in range(4)]
        natives = [NativeCatanatronAgent() for _ in range(4)]
        for seat in range(4):
            wrappers[seat].reset(game, seat, 5000 + seat)
            natives[seat].reset(game, seat, 5000 + seat)
        if jsettlers:
            jsettlers.reset(game, 0, seed)
        try:
            while not game.is_over:
                if game.num_steps % 7 == 0:
                    reference = reconstruct(game, 42)
                    for seat in range(4):
                        f = features(game, seat)
                        baseline = base_fn(dict.fromkeys(WEIGHTS, 0.0))(
                            reference, tuple(Color)[seat]
                        )
                        for name in WEIGHTS:
                            if name in ("reachable_production_0", "longest_road"):
                                continue
                            weights = dict.fromkeys(WEIGHTS, 0.0)
                            weights[name] = 1.0
                            expected = base_fn(weights)(reference, tuple(Color)[seat]) - baseline
                            feature_count += 1
                            if not math.isclose(f[name], expected, abs_tol=1e-8, rel_tol=1e-8):
                                feature_mismatches += 1
                                errors.append([seed, game.num_steps, seat, name, f[name], expected])
                    if jsettlers:
                        reply = jsettlers._request(
                            "eta", view=AgentView.from_game(game, 0).to_json()
                        )
                        eta_count += 1
                        if rolls_per_resource(game.state, 0) != tuple(
                            reply["periods"][r] for r in (5, 1, 3, 4, 2)
                        ) or building_estimates(game.state, 0) != tuple(reply["estimates"][:4]):
                            eta_mismatches += 1
                    seat = game.decision_player
                    if len(game.legal_actions()) > 1:
                        phase = game.state.phase.name
                        decisions[phase] += 1
                        agreements[phase] += wrappers[seat].act(
                            game, seat, game.legal_actions()
                        ) == natives[seat].act(game, seat, game.legal_actions())
                events = game.step(heuristic.act(game, game.decision_player, game.legal_actions()))
                if jsettlers:
                    jsettlers.observe([event.visible_to(0) for event in events], game.num_steps)
        finally:
            if jsettlers:
                jsettlers.close()
    report = {
        "games": games,
        "feature_comparisons": feature_count,
        "feature_mismatches": feature_mismatches,
        "eta_comparisons": eta_count,
        "eta_mismatches": eta_mismatches,
        "f_action_agreement_by_phase": {
            phase: {"compared": n, "agreed": agreements[phase]} for phase, n in decisions.items()
        },
        "examples": errors[:20],
        "interpretation": "Feature/ETA agreement only. Native F/AB use CARLA transitions, exact chance and CARLA action order; full policy equivalence is not asserted. Whole-policy checks: scripts/check_catanatron_policy.py (native vs original on Catanatron's engine) and scripts/check_jsettlers_policy.py (native vs Java SMART/FAST).",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rerun_with_fixed_hash_seed()  # Catanatron decisions depend on string hashing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--output", default="outputs/benchmarks/equivalence.json")
    args = parser.parse_args()
    run(args.games, args.output)
