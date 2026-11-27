# Native JSettlers2 agents

> **Archived, 26 November 2026.** A record of how this port was built and validated, written in rounds as the work progressed. Earlier sections can be superseded by later ones. For current usage see [the benchmark guide](../README.md); for maintenance see [maintenance.md](../maintenance.md).

Native Python SMART and FAST agents now run under CARLA, without Java:
`jsettlers-smart-native` and `jsettlers-fast-native`. They derive from JSettlers
2.6.10, commit `c5141056d8e3fff548817b3a1712eb66085e897b`, with the stock robot
parameters. **Whole-policy conformance remains experimental.** These are
four-player, ten-point base-game ports; expansion/scenario robots are outside
this compatibility profile.

## Implementation

| Area | Native implementation |
| --- | --- |
| Information | Exact own cards, known/unknown opponent resources, visible-event accounting, chronological resource knowledge and independent Java-compatible agent RNG. |
| Opening | Paired settlement search, second settlement selection, initial road direction, opponent-placement estimates, upstream collection/tie order. |
| Tracking | Roads, settlements, cities, prerequisites, conflicts, propagated threats, city speedups, cached road-distance metadata and retained plans. |
| Planning | FAST building/award plans; SMART hypothetical placements, win-time projections, opponent/leader bonuses, development-card valuation and two-road plans. |
| Actions | Building, maritime trades, pre-roll knights, discards, robber/victim choices, Road Building, Monopoly and Year of Plenty. |
| Negotiation | Persistent selling/offer/target memory, offer generation, counteroffers, acceptance, race checks and CARLA confirmation. |
| Integration | Registry, tournament configs, lifecycle hooks, optional Java oracle, decision comparison script and CI checks. |
| Other referee | Observed OSP history for native agents under JSettlers, currently **domestic trading off**. Own card age, turns, setup and public production are reconstructed; opponent discard types are hidden. |

Implementation lives in `src/carla/agents/jsettlers_native/`; the shared fast
estimator remains in `src/carla/planning/jsettlers_eta.py`. Each module identifies
its upstream source. `NOTICE` and `COPYING` retain GPL attribution and licensing.

Milestones 1–7 of the native-port plan now have implementations: the compatibility
contract, Java reference, native knowledge/tracking, opening/build planning,
forced/card decisions, negotiation, and assembled agents. Milestone 8 has
component comparisons, complete-game decision comparisons, CARLA tournaments and
original-referee pilots. Its broader release-validation gate remains open.

Changes to the plan are explicit boundary modes instead of silently forcing
native choices to match the wrapper, bounded pure-function estimator caches,
and an initial external-referee transfer profile with domestic trading off.

The native policy only reads `AgentView` and visible events. Its board-query
copies contain known opponent resources and unknown development types; they are
not hidden-world samples. Real CARLA transitions, true opponent hands and the
referee RNG are never used for planning. Estimator caches have bounded sizes
and immutable keys/results.

## Compatibility modes

`NativeJSettlersAgent("smart", compatibility="native")` is the default. It
retains a buffered second Road Building road and evaluates counteroffers before
confirming a partner. These are explicit CARLA protocol choices.

`compatibility="java-adapter"` reproduces the current Java sidecar's two
different choices: replan the second free road, and confirm the first legal
partner. This mode enables comparisons of the translated algorithms with the
same boundary policy. It uses no Java at runtime. Registry factories also accept
it through `AgentSpec.parameters`.

```python
from carla.agents.api import AgentSpec
from carla.agents.registry import make_agent

agent = make_agent(AgentSpec("jsettlers-smart-native", {"compatibility": "java-adapter"}))
```

The native agents preserve reference details that can affect decisions:
rounded dice probabilities and float32 operations, descending discard-resource
order despite the upstream comment, the unused-on-land ship estimate in city
speedups, SMART's ship-ETA weighting for ordinary good roads, cached road counts,
and tracker ETAs left behind by temporary development-card scoring. Changes to
those behaviors should become separately identified variants.

## Java reference and fixes

`JSettlersReference` owns a separate Java process. Its `inspect` operations export
opening choices, candidate graphs, plans, resource knowledge, bank trades,
discards, free-resource choices, Monopoly, knight decisions, win projections and
SMART scores. Diagnostics mutate reference planning state and must not share a
process with a scored tournament opponent.

This work also corrected integration problems:

- Native `putPiece` was granting setup resources before CARLA's event granted
  them again. The adapter now preserves the pre-placement hand and applies the
  authoritative event once.
- The wrapper now feeds proposals/rejections into the original negotiator's
  memory and clears completed negotiations.
- CARLA's maritime ratios now use `Board.port_corners`; the Java wrapper also
  receives those actual endpoints for custom boards.
- The earlier initial-road randomness claim was incorrect. The pinned opening
  method has no `Math.random()` call. Its unnecessary copy was removed; the
  robot DM's call is in the unsupported Pirate Islands ship scenario.

## Divergence root causes (second follow-up)

A wider corpus (seeds 12, 31, 47 and 58, all four seats, FAST and SMART, with and without
domestic trading: 64 cases) exposed divergences that the single seed-12 history had not.
Each was traced to the first differing internal value with the `inspect` diagnostics
(`try_road` and the `plans` counter were added for this). Five causes were in the native port
and three in the Java wrapper:

| Where | Cause | Fix |
| --- | --- | --- |
| Native | `SOCPlayerTracker` caches `longestRoadETA`, `largestArmyETA` (500) and `roadsToGo` (20). Only `updateWinGameETAs` refreshes them, but the robber's `selectPlayerToThwart` calls `recalcWinGameETA` alone. The port recomputed award estimates on every projection. | Trackers keep the cache. `win_times` = `updateWinGameETAs`, `projected_win_times` = projection alone (robber). Dev-card scoring leaves the VP-card caches, as the original does. |
| Native | `put_hypothetical` placed a road or settlement without recomputing road lengths. `_update_longest_road` only awards from stored lengths, so connecting roads never won Longest Road in SMART's hypotheticals. | Recompute the builder's length after a road and cut opponents' after a settlement (`SOCGame.putTempPiece`). |
| Native | `removeDependents` recomputes a candidate's road count when another player takes one of its prerequisites. The port kept the earlier (stale) count. | `_incremental_count` mirrors expansion (lower through new prerequisites) and removal (recount over the rest). |
| Native | `planAndDoActionForPLAY1` plays a Largest Army knight *before* planning, and the pending robber then skips planning. The port planned first, refreshing ETA caches the original never refreshed. | Knight check first in the main phase. |
| Native | `makeOffer` with no offer sets `doneTrading` until the next turn starts. The port retried at every decision. | Per-turn `done_trading`. |
| Native + wrapper | `handleMAKEOFFER` counteroffers only on `COUNTER_OFFER`. Both sides countered after any non-acceptance. | Reject on `REJECT_OFFER` in both. |
| Wrapper | `decide()` returned a lone legal action without consulting the brain. Robots on their own server plan every main-phase turn (and discard and robber decision), refreshing tracker ETAs even when they can only end the turn. | Removed the shortcut. |
| Native | `threatMultiplier` was not applied to threatened roads. Stock robots use 1.0, so this changed no decision. | Applied through `smart.THREAT_MULTIPLIER`. |

Before these fixes, 26 of the 64 cases diverged (including 5 without trading).

## Negotiation and graph-order root causes (third follow-up)

The remaining trading divergences, and those from a second corpus on fresh seeds (101, 202,
303 and 404), were traced the same way. The `inspect` diagnostics gained `consider` (the raw
`considerOffer2` response for a pending offer):

| Where | Cause | Fix |
| --- | --- | --- |
| Native + wrapper | Turn resets (selling flags, offers made, targets, plan) ran lazily at the agent's next decision. `handleTURN` runs them when the turn starts, so the lazy reset also wiped the first offers of the new turn from memory. | Reset on the `TURN_ENDED` event (the wrapper's `step` and the native `observe`). |
| Native + wrapper | Gains marked a resource as "selling". The pinned robot never calls `markAsSelling`; it resets the flags from holdings at turn boundaries and when it receives its own hand after a production roll (`handlePLAYERELEMENT` in `ROLL_OR_CARD`). | The same rule on both sides. |
| Native | `wantsAnotherOffer` was cleared every turn. Only our own `makeOffer` (and the constructor) clears it. | Kept across turns. |
| Native | `considerOffer2` checks the receiver's cards first, then plans the receiver's target, and plans the sender's only if needed. Planning refreshes the real trackers' ETA caches, and the port planned both up front. | Original order. |
| Native | The generic "plan if no plan" step ran before trade responses. The original only negotiates there. | Trade phases dispatch first. |
| Native | The race check required a threat from the sender. `considerOffer2` matches piece type and location only. | Original match. |
| Native | Prerequisite lists were rebuilt in board order. `SOCPossiblePiece` lists keep insertion order, which decides between equal-length road paths. | Surviving prerequisites keep their place; new ones follow expansion order, with `getAdjacentEdgesToEdge` neighbour order. |
| Native | With no positive-score initial-road destination the port raised. `planInitRoad` keeps destination coordinate 0, and `getAdjacentEdgeToNode2Away` picks the road. | Ported (it raised in 2% of games against random players, e.g. untrained policies). |

Results, `java-adapter` mode, up to 400 decisions per case
([results/policy_corpus.json](../results/policy_corpus.json)):

| Strategy | Domestic trading | Seeds 12, 31, 47, 58 | Seeds 101, 202, 303, 404 |
| --- | --- | --- | --- |
| FAST | Off | 0/16 diverge, 1280/1280 | 1/16, 1196/1197 |
| SMART | Off | 0/16, 1342/1342 | 0/16, 1245/1245 |
| FAST | On | 0/16, 2197/2197 | 1/16, 2038/2039 |
| SMART | On | 0/16, 2400/2400 | 0/16, 2171/2171 |

The two remaining cases are one position (seed 202, seat 2, with and without trading). Java's
setup-time expansion created two equal-length road prerequisites in an order that a
post-setup rebuild cannot see. Matching it would mean porting `SOCPlayerTracker`'s
incremental updates from the first placement. Tracking during setup with the current rebuild
logic broke 30 of 64 cases, so it was reverted. The regression test
`test_java_adapter_compatibility_decision_sequences` replays one formerly diverging history per
cause.

## Validation and evidence

Component tests compare original Java and native resource knowledge, complete
opening sequences in all seats, candidate prerequisites/conflicts/threats, city
speedups, FAST plans, win projections, SMART scoring and SMART plans. Other tests
cover hidden-information invariance, referee immutability, event sequencing,
foreign discard privacy, and complete native games with/without trading.

The full suite passed **769 tests** at the first follow-up, **787** after the second and **813** after the third (808 plus the 5 Arcade window tests). Final focused native, Java-bridge, OSP and
cleanup checks passed **42 tests** after the last follow-up changes. Optional CI now
includes the native/reference checks on Windows and Linux; remote CI has not
been run from this workspace.

The default-mode shadow comparison sampled 80 decisions per case on game seed
12, seat 0, agent seed 42:

| Strategy | Domestic trading | Exact action agreement |
| --- | --- | --- |
| FAST | Off | 80/80 |
| SMART | Off | 78/80 |
| FAST | On | 76/80 |
| SMART | On | 76/80 |

The first disagreements are the declared Road Building/confirmation choices.
Later disagreements can follow from retained plans after that first divergence.
The comparison applies the Java action to the real game; it is not a win-rate
test. Compatibility-mode comparisons are recorded separately in
`outputs/benchmarks/native-jsettlers-verified-compatible-*/report.json`.

Those compatibility-mode runs completed four games with **501/501 exact
decisions**: FAST 81/81 and SMART 82/82 without domestic trading; FAST 165/165
and SMART 173/173 with trading. This establishes agreement on those histories,
not all boards or all possible decisions.

CARLA-hosted pilots use one repeated board, all four heuristic seats, one
heuristic against three native robots:

| Native opponents | Domestic trading | Heuristic wins | Failures/truncations |
| --- | --- | --- | --- |
| FAST | Off | 2/4 | 0/0 |
| SMART | Off | 3/4 | 0/0 |
| FAST | On | 2/4 | 0/0 |
| SMART | On | 1/4 | 0/0 |

These **16 games are integration pilots**, not evidence of a strength ranking.
A portable numerical summary of these pilots was kept with the local development notes.
Rotations share one board. Sources, hashes, manifests, game replays and failures
are retained in `outputs/benchmarks/native-jsettlers-final-no-domestic` and
`outputs/benchmarks/native-jsettlers-validated-domestic`.

Under the original referee, a FAST pilot and a SMART pilot each completed
against three original SMART opponents, with zero recorded fallbacks/failures.
Both native seats lost their pilot. A separate first SMART attempt timed out
after a prolonged game; it is retained in
`outputs/benchmarks/native-jsettlers-referee-smart`, not silently replaced or
counted as completed. The completed SMART pilot is in the `-smart-second` directory.
The timeout also exposed Windows launcher descendants surviving a parent-only
kill. The harness now terminates the entire process tree on timeout/interruption;
a real parent/grandchild regression verifies that cleanup on Windows.

Remaining release work is the setup-time insertion order above, rare-rule coverage and
external-referee trading conversion. The current evidence does not
establish universal equivalence to unmodified robots on their original server.

## Commands

Native tournaments require only CARLA's ordinary dependencies:

```powershell
carla tournament --config configs/benchmarks/jsettlers_fast_native.yaml --games 4 --seeds 55201
carla tournament --config configs/benchmarks/jsettlers_smart_native.yaml --games 4 --seeds 55201
```

After optional Java setup from the benchmark guide:

```powershell
python scripts/check_jsettlers_policy.py --compatibility java-adapter --max-decisions 300
python scripts/check_jsettlers_policy.py --compatibility java-adapter --trading --max-decisions 300
python scripts/run_jsettlers_benchmark.py --agent jsettlers-fast-native --games 1
python scripts/run_jsettlers_benchmark.py --agent jsettlers-smart-native --games 1
```

The policy comparison always reports whole-policy equivalence as unestablished,
even when every sampled action agrees. Each run retains a source archive and
redacted disagreement snapshots for investigation.
