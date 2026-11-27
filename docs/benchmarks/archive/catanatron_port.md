# Native Catanatron ports against the originals

> **Archived, 26 November 2026.** A record of how this port was built and validated, written in rounds as the work progressed. Earlier sections can be superseded by later ones. For current usage see [the benchmark guide](../README.md); for maintenance see [maintenance.md](../maintenance.md).

`native-f` and `native-ab2` are CARLA implementations of Catanatron's ValueFunctionPlayer and
depth-two AlphaBetaPlayer (pinned commit `ecf9311`). `catanatron-f` and `catanatron-ab2` run the
original code under CARLA's referee. This page covers the comparison on **Catanatron's own
engine**, the two chance modes of the ports, and the engine's Catanatron rules profile.

| Registry name | Chance handling | Use |
| --- | --- | --- |
| `native-f`, `native-ab2` | exact expectations (default) | a better-defined variant; CARLA benchmarks |
| `native-f-upstream`, `native-ab2-upstream` | upstream's, quirks included | faithful stand-ins for the originals |

## Results

Native against original on Catanatron's engine, four seeds × four seats, open information,
`PYTHONHASHSEED=0`, AB2's wall-clock deadline lifted
([results/policy_corpus.json](../results/policy_corpus.json)):

| Port | Chance | Decisions agreeing | Robber | Discard | Main | Setup, pre-roll |
| --- | --- | --- | --- | --- | --- | --- |
| F | exact | 1368/1381 (99.1%) | 52/64 | 223/223 | 675/675 | 418/419 |
| F | **upstream** | **1381/1381 (100%)** | 64/64 | 223/223 | 675/675 | 419/419 |
| AB2 | exact | 958/986 (97.2%) | 26/39 | 105/113 | 505/512 | 322/322 |
| AB2 | **upstream** | **984/986 (99.8%)** | 39/39 | 113/113 | 510/512 | 322/322 |

Seated with Catanatron's RNG, the upstream ports reproduce whole games. For F on seeds 3 and 7,
and AB2 on seeds 3 and 7, every action and chance outcome matches an all-original game
(`test_upstream_chance_ports_reproduce_original_games_exactly`). The two remaining AB2
decisions choose between roads. Upstream orders roads by networkx set iteration at inner
search nodes, which the port does not reproduce.

## What the upstream mode reproduces

1. **F scores one sampled outcome.** `game.copy().execute(action)` draws through the live
   game's shared `random`: dice as two `randint(1, 6)`, a steal as `choice` over the victim's
   cards (wood, brick, sheep, wheat, ore), a development card as the top of the pre-shuffled
   deck. So F's scoring advances the real game's RNG and peeks at the next card.
2. **AB2's spectrum quirks.** Its 11 roll branches execute without an action record, so each
   rolls *random* dice and is weighted by its intended sum's probability. Steals are 1/5 per
   resource type; a type the victim lacks raises before the robber moves, so that branch is the
   unchanged position. Every dev-card branch pops the same top card. All moves' spectra at a node
   are expanded before any is searched, which fixes the order of RNG draws.
3. **AB2's search.** Alpha-beta with the same bounds passed into chance children (unsound for
   expectiminimax, but deterministic), terminal at any player's 10 actual VP, and inner moves in
   upstream's type order.
4. **Arithmetic.** `DICE_PROBAS` is built by adding 1/36 per dice pair (so P(7) ≠ 6/36 exactly),
   `proba_point = 2.778 / 100`, production grouped per resource in upstream's order, and terms
   summed in `base_fn`'s order. At values near 1e15 these change exact ties.

Hosted on Catanatron's engine, `CarlaPlayer` gives an upstream-mode port the live RNG
(`share_rng=True`, as the original consumes it) or a copy (comparisons). With open information
it also gives the deck order. Under CARLA's referee the port draws from its own seeded RNG,
so parity is distributional there.

## Commands

```powershell
python scripts/check_catanatron_policy.py compare --bots f ab2 --seeds 12 31 47 58 --information open --chance upstream
python scripts/check_catanatron_policy.py play --bots f --seeds 12 31 --chance upstream
python scripts/run_catanatron_benchmark.py --agent native-ab2-upstream --opponents ab2 --boards 10
```

All three re-run themselves with `PYTHONHASHSEED=0` and lift AB2's 20-second deadline.

## Catanatron-hosted bridge and rules profile

`carla.agents.catanatron.host.CarlaPlayer` is a Catanatron `Player` wrapping any CARLA `Agent`.
By default it keeps a **mirror**: a CARLA game replaying every executed action record with its
chance outcome. Agents get a game with a real history plus redacted events, so PPO checkpoints
and native JSettlers play unchanged (see [the benchmark guide](../README.md#ppo-training-and-evaluation)). At every decision the mirror is
checked against a direct conversion of Catanatron's state, and any disagreement raises.

The checks exposed rule implementations that differ from CARLA's official ones. The opt-in
engine profile `EngineConfig.catanatron_rules` reproduces them (it is omitted from serialization
when off, so existing records are unchanged):

| Rule | Official (CARLA default) | Catanatron ecf9311 |
| --- | --- | --- |
| Bank shortage | nobody is paid, except a single claimant, who takes the rest | nobody is paid |
| Discard order after a 7 | turn order from the roller | seat order from seat 0 |
| Road length | trails may end at an opponent's building | trails never enter an opponent's corner; lengths are running maxima per component |
| Longest Road after a road is split | goes to a unique road of at least 5, else set aside | goes to the longest road, with no 5-road minimum; that holder keeps it until a road of at least 5 beats its length |

With the profile, Catanatron's transitions replay exactly in 94 of 100 games. The rest differ
only in road lengths, because Catanatron's incremental component bookkeeping decides whether
a trail may start at an opponent's corner. The bridge adopts those two fields at each
decision and announces a changed holder as a `LONGEST_ROAD_CHANGED` event. 120 mirrored games
with heuristic, native F and native FAST seats completed under the strict checks (48 road
corrections). Before this, games under CARLA's referee with Catanatron bots used official
rules. That remains the default and is a declared benchmark condition, not an equivalence.

## Recommendations status

| Recommendation | Status |
| --- | --- |
| Upstream-chance mode, drawing through Catanatron's RNG when hosted | done (`chance="upstream"`); F 100%, whole games identical |
| Mirror AB2's spectrum quirks | done; AB2 99.8% |
| Deterministic search bound in comparisons | done (deadline lifted in the scripts) |
| Upstream action order and tie rule | done at the root; type order at inner nodes. Within-type road and maritime orders depend on networkx/set iteration and remain |
| Report open-mode agreement as fidelity, view-mode results as the benchmark | done (this page, `--information`) |
| Corpus gate in CI | the whole-game identity tests (F fast; AB2 marked `slow`) and the legality and mirror tests run with the adapter workflow |

Not attempted: a history-conditioned belief for `view` mode, and bit-exact inner road and
maritime ordering.

## Reproducibility

Catanatron's decisions depend on Python's per-process string hashing. Without a fixed
`PYTHONHASHSEED`, the same seed gave three different `catanatron-f` games under CARLA's
referee in three processes, and different agreement counts on Catanatron's engine. The
benchmark scripts re-run themselves with `PYTHONHASHSEED=0` when it is unset, the benchmark
manifest records the value, and `CatanatronAgent` warns when it is unset (for example under
`carla tournament`). Export it before such runs:

```powershell
$env:PYTHONHASHSEED = "0"
```
