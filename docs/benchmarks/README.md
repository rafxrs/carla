# Benchmark bots

CARLA ships reference opponents from two published Catan AIs: Catanatron (commit `ecf9311`)
and JSettlers2 (2.6.10). Each comes as a **port**, native Python running on CARLA's engine,
and a **wrapper**, the original code driven through an adapter. Benchmark trained models
against the ports. The wrappers exist to validate the ports. How each bot plays is described
in [agents.md](../agents.md). Changing the bots: [maintenance.md](maintenance.md); how the
ports were built: [archive/](archive/).

## Setup

```powershell
python scripts/setup_benchmarks.py          # pinned Catanatron + JSettlers JAR (SHA-checked) + Java adapter
$env:JSETTLERS_JAR = (Resolve-Path outputs/benchmark-deps/jsettlers-2.6.10-full.jar).Path
$env:CARLA_JSETTLERS_CLASSES = (Resolve-Path outputs/benchmark-deps/jsettlers-classes).Path
$env:PYTHONHASHSEED = "0"                   # Catanatron is only reproducible with fixed string hashing
```

Ports need neither Java nor Catanatron. Wrappers need Git, uv and a JDK. Rebuild the adapter
after editing `adapters/jsettlers/` with `python scripts/setup_benchmarks.py --skip-catanatron`.

## Agents

| Registry name | What it is | Use |
| --- | --- | --- |
| `native-f-upstream`, `native-ab2-upstream` | Catanatron ValueFunction / AlphaBeta(2) ports, upstream chance handling | **benchmark** |
| `native-f`, `native-ab2` | the same ports with exact expectations (a better-defined variant, not upstream) | label as CARLA variants |
| `jsettlers-fast-native`, `jsettlers-smart-native` | JSettlers FAST / SMART ports, with domestic trading | **benchmark** |
| `catanatron-f`, `catanatron-ab2` | original Catanatron code on a reconstructed world | validation |
| `jsettlers-fast-adapted`, `jsettlers-smart-adapted` | original Java robot in a sidecar process | validation |
| `heuristic`, `random`, `ppo:<checkpoint>` | CARLA's own bots and trained policies | |

All bots see only their own cards and public counts. Catanatron search plans in a sampled
world consistent with public inventories. JSettlers bots track known and unknown resources
from events, as the Java robot does.

## Running benchmarks

```powershell
carla tournament --config configs/benchmarks/jsettlers_smart_native.yaml --games 20 --seeds 55201
carla policy-tournament runs/x/latest.pt native-ab2-upstream jsettlers-smart-native heuristic --boards 50
carla evaluate runs/x/latest.pt --opponent jsettlers-smart-native --boards 50
python scripts/run_benchmarks.py --opponents native-f-upstream jsettlers-fast-native --boards 10
```

`run_benchmarks.py` seats one agent against three copies of each bot, every board in all four
seats, and writes a manifest, a per-game journal, replays and a summary. Failed games are
recorded and never replaced.

**On the originals' own engines** (external checks, not the main benchmark):

```powershell
python scripts/run_catanatron_benchmark.py --agent ppo:runs/x/latest.pt --opponents ab2 --boards 25
python scripts/run_jsettlers_benchmark.py --agent ppo:runs/x/latest.pt --games 10
```

The first seats any CARLA agent in Catanatron's engine through a CARLA mirror that replays
every move and is checked against Catanatron's state at each decision. The second runs the
real JSettlers server against original SMART robots. It supports heuristic, random, native
JSettlers and PPO seats, with domestic trading off.

## PPO training and evaluation

- **Training opponents:** `kind: agent` with a registry `name`, or `kind: checkpoint` with a
  `path` (see `configs/experiments/ppo_vs_ports.yaml`). They must run in process, because
  rollout workers are pickled, so the Java wrappers are refused.
- **Validation and evaluation:** `eval.opponents` and `carla evaluate --opponent` accept any
  registry name.
- **Foreign referees:** checkpoints accept `max_turns`, `board_seed` and `catanatron_rules` as
  recorded overrides. Checkpoints trained with domestic trading need
  `--allow-trading-override` on Catanatron's engine.

## Fidelity of the ports

Native against original, comparing every decision of the target seat with the original's
move applied ([results/policy_corpus.json](results/policy_corpus.json)):

| Port | Where compared | Decisions agreeing |
| --- | --- | --- |
| `native-f-upstream` | Catanatron's engine, 16 games | 1381/1381 (100%); whole games reproduce exactly when seated |
| `native-ab2-upstream` | Catanatron's engine, 16 games | 984/986 (99.8%); whole games reproduce exactly when seated |
| `jsettlers-fast-native`, `-smart-native` | CARLA, Java wrapper as reference, 128 games, trading on and off | 13,869/13,871 (one position) |

```powershell
python scripts/check_catanatron_policy.py compare --bots f ab2 --seeds 12 31 47 58 --information open --chance upstream
python scripts/check_jsettlers_policy.py --compatibility java-adapter --seed 31 --seats 0 1 2 3 --max-decisions 400 --trading
python scripts/run_strength_parity.py --boards 160 --output outputs/benchmarks/parity
```

`run_strength_parity.py` checks playing strength rather than single decisions. It seats the
CARLA heuristic against three copies of each port and of each wrapper on shared seeds, and
reports the paired difference in the heuristic's win rate, with a board-clustered interval
and an equivalence verdict (±5 points by default). The same run places the heuristic
relative to every bot.

## Declared differences

Report results with these conditions:

- **Rules.** Under CARLA's referee all bots play official rules. Catanatron's own engine differs
  on bank shortages, discard order, road counting and Longest Road after a cut;
  `EngineConfig.catanatron_rules` reproduces them, and the mirror uses it.
- **Randomness.** Under CARLA, the upstream-mode ports draw from their own seeded RNG, so they
  match the originals statistically, not move for move.
- **Trading protocol.** JSettlers accepts immediately. CARLA collects every response and the
  proposer confirms one partner. `compatibility="java-adapter"` confirms the first legal
  partner as the wrapper does; the default re-checks counteroffers. Catanatron never trades.
- **Known gap.** One JSettlers position diverges: two equal-length road paths, ordered by Java's
  setup-time graph expansion.

## Adding a bot

Implement `carla.agents.base.Agent` (`reset`, `act`; optionally `observe`, `finish`, `close`),
or speak the JSON-lines `carla-agent` v1 protocol from any runtime
(`scripts/example_external_agent.py`, `AgentSpec("external", {"command": [...]})`). Errors,
timeouts and illegal moves abort the game. Pin the upstream code, and compare decisions with
a whole-policy diagnostic before calling a port faithful.

## Tests and licensing

```powershell
python -m pytest tests/agents -q            # Java tests skip without the JSettlers environment
```

`.github/workflows/benchmark-adapters.yml` runs the adapter tests on Windows and Linux.
Reused and ported code is GPL-3.0-or-later; notices are in `adapters/jsettlers/` and
`src/carla/agents/catanatron/vendor/`, and each derived module names its upstream source.
