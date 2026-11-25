# Optional lightweight training diagnostics

Enable with the single new configuration field (default `false`):

```yaml
runtime:
  collect_training_diagnostics: false
```

For a short smoke run:

```bash
uv run carla train configs/smoke.yaml runtime.collect_training_diagnostics=true
```

The same override applies to the prepared 300k configurations when a starting checkpoint is
chosen. Inspect the configuration without launching an experiment:

```bash
uv run carla config configs/experiments/full_trade_strategic_economy_05.yaml runtime.collect_training_diagnostics=true
```

## Collection and performance boundary

`CatanEnv` optionally owns one fixed-size learner accumulator through its AEC environment.
The AEC layer captures the actor/phase, applies the action once, then forwards that action
and its existing engine events to the accumulator. Invalid actions are not counted. This
also sees passive income, theft, and trades confirmed by opponents between learner decisions.
The accumulator never generates actions, consumes RNG, reads opponents' hands, or changes
observations/rewards. It uses only learner-owned facts and already-emitted events.

There is no replay, legal-set query, hypothetical purchase/offer analysis, production
capacity calculation, board traversal, economic trajectory, belief calculation, neural
inference, extra transition, tensor creation, or CPU/GPU synchronization for these counters.
Learner VP is cached and refreshed with the engine's constant-time VP helper only after
events that can change it. Integer action/resource counters have fixed sizes. A flat record
is constructed only at episode completion.

With the flag off, no accumulator arrays or diagnostic output list/journal are created.
The engine step has a nullable-observer check; existing training telemetry remains unchanged.
Detailed retrospective [evaluation diagnostics](strategic_diagnostics.md) remain evaluation-only.

The supported trainer is synchronous PPO with embedded fixed bots (`CatanEnv`), including
the existing curriculum pool and multi-process/multi-environment workers. Bare `CatanAECEnv`
self-play has no designated learner and does not enable these counters. The existing planned
self-play worker/pool is unchanged.

## `training_diagnostics.csv` schema

One row is written per completed or truncated learner episode, using the existing episode
journal mechanism, after the rollout reaches the learner. There are no per-decision writes.
Existing `episodes.csv`, `updates.csv`, and evaluation CSV fields are not changed by this
feature. The exact schema is:

```python
(*EPISODE_FIELDS, "worker_id", "policy_version", "opponent_composition", *TRAINING_COUNTER_FIELDS)
```

`EPISODE_FIELDS` is defined in `src/carla/rl/metrics.py`; the counter schema and its fixed
resource/action expansions are in `src/carla/envs/training_diagnostics.py`.
The journal has 140 columns: 16 existing/added metadata columns and 124 diagnostic columns.

Metadata reuses the existing episode identifiers:

| Column | Meaning |
|---|---|
| `update` | Learner update that consumes the terminating rollout |
| `env_steps` | Existing global learner-decision index of the terminating action |
| `env_id` | Existing global environment ID; worker ID × environments/worker + local ID |
| `episode_id` | Existing per-environment episode counter |
| `game_seed` | Actual engine seed |
| `seat` | Learner's absolute seat for this episode |
| `opponent` | Existing comma-separated opponent names |
| `outcome`, `winner` | Existing win/loss/truncation outcome and winner (`-1` if none) |
| `return` | Existing accumulated learner reward |
| `learner_decisions` | Existing episode length in learner decisions |
| `turns` | Game duration in completed global player turns |
| `victory_points` | Existing final learner total VP |
| `worker_id` | Rollout worker; zero for single-process collection |
| `policy_version` | Policy version used for the terminal learner decision, not every action of the episode |
| `opponent_composition` | JSON array of bot names in ascending non-learner absolute seat order |
| `diagnostics_complete` | Whether collection covers the entire episode |

Raw scalar counter columns:

```text
diagnostic_learner_decisions
main_phase_decisions
completed_learner_turns
learner_turns_at_two_vp_or_less
domestic_trade_decisions
opening_trade_proposals
counteroffers
trade_invitations
trade_acceptances
trade_rejections
trade_confirmations
trade_cancellations
negotiations_opened
negotiations_completed
domestic_trades_as_partner
maritime_trades
setup_settlements
setup_roads
settlements_built
cities_built
roads_built
free_roads_built
purchased_roads
dev_cards_purchased
robber_moves
final_victory_points
first_settlement_learner_turn
first_city_learner_turn
```

Additional fixed expansions:

- `action_<action_type>` for all lowercase `ActionType` names, including setup/response actions.
- `dev_played_<card_type>` for all lowercase `DevCard` names.
- `vp_3_learner_turn` through `vp_10_learner_turn`.
- `cards_<source>` and `cards_<source>_<resource>` for every source below. Resource suffixes
  are `lumber`, `brick`, `wool`, `grain`, and `ore`.

```text
gained_dice
gained_setup
gained_domestic_trade
gained_maritime_trade
gained_robber_theft
gained_development
lost_discard
lost_robber_theft
lost_domestic_trade
lost_maritime_trade
lost_development
```

`lost_development` additionally accounts for resources removed by another player's Monopoly.
Source totals are sums of the five per-resource counters; gains are not netted against losses.
These source counters are not a complete hand ledger: construction and development-card
purchase payments are outside the requested resource-loss categories.
Domestic exchange terms are reversed for the responding partner. Only actual `TRADE_EXECUTED`
events count completed trades, not acceptances or offers. Year of Plenty and Monopoly gains
are development effects; neither is counted as dice production. Normal post-setup roads
include paid roads and development-card free roads; setup roads have a separate count.

`negotiations_opened` includes opening proposals and invitations, excluding counteroffers.
`negotiations_completed` counts executed negotiations initiated by this learner;
`domestic_trades_as_partner` counts completed responding participation. Domestic decision
counts include protocol responses/conclusions, so they are distinct from voluntary opening
proposals. `main_phase_decisions` is based on the actual pre-action phase.

Milestones use the number of **completed learner turns at the event**, starting at zero.
A first-turn construction is `0`; an unreached milestone is `None` (empty CSV cell).
Award changes can reach VP milestones during an opponent's turn. Low-VP turns sample the
learner's total VP at its own turn end. A winning partial turn is not a completed turn.

## Checkpoint, reset and recovery

Each environment's accumulator is pickled with the already-checkpointed environment state,
including worker snapshots. Resuming with collection enabled keeps the active counters;
resetting a game creates fresh counters bound to its newly selected learner seat. `--init-from`
loads weights only and starts fresh environments, counters, episode IDs and journals.

Older configurations default to disabled. Older pickled environments lacking the new fields
are upgraded without replay. If collection is newly enabled on an active old episode, its
first diagnostic row is explicitly `diagnostics_complete=False`. Counters then cover only
the observed suffix: `diagnostic_learner_decisions` can differ from the full episode length.
All first-construction/VP milestone times are missing in an incomplete row; earlier first
occurrences cannot be recovered without replay. The next new episode is complete.
Very old checkpoints without episode trackers may also have an incomplete ordinary episode
record; the diagnostic row still uses the tracker's episode ID and is marked incomplete.

The new journal participates in checkpoint `log_rows` and existing CSV recovery. Rows flushed
after a durable checkpoint are backed up/truncated on resume like the existing journals;
already committed records are retained, preventing duplicate completed episodes. Disabling
collection does not open/write its CSV and preserves its committed row count for later
re-enabling. Newly enabled active episodes are then marked incomplete again.

## Analysis against cumulative decisions

Join with episodes using `(env_id, episode_id, game_seed)` and plot against `env_steps`.
For aggregate comparisons, first filter `diagnostics_complete == True`, sum numerators and
denominators over the desired training interval, then divide. Do not average per-episode
ratios or substitute zero for an undefined ratio. A truncated game may give the learner
zero completed turns.

| Metric | Numerator | Denominator |
|---|---|---|
| Opening proposals per learner turn | `opening_trade_proposals` | `completed_learner_turns` |
| Domestic decisions (%) | 100 × `domestic_trade_decisions` | `diagnostic_learner_decisions` |
| Domestic completions per learner turn | `negotiations_completed + domestic_trades_as_partner` | `completed_learner_turns` |
| Negotiation completion rate | `negotiations_completed` | `negotiations_opened` |
| Settlements/cities/roads per learner turn | `settlements_built` / `cities_built` / `roads_built` | `completed_learner_turns` |
| Development purchases per learner turn | `dev_cards_purchased` | `completed_learner_turns` |
| Actual dice production per learner turn | `cards_gained_dice` | `completed_learner_turns` |
| Turns at ≤2 VP (%) | 100 × `learner_turns_at_two_vp_or_less` | `completed_learner_turns` |

For first construction, retain both reach frequency (nonmissing milestone count / episode
count) and the distribution of reached milestone times. Never treat missing times as zero.
No plotting framework is required; all raw counts and denominators are retained in CSV.

## Reproducible benchmark

```bash
uv run python scripts/benchmark_training_diagnostics.py --env-steps 20000 --training-steps 4096 --repeats 5 --device cpu --cpu auto --output runs/training_diagnostics_benchmark.json
```

The harness warms both modes with full-size PPO minibatches, alternates disabled/enabled execution order between pairs,
uses the same seeds/settings in each pair, and verifies equal final environment states and
policy weight digests. It measures environment decisions/second including terminal counter
snapshots, and end-to-end small training runs including initialization, CSVs and final
checkpoint writes. Temporary runs are isolated and removed; an existing report path is
refused. The default CPU benchmark uses GNN hidden size 64, two layers, belief size 32, set
attention and trade GRU, four environments, H25/H25/H50 opponents and strategic rewards.
These reduced dimensions make it a small benchmark, not one of the 300k experiments.

On Linux/WSL, `--cpu auto` pins the benchmark process to its first allowed CPU to reduce
core-migration noise; `--cpu none` disables affinity. This affects only the benchmark process.
Single-thread Torch settings are identical in both modes. Environment trials finish before
training trials begin. Both wall-clock and process CPU time are recorded, alongside the
trainer's collection/optimization timings; initialization and journal/checkpoint costs remain
included in the end-to-end wall-clock measurement.

The JSON report contains every trial, median throughput, paired percentage differences,
approximate per-environment Python counter memory and pickle size, extra checkpoint bytes,
and diagnostic CSV bytes per completed episode. Counter memory is CPU-only. CPU timings do
not establish GPU/Colab throughput; rerun with `--device cuda` on the target machine. The <2%
overhead goal is measured, not assumed. Do not run other training/test jobs concurrently
with a benchmark used for performance conclusions.

### Measured local results (2026-11-25)

The final run used WSL, CPU inference/training, one Torch thread and CPU affinity `[0]`.
There were five matched seed pairs, each with 20,000 environment decisions and 4,096 training
decisions per mode. Every environment-state and learned-weight equality check passed.
CUDA was unavailable; these results do not establish Colab/GPU overhead.

| Measurement | Disabled median decisions/s | Enabled median decisions/s | Median paired throughput difference | Median paired elapsed overhead |
|---|---:|---:|---:|---:|
| Environment, wall clock | 4,204.39 | 3,888.62 | -2.66% | +2.73% |
| End-to-end training, wall clock | 182.20 | 178.21 | -2.19% | +2.24% |
| Collection + optimization only, wall clock | 199.86 | 195.13 | -2.52% | +2.58% |
| End-to-end training, process CPU time | 199.27 | 194.38 | -2.46% | +2.52% |

The first two rate columns are independent medians; the difference/overhead columns are
medians of same-seed paired ratios, rather than ratios of those independent medians.
The <2% end-to-end goal is **not established** by these measurements. Training wall-clock
throughput differences ranged from -36.90% to +66.12% across pairs, with large variation in
optimizer time despite identical learned weights and no diagnostic work inside PPO updates.
Environment pairs ranged from -10.06% to +2.87%. Retain this variation when interpreting the
median; a controlled target-machine run is needed for a reliable GPU performance conclusion.

The initial short benchmark showed a -6.85% median paired environment throughput change,
triggering investigation. A 10,000-decision cProfile run identified unnecessary callbacks
for opponent actions and repeated checks/allocation for irrelevant events. The implementation
now only forwards learner actions, uses precomputed event sets, and quickly skips unrelated
opponent events. A longer unpinned rerun measured -4.67% paired environment throughput;
the final affinity run above measured -2.66%. Unpinned training timings were also highly
variable and are not evidence of a diagnostic-induced speedup.

Measured additional storage/memory in the final run:

- Approximately 2,404 bytes of Python counter heap per active environment (recursive estimate).
- 1,072 bytes for a standalone serialized active counter.
- Median 2,944 additional bytes in the four-environment training checkpoint; shared pickle
  objects mean this is not four times the standalone size.
- A 3,201-byte CSV header and approximately 428.37 bytes per completed episode row with
  these values. Row sizes vary with counter magnitudes and metadata.

Local raw reports are preserved in `runs/training_diagnostics_benchmark_20261125.json`,
`runs/training_diagnostics_benchmark_20261125_optimized.json`, and
`runs/training_diagnostics_benchmark_20261125_affinity.json`. These generated artifacts are
under the repository's existing ignored `runs/` directory. The harness can reproduce the
method with the command above; its temporary training directories are removed automatically.

## Validation

Validated locally in the existing WSL virtual environment with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`:

```text
uv run --no-sync pytest tests/envs/test_training_diagnostics.py tests/rl/test_training_diagnostics.py -n 2 -q
15 passed in 194.33s

uv run --no-sync pytest -n 4 -q
810 passed, 5 skipped in 361.80s
```

The skips were four unavailable-CUDA GNN cases and the optional Arcade GUI smoke test.
Ruff and Black checks passed for the ten added/modified Python files in this supplement;
`git diff --check` also passed. The `carla config` CLI accepted the dotted runtime override.

Coverage includes disabled defaults and legacy configs, identical observations/rewards/game
and opponent RNG states, unchanged legality-query counts, actual trade/construction events,
separate setup construction, resource source attribution, VP milestones, termination versus
truncation, independent environments, multi-process metadata, identical enabled/disabled
training weights and episode records, active-counter checkpoint continuation, legacy partial
episodes, journal recovery without duplicates, feature toggling, and fresh `--init-from` state.

## Supplemental implementation files

Modified: `src/carla/config/schema.py`, `src/carla/envs/catan_env.py`,
`src/carla/rl/metrics.py`, `src/carla/rl/rollout.py`, `src/carla/rl/process_rollouts.py`,
`src/carla/rl/train.py`, and `README.md`.

Added: `src/carla/envs/training_diagnostics.py`, `tests/envs/test_training_diagnostics.py`,
`tests/rl/test_training_diagnostics.py`, `scripts/benchmark_training_diagnostics.py`, and
`docs/training_diagnostics.md`.

The reward, model, PPO objective and evaluator changes from the previous task are preserved.
