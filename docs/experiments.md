# PPO experiments and analysis

The sweep and analysis tools below were built for the MLP baseline and apply to every model.
The [full-model training](#full-model-training) section covers the full GNN model's warm start,
reward shaping, behaviour-clone regularization, and league opponents.

Each training run saves CSV metrics, the resolved config, and hardware/software details.
No additional dependencies beyond the existing project extras are needed.
Commands below run from the repository root; `uv run --no-sync` uses the installed CPU or
CUDA environment without changing extras.

## Quick smoke run

```powershell
uv run --no-sync python scripts/run_experiments.py configs/experiments/lambda_study.yaml --output runs/pilot --lambdas 0.95 1.0 --rollout-steps 8 --seeds 0 --set ppo.total_env_steps=64 ppo.epochs=1 model.hidden_dim=32 model.mlp_layers=1 engine.max_turns=12 runtime.envs_per_worker=2 eval.every_env_steps=32 eval.boards=1
uv run --no-sync python scripts/analyze_runs.py runs/pilot --output outputs/pilot
```

This checks that training, evaluation, and plotting run end to end. The budget is too small
to measure playing strength. Inspect `configuration_summary.csv`, `report.json`, and the PNG figures.

## Gamma, lambda, and rollout study

```powershell
uv run --no-sync python scripts/run_experiments.py configs/experiments/lambda_study.yaml --output runs/gamma-lambda-study --gammas 0.99 0.999 --rollout-steps 256 --dry-run
uv run --no-sync python scripts/run_experiments.py configs/experiments/lambda_study.yaml --output runs/gamma-lambda-study --gammas 0.99 0.999 --rollout-steps 256
uv run --no-sync python scripts/analyze_runs.py runs/gamma-lambda-study --output outputs/gamma-lambda-study
```

The default grid is gamma `{0.999}`, lambda `{0.95, 0.99, 1.0}`, rollout length
`{256, 1024}`, and seeds `{0, 1, 2}`: 18 runs. The command above compares two gamma
values at 256 steps, also giving 18 runs. Adding 1024 steps raises it to 36.
Use `--gammas`, `--lambdas`, `--rollout-steps`, and `--seeds` to set the grid;
`--set KEY=VALUE ...` applies shared config overrides.
Grid parameters take precedence over shared overrides. Runs share a validation protocol
and decision budget; their initialization/training seeds differ.

The example budget is 1,048,576 learner decisions per run. Validation runs before training,
every 20,480 decisions, and at completion, against both random and heuristic bots. Ten
boards per opponent means 40 games per opponent at each evaluation. Measure pilot throughput
and evaluation cost before choosing a larger budget. You can disable validation with
`eval.every_env_steps=0`; CSV training/episode logging remains active. The study config sets
`eval.retain_checkpoints=false` to keep only the regularly updated `latest.pt`; validation
curves still use the per-game CSV records.

Training evaluates after the first update crossing a decision threshold, without shortening
the rollout to hit it. Both requested and actual decision counts are saved. Choose an
evaluation interval divisible by all compared collection sizes when possible. With four
envs, 256 versus 1024 steps gives 1024 versus 4096 samples per update: this also changes batch
size and policy refresh frequency. It does not isolate trajectory length alone.

`runtime.num_workers>1` starts independent CPU rollout processes. Each worker batches
`runtime.envs_per_worker` observations; all workers use the same weights for an update,
and the learner waits for all batches before changing the policy. Budget at least one CPU
core per worker plus the learner, then compare `decisions_per_second` in `updates.csv`
with a one-worker pilot. Changing the worker count changes the update batch size and
therefore the experiment condition.

## Outputs and metric meanings

Each run contains:

* `config.yaml`, `git.txt`, `metadata.json`: resolved settings, code version, software/device
  details, run status, and a session history (including resumes on another machine).
* `train.log`: timestamped console progress and validation results, appended on resume.
* `updates.csv`: one row per PPO update, with cumulative learner decisions, optimizer steps,
  collection/update time, cumulative training/evaluation/wall time, throughput, losses,
  approximate sampled KL, clip fraction, pre-clipping gradient norm, explained variance,
  and raw advantage/return/value statistics. CSV logging occurs every update regardless
  of the console logging interval.
* `episodes.csv`: one row per completed training episode, including its env/episode ID,
  actual completion decision, seat, opponents, win/loss/truncation, accumulated return,
  learner decisions, turns, and final VP. Accumulators span rollouts and checkpoints.
* `evaluation_games.csv`: one row per validation game, with actual training decisions,
  board/game seed, seat, opponent, greedy/sampled mode and outcome. The checkpoint field
  is blank when retention is disabled because `latest.pt` can later be overwritten.
* `latest.pt`: the resumable training checkpoint, rewritten every
  `runtime.checkpoint_every` updates and on completion, plus an initial save when validation
  is enabled. The baseline's 100-update interval is about 102,400 decisions with four
  environments and 256 steps each (roughly 3.4 minutes at 500 decisions/s).
* `checkpoints/step_XXXXXXXXXXXX.pt`: optional full checkpoints at validation points,
  including step zero, when `eval.retain_checkpoints=true`. At a 20,480-decision interval,
  a 50-million-decision run would retain roughly 2,442 files. A 4,096-decision CPU pilot
  produced roughly 7.4 MB per checkpoint, implying about 18 GB per run at that cadence;
  actual size depends on game state and model.

Entropy diagnostics are measured during collection, before the update. `head_*_entropy`
averages conditional entropy where that head has a valid choice (unavailable dummy masks
are excluded). `main_type_entropy` and `main_type_max_probability` use only MAIN decisions
with multiple legal types. End-turn statistics use MAIN decisions where a road, settlement,
or city is legal. Blank values mean no eligible observations, not zero entropy.

Training time is collection plus optimizer time; wall time also includes evaluation and
I/O during the training session. Throughput excludes evaluation. Pauses between sessions
are excluded. Evaluate both sample efficiency and elapsed compute cost. Value loss and
explained variance refer to lambda-dependent training targets, not observed true values.

### CPU pilot and run time

On my PC's CPU, a 4,096-decision run with the standard 256-wide MLP measured 381
decisions/s with one worker and 515 with two, counting collection and PPO updates.
With the same four environments in both runs, the two-worker run spent 15.1 seconds up to
its final update versus 10.8 seconds for one worker: starting the extra processes costs
more than it saves on a short run.

If those rates hold, **50 million decisions would take about 36 hours with one CPU worker
or 27 hours with two**, before validation and checkpoint time. At 515 decisions/s, the
1,048,576-decision pilot would take about 34 minutes per run, or about 10 hours for the
18-run gamma/lambda study above. These are rough estimates from a short CPU run. I have
not timed training on the RTX 5080 or Margaret yet.

## Resume and move between laptops

Repeat the exact sweep command with `--resume`. The manifest skips completed cases and
resumes interrupted cases from `latest.pt`. If a case failed before its first checkpoint,
it restarts from its seed. A changed grid or config requires a new sweep output directory.
The manifest stores relative run paths; copying the entire sweep directory preserves it.
Each laptop needs its own installed environment; do not copy `.venv` between machines.

For an individual run:

```powershell
uv run --no-sync carla train configs/experiments/lambda_study.yaml --resume runs/<run>/latest.pt ppo.total_env_steps=2097152
```

On resume, you can extend `ppo.total_env_steps` and change the device, log interval,
checkpoint interval, or `ppo.micro_batch_size`. Keep the seed, game rules, model, opponents, evaluation settings,
other PPO settings, `runtime.num_workers`, and `runtime.envs_per_worker` the same.
Checkpoints save episode accumulators, optimizer counters, CSV committed-row counts,
and cumulative timers. On resume, uncommitted CSV tails are removed and backed up as
`*.csv.recovery.bak`; missing committed rows cause an error. Copy CSVs with the checkpoint.
Future retained checkpoint files from a rolled-back attempt are not deleted; only records
in the reconciled CSVs belong to the resumed trajectory. Legacy checkpoints can resume,
but omit the first partial episode because they lack its earlier statistics.

Validation restores Python/NumPy/PyTorch RNG state and module modes, and does not update
weights or training environments. Changing evaluation frequency should not change training
actions. Greedy validation measures deployed greedy behavior; sampled validation uses a
separate fixed RNG and can be selected with `eval.deterministic=false`.

## Full-model training

The full model (`configs/experiments/full_trade.yaml`) is the GNN with set attention, the GRU
belief, and the GRU trade decoder, with multi-card domestic trading on. Its later stages are
forks: each starts a new run from an earlier run's weights.

### Warm starts and long runs

`--init-from` starts a new run from another run's policy weights. The optimizer, counters,
environments, and journals start fresh, so opponents, rewards, and PPO settings may change
between stages; the model config must match exactly.

```powershell
uv run --no-sync carla train configs/experiments/full_trade_league.yaml --init-from runs/<run>/checkpoints/step_000010006528.pt
```

For long runs, `scripts/train_with_restarts.py` runs `carla train` and resumes the newest run
under `runtime.output_dir` from `latest.pt` whenever the process dies (for example after a GPU
reset). `--init-from` applies to the first launch only; overrides pass through to every resume.

```powershell
uv run --no-sync python scripts/train_with_restarts.py configs/experiments/full_trade_league.yaml runtime.output_dir=runs/full_trade_league --init-from runs/<run>/checkpoints/<step>.pt
```

### Behaviour cloning

`carla bc` plays games in which the heuristic bot makes the learner's decisions against the
config's opponents, then trains the policy on those decisions (action log-likelihood), the value
head on the expert's discounted returns, and the belief head. Use the resulting `bc.pt` as an
`--init-from` warm start or as a reference policy.

```powershell
uv run --no-sync carla bc configs/experiments/full_trade_own_heuristic.yaml --output runs/bc/full_model_heuristic --games 600 --workers 4 runtime.device=cuda
```

### Reward shaping

`env.shaping` adds potential-based shaping F = gamma * Phi(s') - Phi(s) between the learner's
decisions (Phi = 0 once the game is over, so the optimal policy is unchanged; `env.shaping_gamma`
must equal `ppo.gamma`):

| `env.shaping` | Phi / `env.shaping_coef` |
| --- | --- |
| `own` | own VP, hidden VP cards included |
| `margin` | own VP minus the top opponent's public VP |
| `production` | own VP plus `env.production_weight` times own expected resources per round |

`production` credits a settlement or city for its future income as soon as it is built. An
average settlement yields about 1.1 resources per round, so weight 0.5 values it at about
0.55 VP on top of its point.

### Behaviour-clone regularization and memory

`ppo.reference_kl_coef` adds that coefficient times KL(reference || policy) over the masked
action type to the loss, with the frozen reference loaded from `ppo.reference_checkpoint`
(for example the behaviour clone). The forward direction penalises the policy for nearly ruling
out a type the reference plays, such as ending the turn to save up, while where to build, which
trade, and which card stay free. Rollout workers run the reference with its own recurrent
state and store its action-type log-probabilities with each decision; `updates.csv` logs the
mean as `reference_kl`. A coefficient of 0.05 kept the policy saving without stalling it; 0.2
pulled it too close to the clone.

`ppo.micro_batch_size` splits each minibatch into chunks whose gradients are accumulated, with
each chunk's mean loss weighted by its share of the minibatch. The update is the same; peak GPU
memory drops (the full model peaks at about 14 GB for a 2048-decision minibatch without it).
`ppo.trade_entropy_coef` weights the trade decoder's token entropy separately from
`ppo.entropy_coef`.

### League opponents

A pool of only `random` and `heuristic` specs gives each environment fixed bots for the whole
run. Adding any policy kind makes the pool a league: each opponent seat of each game samples a
spec in proportion to its weight.

| Kind | Opponent | Label in `episodes.csv` |
| --- | --- | --- |
| `random`, `heuristic` | that bot | the bot's name |
| `self` | the current policy | `self` |
| `snapshot` | a frozen copy of the policy, taken every `opponents.snapshot_every` updates (the newest `opponents.max_snapshots` kept) | `snapshot:<update>` |
| `checkpoint` | the fixed policy in `path`, with the same engine rules and `env.max_events` | `name`, or the file stem |

Policy opponents sample their actions and keep their own belief state. Each rollout process
batches the waiting opponent decisions of its environments per network, so every network runs
once per round. Snapshots live in the rollout processes: games starting in one rollout share
one snapshot, and snapshots are not saved in checkpoints, so after a resume `snapshot` seats
play the current policy until the next snapshot. Expect slower collection than against bots:
each learner decision brings about two opponent network passes.

### Full-model configs

| Config | Starts from | Change |
| --- | --- | --- |
| `full_trade.yaml` | scratch | full model against three random bots |
| `full_trade_own.yaml`, `full_trade_margin.yaml` | scratch | own-VP or VP-margin shaping |
| `full_trade_own_mixed.yaml` | a stage-1 checkpoint | two random bots and one heuristic bot per table |
| `full_trade_own_heuristic.yaml` | a stage-1 checkpoint | three heuristic bots |
| `full_trade_production_heuristic.yaml` | the behaviour clone | production shaping (weight 0.5) |
| `full_trade_production_push.yaml` | the previous stage | production weight 1.0, `max_grad_norm` 1.0 |
| `full_trade_production_bc_kl.yaml` | the previous stage | KL 0.05 toward the behaviour clone (`_strong`: 0.2) |
| `full_trade_league.yaml` | the previous stage | league: 40% self, 15% snapshots, two fixed checkpoints, 30% heuristic |

`scripts/heuristic_figures.py` plays games between heuristic bots and draws how the bot places
its opening (`docs/img/heuristic_setup.png`) and its turn as a state machine with measured branch
frequencies (`docs/img/heuristic_turn.png`); it needs Graphviz.

## Analysis and final tests

Analysis accepts individual run directories or a sweep root. It rejects duplicate records,
nonfinite update metrics and missing seat rotations; incomplete runs are skipped unless
`--include-incomplete` is specified. Use one attempt per configuration/training seed.

Outputs include:

* `evaluation_points.csv`: results at each validation point.
* `run_summary.csv`: each run's last evaluation per opponent/mode.
* `configuration_summary.csv`: equal-weight training-seed means, observed seed range/std,
  and exploratory hierarchical bootstrap intervals (runs then boards, four seats together).
* PNGs: win rate/VP/truncation curves against decisions and wall time, optimizer/entropy
  diagnostics, and rolling training-episode returns/VP/turns/truncations.
* `report.json`: data warnings and interpretation limits.

Thin curves show individual seeds. Shading is their observed range, not a confidence
interval. Aggregate curves interpolate only over common support, with no extrapolation.
Three training seeds are exploratory evidence; narrow or degenerate bootstrap intervals,
especially with unanimous outcomes, do not establish certainty. Summaries use the last
evaluation, not whichever validation point happened to score highest. Do not select
the best configuration from optimizer loss alone.

After selecting configurations, copy the final checkpoint and evaluate it on separate test seeds:

```powershell
New-Item -ItemType Directory -Force runs/<run>/checkpoints
Copy-Item runs/<run>/latest.pt runs/<run>/checkpoints/final.pt
uv run --no-sync carla evaluate runs/<run>/checkpoints/final.pt --opponent random --boards 250 --seed 20000 --csv runs/<run>/evaluation_test_games.csv
uv run --no-sync carla evaluate runs/<run>/checkpoints/final.pt --opponent heuristic --boards 250 --seed 20000 --csv runs/<run>/evaluation_test_games.csv
uv run --no-sync python scripts/analyze_runs.py runs/gamma-lambda-study --output outputs/gamma-lambda-test --split test
```

250 boards give 1,000 games per opponent and trained seed. Test exports are separate from
the training journal and survive resume reconciliation. Analysis discovers
`evaluation_*_games.csv` beside the journals. A test export rejects the validation seed and
duplicate checkpoint/game records. Add `--sampled` to evaluate sampled rather than greedy
actions. Keep the copied final checkpoint: a later resume overwrites `latest.pt`. The CLI's
printed Wilson interval remains a game-level approximation; use the
saved board-grouped and training-seed results when making final comparisons.

For a direct four-player comparison, put checkpoints or fixed bots in the four slots:

```powershell
uv run --no-sync carla policy-tournament runs/<run-a>/latest.pt runs/<run-b>/latest.pt heuristic random --boards 250 --seed 20000 --csv outputs/policy-tournament.csv
```

Each board is played in four rotations, so every slot occupies every seat once. All
checkpoints must use the same engine and environment settings. The CSV records every
slot's outcome and victory points; the printed game-level Wilson intervals do not
account for shared boards, so use board-grouped comparisons for conclusions.
