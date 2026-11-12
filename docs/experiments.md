# MLP PPO experiments and analysis

Training now writes CSV journals and machine/software provenance alongside the resolved
configuration. No additional dependencies beyond the existing project extras are needed.
Commands below run from the repository root; `uv run --no-sync` uses the installed CPU or
CUDA environment without changing extras.

## First check: a small complete pipeline

```powershell
uv run --no-sync python scripts/run_experiments.py configs/experiments/lambda_study.yaml --output runs/pilot --lambdas 0.95 1.0 --rollout-steps 8 --seeds 0 --set ppo.total_env_steps=64 ppo.epochs=1 model.hidden_dim=32 model.mlp_layers=1 engine.max_turns=12 runtime.envs_per_worker=2 eval.every_env_steps=32 eval.boards=1
uv run --no-sync python scripts/analyze_runs.py runs/pilot --output outputs/pilot
```

This validates the pipeline, not playing strength: the tiny turn limit and budget are for
testing only. Inspect `configuration_summary.csv`, `report.json`, and the PNG figures.

## Lambda and rollout study

```powershell
uv run --no-sync python scripts/run_experiments.py configs/experiments/lambda_study.yaml --output runs/lambda-study --dry-run
uv run --no-sync python scripts/run_experiments.py configs/experiments/lambda_study.yaml --output runs/lambda-study
uv run --no-sync python scripts/analyze_runs.py runs/lambda-study --output outputs/lambda-study
```

The default grid is lambda `{0.95, 0.99, 1.0}`, rollout length `{256, 1024}`, and seeds
`{0, 1, 2}`: 18 sequential runs. Gamma stays 0.999. Override the grid with `--lambdas`,
`--rollout-steps`, and `--seeds`; use `--set KEY=VALUE ...` for shared config overrides.
Grid parameters take precedence over shared overrides. Runs share a validation protocol
and decision budget; their initialization/training seeds differ.

The example budget is 1,048,576 learner decisions per run. Validation runs before training,
every 20,480 decisions, and at completion, against both random and heuristic bots. Ten
boards per opponent means 40 games per opponent at each evaluation. Measure pilot throughput
and evaluation cost before choosing a larger budget. You can disable validation with
`eval.every_env_steps=0`; CSV training/episode logging remains active.

Training evaluates after the first update crossing a decision threshold, without shortening
the rollout to hit it. Both requested and actual decision counts are saved. Choose an
evaluation interval divisible by all compared collection sizes when possible. With four
envs, 256 versus 1024 steps gives 1024 versus 4096 samples per update: this also changes batch
size and policy refresh frequency. It does not isolate trajectory length alone.

## Outputs and metric meanings

Each run contains:

* `config.yaml`, `git.txt`, `metadata.json`: resolved settings, code version, software/device
  details, run status, and a session history (including resumes on another machine).
* `updates.csv`: one row per PPO update, with cumulative learner decisions, optimizer steps,
  collection/update time, cumulative training/evaluation/wall time, throughput, losses,
  approximate sampled KL, clip fraction, pre-clipping gradient norm, explained variance,
  and raw advantage/return/value statistics. CSV logging occurs every update regardless
  of the console logging interval.
* `episodes.csv`: one row per completed training episode, including its env/episode ID,
  actual completion decision, seat, opponents, win/loss/truncation, accumulated return,
  learner decisions, turns, and final VP. Accumulators span rollouts and checkpoints.
* `evaluation_games.csv`: one row per validation game, linked to its retained checkpoint,
  actual training decisions, board/game seed, seat, opponent, greedy/sampled mode and outcome.
* `latest.pt`: the resumable training checkpoint.
* `checkpoints/step_XXXXXXXXXXXX.pt`: retained checkpoints at validation points, including
  step zero. These contain full training state, so allow disk space for the sweep.

Entropy diagnostics are measured during collection, before the update. `head_*_entropy`
averages conditional entropy where that head has a valid choice (unavailable dummy masks
are excluded). `main_type_entropy` and `main_type_max_probability` use only MAIN decisions
with multiple legal types. End-turn statistics use MAIN decisions where a road, settlement,
or city is legal. Blank values mean no eligible observations, not zero entropy.

Training time is collection plus optimizer time; wall time also includes evaluation and
I/O during the training session. Throughput excludes evaluation. Pauses between sessions
are excluded. Evaluate both sample efficiency and elapsed compute cost. Value loss and
explained variance refer to lambda-dependent training targets, not observed true values.

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

Keep seed, engine, env, model, opponents, evaluation protocol, PPO parameters other than the
budget, and environment count unchanged. Device and runtime logging/checkpoint settings can
change. Checkpoints save episode accumulators, optimizer counters, CSV committed-row counts,
and cumulative timers. On resume, uncommitted CSV tails are removed and backed up as
`*.csv.recovery.bak`; missing committed rows cause an error. Copy CSVs with the checkpoint.
Future retained checkpoint files from a rolled-back attempt are not deleted; only records
in the reconciled CSVs belong to the resumed trajectory. Legacy checkpoints can resume,
but omit the first partial episode because they lack its earlier statistics.

Validation restores Python/NumPy/PyTorch RNG state and module modes, and does not update
weights or training environments. Changing evaluation frequency should not change training
actions. Greedy validation measures deployed greedy behavior; sampled validation uses a
separate fixed RNG and can be selected with `eval.deterministic=false`.

## Analysis and final tests

Analysis accepts individual run directories or a sweep root. It rejects duplicate records,
nonfinite update metrics and missing seat rotations; incomplete runs are skipped unless
`--include-incomplete` is specified. Use one attempt per configuration/training seed.

Outputs include:

* `evaluation_points.csv`: measured checkpoint results.
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
checkpoint, not whichever validation checkpoint happened to score highest. Do not select
the best configuration from optimizer loss alone.

After selecting configurations, evaluate retained checkpoints on separate final-test seeds:

```powershell
uv run --no-sync carla evaluate runs/<run>/checkpoints/step_000001048576.pt --opponent random --boards 250 --seed 20000 --csv runs/<run>/evaluation_test_games.csv
uv run --no-sync carla evaluate runs/<run>/checkpoints/step_000001048576.pt --opponent heuristic --boards 250 --seed 20000 --csv runs/<run>/evaluation_test_games.csv
uv run --no-sync python scripts/analyze_runs.py runs/lambda-study --output outputs/lambda-test --split test
```

250 boards give 1,000 games per opponent and trained seed. Test exports are separate from
the training journal and survive resume reconciliation. Analysis discovers
`evaluation_*_games.csv` beside the journals. A test export rejects the validation seed and
duplicate checkpoint/game records. Add `--sampled` to evaluate sampled rather than greedy
actions. The CLI's printed Wilson interval remains a game-level approximation; use the
saved board-grouped and training-seed results when making final comparisons.
