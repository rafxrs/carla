# Strategic reward controls, trade entropy and diagnostic evaluation

The reward remains `r + gamma * Phi(next) - Phi(current)`. Terminal potential is zero;
win/loss/truncation rewards are unchanged. Training still requires
`env.shaping_gamma == ppo.gamma`. No learned belief predictions or hidden opponent cards
enter the reward. The learner's own hand and own hidden VP cards remain available to it.

## Configuration parameters

All fields below live under `env`. Defaults reproduce the previous strategic reward.
Existing YAMLs/checkpoint configurations can omit every new field.

| Field | Default | Meaning |
|---|---:|---|
| `strategic_vp_weight` | 0.50 | Own total VP / 10, saturated at 1 |
| `strategic_production_weight` | 0.25 | Normalized own production |
| `strategic_readiness_weight` | 0.10 | Own purchase readiness |
| `strategic_award_weight` | 0.15 | Own award progress |
| `readiness_road_weight` | 0.15 | Road readiness |
| `readiness_settlement_weight` | 0.35 | Settlement readiness |
| `readiness_city_weight` | 0.35 | City readiness |
| `readiness_dev_weight` | 0.15 | Development-card readiness |
| `opponent_threat_coef` | 0.35 | Multiplier on weighted public opponent strength |
| `opponent_vp_weight` | 0.56 | Opponent public VP / 10, saturated at 1 |
| `opponent_production_weight` | 0.28 | Opponent production |
| `opponent_award_weight` | 0.16 | Opponent award progress |
| `award_road_weight` | 0.50 | Longest Road contribution to award progress |
| `award_army_weight` | 0.50 | Largest Army contribution to award progress |
| `production_normalization` | 2.0 | Divide raw expected cards/roll by this, then cap at 1 |
| `opponent_softmax_temperature` | 2.0 | Softmax divisor for **raw public VP**, with max subtraction |

The four utility weights, four readiness weights, three opponent strength weights, and two
award weights each form a separate group. Every group must be finite/nonnegative and sum
to 1 (relative/absolute tolerance `1e-6`). The threat multiplier must be finite/nonnegative;
production normalization and temperature must be finite/positive. Weights are validated,
not silently renormalized. The opponent defaults remain the supplied rounded weights
0.56/0.28/0.16. `env.shaping_coef` still scales the whole potential.

Existing two-argument reward helper calls retain their default behaviour. Helpers also
accept an optional `EnvConfig`; `expected_production(state, seat, robber=True)` returns five
raw expected per-resource yields without normalization or saturation.

```yaml
extends: full_trade_strategic.yaml
env:
  shaping_coef: 0.5
  strategic_vp_weight: 0.50
  strategic_production_weight: 0.30
  strategic_readiness_weight: 0.15
  strategic_award_weight: 0.05
  readiness_road_weight: 0.10
  readiness_settlement_weight: 0.40
  readiness_city_weight: 0.40
  readiness_dev_weight: 0.10
ppo:
  trade_entropy_coef: null
```

Prepared experiments:

- `configs/experiments/full_trade_strategic_economy_05.yaml`: 300,000 learner decisions,
  shaping coefficient 0.50.
- `configs/experiments/full_trade_strategic_economy_10.yaml`: identical except experiment
  name and shaping coefficient 1.00; it extends the first configuration.

Both inherit the full trade model: four GNN layers, hidden size 256, belief size 128, set
attention and GRU trade decoder. Learning rate is `1e-4`, both gammas are `0.999`, global
entropy coefficient is `0.01`. The fixed training composition is H25/H25/H50, expressed as
opponent pool weights 2:1. No starting checkpoint has been selected and no training is launched
by these files. The separate trade entropy coefficient is unset in both.

Inspect resolved YAML and dotted overrides without training:

```bash
uv run carla config configs/experiments/full_trade_strategic_economy_05.yaml
uv run carla config configs/experiments/full_trade_strategic_economy_10.yaml ppo.trade_entropy_coef=0.001 env.opponent_threat_coef=0.35
```

The same dotted overrides are accepted by `carla train` when a starting checkpoint is chosen.

## Separate entropy regularization

`ppo.trade_entropy_coef: float | null` defaults to `null` and must be finite/nonnegative
when set. `null` retains the old expression `entropy_coef * total_entropy` exactly.
Otherwise the bonus subtracted from the PPO loss is:

```text
entropy_coef * (total_entropy - trade_token_entropy)
+ trade_entropy_coef * trade_token_entropy
```

The policy objective, value loss, auxiliary loss, action sampling, saved offer tokens and
their joint log probabilities are unchanged. No model parameters or checkpoint tensor
shapes change. `distribution.trade_token_entropy()` is a differentiable per-row tensor:
conditional token entropy along the scored stored offer prefix on offer rows, and zero
elsewhere. It is a sampled-prefix regularizer, not exact entropy over every possible offer.
For a baseline distribution with no decoder it is zero. Counteroffers are offer actions too.

New PPO metrics (the existing raw `entropy` metric is unchanged):

| Metric | Meaning |
|---|---|
| `trade_token_entropy` | Token contribution averaged over all rows of a minibatch, then averaged over optimizer steps like the existing `entropy` metric |
| `non_trade_entropy` | Total entropy less that token contribution |
| `entropy_bonus` | Effective weighted bonus used in the loss |
| `trade_token_entropy_conditional` | Token entropy sum / number of proposal rows processed across optimizer steps; zero when there are none |
| `trade_proposal_fraction` | Proposal rows / all rows processed across optimizer steps |

The conditional metric is row-weighted; the first three follow the existing per-minibatch
logging convention. Unequal minibatch sizes can therefore give different averaging weights.
New training runs include these columns in `updates.csv`. Resuming a legacy journal keeps
its original columns and history; new entropy columns are omitted from that legacy file.
The learner still returns all metrics. Use a new run to retain the new CSV columns.
Older model and optimizer state dictionaries remain loadable; missing configuration fields
receive the old defaults. No previous experiment files are rewritten by this feature.

## Homogeneous and mixed evaluation

```bash
uv run carla evaluate checkpoint.pt --opponent heuristic50 --boards 50 --seed 30000 --batch-games 64 --device cuda
uv run carla evaluate checkpoint.pt --opponents random heuristic25 heuristic50 --boards 50 --seed 40000 --batch-games 64 --device cuda --csv outputs/mixed_games.csv
uv run carla evaluate checkpoint.pt --opponents heuristic25 heuristic25 heuristic50 --boards 10 --seed 30000 --batch-games 40 --device cuda --sampled
```

Use `--device cpu` without CUDA; `auto` still selects CUDA when available. Programmatic
`evaluate_policy` and `evaluate_checkpoint` accept either `opponent="heuristic50"` or
`opponents=["random", "heuristic25", "heuristic50"]`, not both. Omitting both still means
three heuristic bots. Any name in the agent registry is accepted (including `random_flat`).

Exactly three names are required. For each learner seat, the names are assigned in the given
order to the remaining **absolute seats in ascending order**, as in `CatanEnv`:

```text
opponents = [random, heuristic25, heuristic50]
learner seat 0 -> [learner, random, heuristic25, heuristic50]
learner seat 1 -> [random, learner, heuristic25, heuristic50]
learner seat 2 -> [random, heuristic25, learner, heuristic50]
learner seat 3 -> [random, heuristic25, heuristic50, learner]
```

There are exactly `boards * 4` games: no additional opponent permutations. Each rotation of
a board uses the same held-out game seed. Agents are distinct objects for every game/seat.
Batched inference, survivor-index selection of recurrent states, isolated RNG and module
mode restoration are preserved. Sampled evaluation is reproducible for fixed seed/device/
batch settings; changing batch size can change the random draw order.

Mixed identity is an ordered JSON list prefixed by `mixed:`, e.g.
`mixed:["random","heuristic25","heuristic50"]`. Three identical types retain the homogeneous
identity and produce the same game results. Explicit triplets add `opponent_composition`
and `opponent_seats` (JSON strings) to each standard per-game record. Homogeneous records
and CSV columns remain unchanged. Mixed CSVs append these two columns; appending mixed
records to a legacy homogeneous CSV is refused, so use a separate export path. Existing
export duplicate checks and validation/test seed separation still apply.

## Diagnostic evaluation

```bash
uv run carla evaluate checkpoint.pt --opponents heuristic25 heuristic25 heuristic50 --boards 10 --seed 30000 --batch-games 40 --device cuda --diagnostics --diagnostics-csv outputs/diagnostics.csv --trajectories
```

`--diagnostics-csv` and `--trajectories` imply `--diagnostics` on the CLI. In Python use
`evaluate_checkpoint(..., diagnostics=True, trajectories=True)` and inspect
`result.diagnostics.players`, `.summary`, `.trades`, `.offers`, and `.trajectories`.
Without diagnostics, histories are not replayed and no diagnostic legal sets, hand copies,
production trajectories or counters are computed. The ordinary summary and per-game records
remain available with diagnostics enabled.

After each game finishes, diagnostics replay `game.history` with recorded chance outcomes
through the same engine. They check the reconstructed state **and RNG state** against the
original. The replay's pre-action legal set and returned engine events drive all counters.
Unredacted hands are inspected solely after the game for affordability and impossible
opening requests; they never enter policy observations, action selection, or rewards.

Exports (all are new files; existing targets are refused):

| File | Rows |
|---|---|
| `diagnostics.csv` | Four per-player records for every evaluation game |
| `diagnostics_summary.csv` | Aggregated learner and opponent-type groups, with summed counts and recomputed rates |
| `diagnostics_trades.csv` | One row per participant per executed domestic trade, or one for a maritime trade |
| `diagnostics_offers.csv` | Opening offers and counteroffers with terms and quality flags |
| `diagnostics_trajectories.csv` | Optional per-player setup-end, completed-turn and terminal samples |

Per-game rows carry checkpoint path (null for direct policy evaluation), evaluation seed,
board, game seed, learner/player seats, role/type, opponent composition/seat map, mode,
outcome/winner, game turns and engine steps. Count columns include `action_<type>` and
`dev_played_<card>`. Vector/list columns are JSON inside CSV cells. Undefined ratios and
unreached milestones are Python `None` and empty CSV cells. Empty trade/offer exports still
have metadata/term columns. CLI stdout includes the usual summary and aggregate diagnostics.

Attribution and denominator conventions:

- Negotiation/opening/direct-acceptance/completion/failure counts belong to its initiator.
  Completed trades additionally count once for the initiator and once for the responding
  partner. Invitations, counteroffers and required protocol responses are separate from
  voluntary main-phase initiation. All-rejected requires three explicit rejections of an
  opening offer; an invitation or counteroffer is not a direct acceptance.
- Failed-offer repetition includes opening terms and counteroffers not selected for execution.
  A counteroffer completion can succeed as a negotiation while the original opening terms
  were not executed. Repeat keys are exact canonical count vectors. Reverse exchanges compare
  resource directions within one engine turn, allowing different quantities; exact executed
  terms remain available in each participant's row.
- Trade unlocking/removing a purchase concerns **resource costs only**. It does not assert
  that pieces, deck cards, or legal board locations are available. Main-phase opportunity
  counts use the actual complete legal action set instead. Construction means road,
  settlement or city; development purchases have separate opportunity/choice counters.
- Trade rows retain subsequent constructions and development purchases by that participant
  during the same engine player turn. These are temporal associations, not causal claims.
- Setup settlements/roads are excluded from normal construction. Post-setup free roads
  from Road Building are separated from paid roads. Dice resource events are separated from
  setup grants, stealing and development effects. Maritime/domestic exchanges are separate.
- Production is raw expected cards per roll by resource, with and without the robber. It
  ignores bank scarcity and has no reward normalization/saturation. Means sample after each
  completed engine player turn; raw sample counts and optional trajectories are retained.
- `completed_turns` is per player, excluding an unfinished winning turn. Rates divide by that
  player's completed turns. `turns_at_two_vp_or_less` samples total VP at that player's turn
  end. A zero denominator remains missing. Milestone times are zero-based **global completed
  player-turn counts**; `0` means reached during the first turn, `None` means never reached.

No engine rules, model architecture, action encoding, or live rollout collection change.

## File inventory

Modified:

- `src/carla/config/schema.py`, `src/carla/envs/rewards.py`: validated reward coefficients and compatible helper defaults.
- `src/carla/models/distributions.py`, `src/carla/rl/ppo.py`, `src/carla/rl/metrics.py`: public entropy components, optional weighting and raw metrics.
- `src/carla/rl/train.py`, `src/carla/utils/logging.py`: preserve legacy CSV headers on resume.
- `src/carla/rl/evaluate.py`, `src/carla/cli.py`: mixed composition, optional diagnostics, CLI and compatible exports.
- `tests/envs/test_strategic_rewards.py`: configurable formulas, validation, gamma identity and experiment configuration coverage.
- `README.md`: link to this guide.

Added:

- `src/carla/eval/diagnostics.py`: verified engine replay, per-player counters, summaries and exports.
- `tests/eval/test_diagnostics.py`: actual engine trade/build/production scenarios and replay verification.
- `tests/rl/test_trade_entropy.py`: objective equivalence, component gradients, empty-trade and baseline cases.
- `tests/rl/test_mixed_evaluation.py`: compositions, recurrent batching, diagnostics invariance, legacy checkpoints/CSVs and CLI.
- `configs/experiments/full_trade_strategic_economy_05.yaml`, `configs/experiments/full_trade_strategic_economy_10.yaml`: the prepared experiment pair.
- `docs/strategic_diagnostics.md`: this guide.
