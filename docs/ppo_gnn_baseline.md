# Synchronous GNN PPO baseline

`model.encoder: gnn` selects a native PyTorch policy/value network through the existing
factory. It uses the same observation arrays, action masks, `PolicyOutput`, factored
distribution, rollouts, GAE, PPO update, optimizer, checkpoints, and evaluator as the MLP.
No graph library or observation/engine/action changes are required.

## Architecture and shapes

Let `B` be the batch size and `H = model.hidden_dim` (256 in the experiment config;
64 in the smoke config). Separate shared Linear/ReLU projections encode:

| Stage | Shape |
| --- | --- |
| Raw corners | `(B, 54, 14)` |
| Raw roads | `(B, 72, 4)` |
| Raw tiles | `(B, 19, 8)` |
| Projected corners / roads / tiles | `(B, 54, H)` / `(B, 72, H)` / `(B, 19, H)` |
| Directed road messages | `(B, 144, H)` |
| Tile/corner incidences | `(B, 114, H)` |
| Embeddings after each round | `(B, 54, H)` / `(B, 72, H)` / `(B, 19, H)` |
| Three mean-pooled graph vectors | `(B, 3H)` |
| Non-board features | `(B, 171)` = players 64 + hand 5 + dev cards 10 + globals 36 + trade context 56 |
| Context MLP input | `(B, 3H + 171)`; `(B, 939)` at H=256, `(B, 363)` at H=64 |
| Shared context | `(B, H)` |
| Critic output | `(B,)` |

`gnn_layers` rounds operate on the engine's canonical topology obtained from `board_graph()`.
Index arrays and normalization degrees are registered model buffers, constructed once and
shared by all rounds. They move with the model and are included in checkpoints.

Each corner receives mean-aggregated messages computed from its neighboring corners and
connecting road embeddings. It also receives transformed tile embeddings from its adjacent
tiles. An MLP updates the corner using its old embedding and both aggregates, with a residual
connection and LayerNorm. Road updates combine the old road embedding with the **sum** and
**absolute difference** of the newly updated endpoints, followed by an MLP, residual, and
LayerNorm. Endpoint ordering therefore has no effect. Each tile receives the mean of its six
updated corner embeddings and updates using an MLP, residual, and LayerNorm. Forward passes
use batched gathers and `index_add_`; there are no per-entity Python loops.

`gnn_tile_nodes: false` is supported: tile projections are mean-folded into adjacent corners
before the first round. Tiles skip the per-round updates, then receive a final readout from
their original projection and their six encoded corners. This mode retains per-tile
embeddings for pooling and robber scoring while reducing repeated tile/corner communication.

Corner, road, and tile means are concatenated with flattened `players`, `hand`, `dev_cards`,
`globals`, and `trade_context`, matching all non-board inputs of the MLP. A two-layer ReLU
context MLP produces `(B, H)` for the actor and critic. Board arrays are never flattened into
this context.

## Policy contract

Settlement and city use separate shared pointer scorers on corner embeddings, road uses a
scorer on road embeddings, and robber uses a scorer on tile embeddings. Each computes
`Linear(tanh(Linear(entity) + Linear(context)))`. The nonlinear interaction lets context
change the ranking of positions. The robber scorer has four output channels, one for each
relative victim (including the existing no-victim slot), preserving a joint tile/victim
categorical rather than imposing independence between tile and victim.

All other heads are linear projections from context. Every output is a raw logit; legality
and normalization stay in the existing masked distribution.

| Head | Shape |
| --- | --- |
| `type` | `(B, 16)` |
| `settlement`, `city` | `(B, 54)` each |
| `road` | `(B, 72)` |
| `robber` | `(B, 19, 4)` |
| `dev`, `monopoly`, `discard` | `(B, 5)` each |
| `year_of_plenty`, `maritime`, `trade` | `(B, 5, 5)` each |
| `partner` | `(B, 4)` |

The feed-forward network returns `state=None`, `belief=None`, and one value per sample.

## Integration and restrictions

The trainer and evaluator now accept feed-forward `mlp` and `gnn` encoders. The factory
already selected `GNNPolicy`, so it required no dispatch change. PPO math, rollout collection,
GAE, action masking, checkpoint serialization, and the MLP implementation are unchanged.

The existing trainer supports **one worker with one or more synchronous environments**;
that behavior is retained, and the GNN smoke config uses one environment. Each game has
three fixed bots of a single configured kind (`random` or `heuristic`). Belief networks,
set attention, variable-length trade decoders, self-play pools, and parallel rollout workers
remain unsupported. Domestic trading remains disabled in the GNN configs.

`configs/experiments/gnn.yaml` inherits the general base configuration; its inherited worker
and opponent-pool settings still require overrides for synchronous training. Use
`configs/gnn_smoke.yaml` for a fully runnable, 128-decision example:

```bash
uv run --no-sync carla train configs/gnn_smoke.yaml
```

For small CPU batches, setting `OMP_NUM_THREADS=1` before launching can avoid thread overhead.
The smoke run performs two updates and writes `config.yaml`, `git.txt`, and `latest.pt` under
`runs/gnn_smoke/<run>/`. The resolved model settings are saved with the checkpoint.

```bash
uv run --no-sync carla train configs/gnn_smoke.yaml --resume runs/gnn_smoke/<run>/latest.pt ppo.total_env_steps=256
uv run --no-sync carla evaluate runs/gnn_smoke/<run>/latest.pt --opponent random --boards 1 --seed 10000
```

## Symmetry and information limits

Shared projections, symmetric endpoint updates, mean aggregation, and shared pointer scorers
make board embeddings/spatial logits equivariant under consistent permutations of board
features **and topology**. Pooling makes context, value, and non-spatial logits invariant.
Arbitrary array shuffles with the fixed canonical topology are not a symmetry; actual board
rotation/reflection automorphisms are. Masks must be transformed with spatial actions when
comparing complete distributions. Ties in greedy `argmax` can still select by array order.

This does not make the network equivariant to arbitrary permutations of player seats or
resource identities: those feature channels have fixed meanings, and player/trade features
retain their relative-seat order. Mean aggregation and pooling are lossy; finite message
depth does not exactly compute global properties such as longest road. The graph may assign
equal embeddings to locally indistinguishable positions.

Existing tile observations encode production **probability**, so equal-probability numbers
(e.g. 6 and 8) are indistinguishable. This is often appropriate for production expectations,
but exact number identity and correlations between tiles triggered by the same roll are
lost. Event history is ignored by this feed-forward baseline, just as in the MLP; hidden
opponent resources are intentionally absent. No belief module, positional IDs, coordinates,
or extra observations are introduced to compensate.

## Validation

`tests/models/test_gnn.py` covers construction, all head shapes, single/batched observations,
both tile modes, legal masked sampling, finite log probabilities/entropy, backward gradients,
actual neighbor/road/tile information flow, endpoint-order symmetry, and consistent graph
relabeling. CUDA forward/device coverage runs when available. Existing rollout/checkpoint,
PPO-update, and CLI train/resume/evaluation tests exercise both encoders; CLI coverage checks
that resumed weights match an uninterrupted run exactly. Trainer validation tests preserve
unsupported-setting checks, and all shipped YAML files are loaded by config tests.

Run the normal project checks:

```bash
uv run --no-sync ruff check src tests scripts
uv run --no-sync black --check src tests scripts
uv run --no-sync pytest
```

Validation for this implementation used the repository's existing WSL CPU environment
(PyTorch 2.14.1), with one PyTorch thread for the small test and smoke batches:

- Full `pytest -n auto -q`: **643 passed, 5 skipped** in 113.64 seconds. Four CUDA cases
  were skipped because CUDA was unavailable; the GUI smoke module was skipped without Arcade.
- `ruff check src tests scripts` and `black --check src tests scripts`: passed.
- GNN smoke: **128 learner decisions, 2 PPO updates**, finite parameters, saved optimizer
  state and checkpoint counters verified. Checkpoint:
  `runs/gnn_smoke/gnn_smoke-20261109-005110-b5d9b2da/latest.pt`.
- Evaluation of that checkpoint against random bots, one board in all four seats: completed
  with 0/4 wins and mean VP 2.00. This checks integration, not trained playing strength.
