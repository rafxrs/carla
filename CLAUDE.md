# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CARLA is a four-player Settlers of Catan RL project (ECSE 552): an in-house deterministic rules engine, Gymnasium/AEC envs with factored masked action heads, and synchronous PPO over MLP/GNN policies. README.md is the authoritative overview (architecture diagrams, engine API, rule choices); focused protocols live in `docs/` (`domestic_trading.md`, `trade_decoder.md`, `experiments.md`, `ppo_*_baseline.md`).

## Commands

Environment is managed with uv (`uv sync --extra cpu --extra dev --extra gui`; `cpu` and `cu130` extras are mutually exclusive). Activate `.venv` or prefix with `uv run`.

```bash
pytest -n auto                                  # full suite (what CI runs)
pytest -n auto -m "not slow"                    # skip the 200-game random sweep
pytest tests/engine/test_dev_cards.py::test_name -q   # single test
ruff check src tests scripts && black --check src tests scripts   # CI lint (line length 100)
carla config configs/experiments/gnn.yaml model.belief=gru        # print resolved config
carla train configs/smoke.yaml                  # quick end-to-end PPO check (gnn_smoke.yaml for GNN)
```

Other CLI subcommands (`simulate`, `replay`, `tournament`, `evaluate`, `policy-tournament`, `gui`) are listed in README.md. Entry point is `src/carla/cli.py`.

Test notes:
- `tests/models/` is skipped entirely when torch isn't installed (`conftest.py`); GUI tests `importorskip("arcade")`. The real-window smoke test needs `CARLA_REQUIRE_GUI=1` and a display (CI uses `xvfb-run`).
- pytest turns `DeprecationWarning` from `carla.*` into errors, and `--strict-markers` is on (only `slow` is registered).
- `tests/` mirrors `src/carla/`; `tests/engine/scenario.py` builds hand-made positions (`post_setup_game`, `give`) and checks invariants. Every rule change needs a test.

## Architecture

Layered packages under `src/carla/`; imports only go "down":
`engine` (stdlib only, imports no other carla package) ← `config`, `agents`, `envs` ← `models` ← `rl`; `eval` and `gui` sit on `agents`. **torch may only be imported in `models` and `rl`** (and lazily in `agents/policy_agent.py`); keep `engine`, `envs`, bots and eval torch-free.

Key cross-file contracts:
- **Engine**: `Game` (`engine/game.py`) with `legal_actions()`, `step()`, `snapshot()/restore()`; `GameRecord` stores every decision plus chance outcomes and per-step state hashes for exact replay. Actions are pre-built `Action` named-tuple singletons indexed by `ACTION_INDEX`; their fields are the policy heads. Board and dice use separate RNG streams from the seed.
- **Seats**: engine uses absolute seats 0–3; envs/models use seats relative to the actor (0 = actor). Hidden info is redacted via `Event.visible_to(seat)`, and a test checks the observation encoder leaks no hidden cards.
- **Observations** are defined in `envs/observation.py`; **action heads and the mask table** in `envs/action_space.py`; **policy output** (`PolicyOutput`) in `models/base.py`. Changing one of these usually requires touching env, model and rollout code together. Multi-card trade offers use canonical tokens (`envs/trade_tokens.py`) and the GRU decoder (`models/trade_decoder.py`); `TradeActionWrapper` adapts shapes for Gymnasium clients.
- **Envs**: `CatanAECEnv` is the multi-seat self-play env (workers choose the policy per seat); `CatanEnv` is the single-learner Gymnasium wrapper with `ExternalAgent` seats. Rewards and potential-based shaping are in `envs/rewards.py`.
- **GNN**: message passing is hand-rolled (no PyG/DGL) — the board graph is fixed, so batches share static index tensors from `models/graph.py`. `models/factory.py::build_policy` picks the network from `ModelConfig`; belief/set attention require `encoder=gnn` (validated in `rl/train.py`).
- **Training** (`rl/train.py`): CPU rollout worker processes (`rl/rollout.py`, `process_rollouts.py`) batch actions across envs, then synchronous PPO updates (`rl/ppo.py`, `gae.py`, `buffer.py`). Opponents come from `rl/opponents.py` (bots, fixed checkpoints, or a `League` of current policy + snapshots). Optional BC warm start (`rl/imitation.py`) and KL pull to a reference policy. Checkpoints are resumable; `train.py` upgrades observations for older pre-decoder checkpoints, so keep checkpoint backward compatibility in mind when changing observation fields.
- **Agents** are built by name via `agents/registry.py::make_agent` (`random`, `random_flat`, `heuristic`, the styled teachers `heuristic_city|expansion|devcard`, `ppo:<checkpoint>[@variant]` greedy or `ppo-sampled:` sampled); configs, tournaments and the GUI all use these names.
- **AlphaCat** (`docs/alphacat.md`): `encoder: strategy` (`models/strategy_net.py`) needs `env.strategic_features`; its recurrent state packs [belief | LSTM h,c]. The strategy statistic z, controllers and pseudo-rewards live in `envs/strategy.py`; every place that runs a strategy policy goes through `rl/acting.py::decide` (selection, re-run under the new z, joint log-prob). The strategy pseudo-reward is a separate stream (`RolloutBatch.strategy_rewards`, own critic), not shaping. `ppo.bptt: sequence` trains chunks via `PolicyNetwork.unroll`. The league is `rl/league/` (pure-Python `core.py`, worker processes keep envs per learner, `trainer.py` coordinates); `carla league`, `league-report`, `robustness`, `profile`, `strategy-bc` are its CLI. `configs/alphacat_smoke.yaml` is the fast end-to-end check.

## Configs and reproducibility

- Configs are frozen dataclasses in `config/schema.py`; YAML files use `extends:` and are strictly validated (unknown keys fail). New options need a schema field with a default. CLI dotted overrides (`ppo.total_env_steps=200000`) apply on top.
- `configs/base.yaml` holds shared settings; each file in `configs/experiments/` is one ablation that changes as few fields as possible.
- Derive every seed with `utils/seeding.py::derive_seed(config.seed, ...)`. `carla train` writes the resolved config and git commit into its run directory under `runs/` (`runs/` and `outputs/` are gitignored).

## Conventions

- ruff + black via pre-commit (local hooks using the venv's tools).
- Commit messages: short imperative sentence (e.g. "Add league opponents: current policy, snapshots and checkpoints").
- README.md covers overall architecture; put protocol-level docs in `docs/` and update README's status table when a module's capabilities change.
