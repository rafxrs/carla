"""Experiment configuration schema. Every run is fully described by one `ExperimentConfig`.

Ablations are single fields: `model.encoder`, `model.belief`, `model.set_attention`,
`model.trade_decoder`, `engine.allow_domestic_trade`, `opponents.pool`, `ppo.*`.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

from carla.engine.config import EngineConfig


@dataclass(frozen=True)
class EnvConfig:
    learner_seat: int = -1
    """Seat of the learning agent in `CatanEnv` (-1: random each episode)."""
    win_reward: float = 1.0
    loss_reward: float = -1.0
    truncation_reward: float = 0.0
    vp_reward: float = 0.0
    """Shaping: reward per public victory point gained (0 disables)."""
    shaping: Literal["none", "own", "margin", "strategic"] = "none"
    """Potential-based shaping in `CatanEnv`: F = gamma * Phi(s') - Phi(s) between consecutive
    learner decisions, with Phi = 0 once the game is over, so the optimal policy is unchanged.
    own: Phi = shaping_coef * own VP (own hidden VP cards included); margin: Phi = shaping_coef *
    (own VP - the top opponent's public VP). strategic: Phi = shaping_coef * (learner utility -
    0.35 * public opponent threat), combining VP, production, build readiness (own hand only),
    and award progress. Opponent threat uses public VP, production, and award progress with
    public-VP softmax weights; no hidden opponent cards or learned beliefs are used."""
    shaping_coef: float = 0.1
    shaping_gamma: float = 0.999
    """Discount used by the shaping; must equal `ppo.gamma` to keep the optimal policy."""
    max_events: int = 64
    """Event-stream slots per observation (belief module input); older events are dropped."""
    strategic_vp_weight: float = 0.50
    strategic_production_weight: float = 0.25
    strategic_readiness_weight: float = 0.10
    strategic_award_weight: float = 0.15
    readiness_road_weight: float = 0.15
    readiness_settlement_weight: float = 0.35
    readiness_city_weight: float = 0.35
    readiness_dev_weight: float = 0.15
    opponent_threat_coef: float = 0.35
    opponent_vp_weight: float = 0.56
    opponent_production_weight: float = 0.28
    opponent_award_weight: float = 0.16
    award_road_weight: float = 0.50
    award_army_weight: float = 0.50
    production_normalization: float = 2.0
    opponent_softmax_temperature: float = 2.0

    def __post_init__(self) -> None:
        for names in (
            (
                "strategic_vp_weight",
                "strategic_production_weight",
                "strategic_readiness_weight",
                "strategic_award_weight",
            ),
            (
                "readiness_road_weight",
                "readiness_settlement_weight",
                "readiness_city_weight",
                "readiness_dev_weight",
            ),
            ("opponent_vp_weight", "opponent_production_weight", "opponent_award_weight"),
            ("award_road_weight", "award_army_weight"),
        ):
            values = [getattr(self, name) for name in names]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{names}: weights must be finite and nonnegative")
            if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"{names}: weights must sum to one")
        if not math.isfinite(self.opponent_threat_coef) or self.opponent_threat_coef < 0:
            raise ValueError("opponent_threat_coef must be finite and nonnegative")
        for name in ("production_normalization", "opponent_softmax_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ModelConfig:
    encoder: Literal["mlp", "gnn"] = "mlp"
    hidden_dim: int = 256
    mlp_layers: int = 3
    gnn_layers: int = 4
    gnn_tile_nodes: bool = True
    """Include tiles as graph nodes (else tile features are folded into their corners)."""
    set_attention: bool = False
    """Attention pooling over dev cards / opponents / trade sets instead of flat features."""
    attention_heads: int = 4
    belief: Literal["none", "gru", "lstm"] = "none"
    belief_dim: int = 128
    belief_loss_coef: float = 0.1
    trade_decoder: Literal["none", "gru", "transformer"] = "none"
    max_trade_cards_per_side: int = 4
    """Decoder cap; the engine's offer limit can further restrict it."""


@dataclass(frozen=True)
class PPOConfig:
    total_env_steps: int = 50_000_000
    rollout_steps: int = 256
    """Decisions collected per environment before sending a batch to the learner."""
    epochs: int = 4
    minibatch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    max_policy_lag: int = 1
    """Semi-asynchronous PPO: accept rollouts from policies at most this many updates old."""
    sequence_length: int = 32
    """Truncated BPTT window for recurrent (belief) models."""
    trade_entropy_coef: float | None = None
    """Optional coefficient for trade-token entropy; None preserves the original objective."""

    def __post_init__(self) -> None:
        if self.trade_entropy_coef is not None and (
            not math.isfinite(self.trade_entropy_coef) or self.trade_entropy_coef < 0
        ):
            raise ValueError("trade_entropy_coef must be finite and nonnegative or None")


@dataclass(frozen=True)
class OpponentSpec:
    kind: Literal[
        "self",
        "snapshot",
        "random",
        "heuristic05",
        "heuristic10",
        "heuristic15",
        "heuristic25",
        "heuristic50",
        "heuristic75",
        "heuristic",
        "checkpoint",
    ]
    weight: float = 1.0
    path: str | None = None
    """Checkpoint file for kind == "checkpoint"."""


@dataclass(frozen=True)
class OpponentPoolConfig:
    pool: tuple[OpponentSpec, ...] = (
        OpponentSpec("self", 0.6),
        OpponentSpec("snapshot", 0.3),
        OpponentSpec("heuristic", 0.1),
    )
    """Each opponent seat samples a spec with probability proportional to its weight."""
    snapshot_every: int = 50
    """Policy updates between snapshots added to the pool."""
    max_snapshots: int = 20


@dataclass(frozen=True)
class EvalConfig:
    agents: tuple[str, ...] = ("heuristic", "random", "random", "random")
    """Agent registry names (see `carla.agents.registry`), one per seat before rotation."""
    num_games: int = 200
    seeds: tuple[int, ...] = (0, 1, 2)
    rotate_seats: bool = True
    confidence: float = 0.95
    num_workers: int = 1
    save_records: bool = False
    every_env_steps: int = 0
    """Periodic training evaluation interval (learner decisions); 0 disables it."""
    boards: int = 10
    opponents: tuple[str, ...] = ("random", "heuristic")
    seed: int = 10_000
    deterministic: bool = True
    """Greedy actions when True, policy sampling otherwise. Evaluation has its own RNG."""
    retain_checkpoints: bool = True
    """Save a full checkpoint at every validation point; latest.pt is always resumable."""


@dataclass(frozen=True)
class RuntimeConfig:
    num_workers: int = 8
    """Rollout worker processes (CPU)."""
    envs_per_worker: int = 4
    device: str = "auto"
    """Learner device: "auto", "cpu" or "cuda[:i]"."""
    torch_threads_per_worker: int = 1
    deterministic_torch: bool = False
    output_dir: str = "runs"
    checkpoint_every: int = 100
    log_every: int = 10
    collect_training_diagnostics: bool = False
    """Incremental learner-only episode counters; no replay, extra inference or legal queries."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "default"
    seed: int = 0
    engine: EngineConfig = field(default_factory=EngineConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    opponents: OpponentPoolConfig = field(default_factory=OpponentPoolConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
