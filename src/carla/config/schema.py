"""Experiment configuration schema. Every run is fully described by one `ExperimentConfig`.

Ablations are single fields: `model.encoder`, `model.belief`, `model.set_attention`,
`model.trade_decoder`, `engine.allow_domestic_trade`, `opponents.pool`, `ppo.*`.
"""

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
    max_events: int = 64
    """Event-stream slots per observation (belief module input); older events are dropped."""


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


@dataclass(frozen=True)
class OpponentSpec:
    kind: Literal["self", "snapshot", "random", "heuristic", "checkpoint"]
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
