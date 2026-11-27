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
    shaping: Literal["none", "own", "margin"] = "none"
    """Potential-based shaping in `CatanEnv`: F = gamma * Phi(s') - Phi(s) between consecutive
    learner decisions, with Phi = 0 once the game is over, so the optimal policy is unchanged.
    own: Phi = shaping_coef * own VP (own hidden VP cards included); margin: Phi = shaping_coef *
    (own VP - the top opponent's public VP)."""
    shaping_coef: float = 0.1
    shaping_gamma: float = 0.999
    """Discount used by the shaping; must equal `ppo.gamma` to keep the optimal policy."""
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
    trade_entropy_coef: float | None = None
    """Weight of the trade decoder's offer-token entropy (None: `entropy_coef`). The token term
    sums several tokens (about 5 nats against 0.2-0.8 for other heads), so at `entropy_coef` it
    dominates the bonus and keeps offers random; a smaller weight lets offers sharpen."""
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    max_policy_lag: int = 1
    """Semi-asynchronous PPO: accept rollouts from policies at most this many updates old."""
    sequence_length: int = 32
    """Truncated BPTT window for recurrent (belief) models."""


@dataclass(frozen=True)
class OpponentSpec:
    kind: Literal["self", "snapshot", "random", "heuristic", "checkpoint", "agent"]
    weight: float = 1.0
    path: str | None = None
    """Checkpoint file for kind == "checkpoint"."""
    name: str | None = None
    """Agent registry name for kind == "agent", e.g. "native-f", "jsettlers-smart-native",
    "catanatron-ab2" or "ppo:<checkpoint>". Process-backed agents (the Java-wrapped JSettlers
    bots, external agents) cannot be training opponents: worker states are pickled."""


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
