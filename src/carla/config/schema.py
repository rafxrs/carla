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
    shaping: Literal["none", "own", "margin", "production"] = "none"
    """Potential-based shaping in `CatanEnv`: F = gamma * Phi(s') - Phi(s) between consecutive
    learner decisions, with Phi = 0 once the game is over, so the optimal policy is unchanged.
    own: Phi = shaping_coef * own VP (own hidden VP cards included); margin: Phi = shaping_coef *
    (own VP - the top opponent's public VP); production: Phi = shaping_coef * (own VP +
    production_weight * own expected resources per round), crediting settlements and cities for
    their future income as soon as they are built."""
    shaping_coef: float = 0.1
    production_weight: float = 0.5
    """VP-equivalent of one expected resource per round (four rolls) for `shaping=production`.
    An average settlement yields about 1.1 resources per round, so it is worth about 0.55 VP."""
    shaping_gamma: float = 0.999
    """Discount used by the shaping; must equal `ppo.gamma` to keep the optimal policy."""
    max_events: int = 64
    """Event-stream slots per observation (belief module input); older events are dropped."""
    strategic_features: bool = False
    """Add the planning features of `carla.envs.strategy_features` (production per seat and
    resource, build affordability and deficits, reachable sites, public build statistics, the
    observer's previous action type). Public information and the observer's own cards only."""


@dataclass(frozen=True)
class ModelConfig:
    encoder: Literal["mlp", "gnn", "strategy"] = "mlp"
    """"strategy": the AlphaStar-style network of `carla.models.strategy_net` (board GNN, seat
    transformer, recurrent core, strategy-conditioned pointer heads)."""
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
    core: Literal["none", "lstm"] = "none"
    """Recurrent core over the encoded observation (encoder=strategy)."""
    core_dim: int = 256
    core_layers: int = 1
    strategy_conditioning: bool = False
    """Condition heads, trade decoder and values on the strategy statistic z (encoder=strategy)."""
    strategy_dim: int = 64
    """Width of the z embedding."""
    strategy_selector: bool = False
    """Add a head that chooses a strategy archetype (or none) from the core output."""
    num_strategies: int = 3
    """Archetypes the selector chooses among (plus one "no strategy" option)."""
    privileged_value: bool = False
    """Feed the opponents' hidden cards to the value heads only (training-time critic). The
    policy logits never read them."""
    seat_attention_layers: int = 2
    """Self-attention layers over the four seats (encoder=strategy)."""


@dataclass(frozen=True)
class PPOConfig:
    total_env_steps: int = 50_000_000
    rollout_steps: int = 256
    """Decisions collected per environment before sending a batch to the learner."""
    epochs: int = 4
    minibatch_size: int = 4096
    micro_batch_size: int | None = None
    """Split each minibatch into chunks of this many decisions and accumulate their gradients
    (None: one pass). The update is unchanged; peak GPU memory shrinks with the chunk."""
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
    reference_checkpoint: str | None = None
    """Checkpoint (e.g. a behaviour clone) whose action-type choices regularize the policy."""
    reference_kl_coef: float = 0.0
    """Weight of KL(reference || policy) over the action type. The forward direction keeps
    probability on every type the reference plays (such as ending the turn to save up), which
    an entropy bonus cannot do for types the policy has driven to near-zero probability."""
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    max_policy_lag: int = 1
    """Semi-asynchronous PPO: accept rollouts from policies at most this many updates old."""
    sequence_length: int = 32
    """Truncated BPTT window for recurrent models when `bptt` is "sequence"."""
    bptt: Literal["step", "sequence"] = "step"
    """step: recompute each decision from its stored state (truncated BPTT of one decision).
    sequence: unroll chunks of `sequence_length` decisions from the stored state at the chunk
    start, resetting the state at episode boundaries (needed for a recurrent core)."""
    strategy_value_coef: float = 0.5
    """Weight of the strategy value head's loss (strategy pseudo-reward critic)."""
    upgo_coef: float = 0.0
    """Weight of the upgoing policy update (UPGO, AlphaStar) on the outcome stream."""
    reference_kl_heads: Literal["type", "all"] = "type"
    """KL toward the reference over the action type only, or over every head the taken action
    uses (AlphaStar's distillation over all arguments)."""


@dataclass(frozen=True)
class OpponentSpec:
    kind: Literal[
        "self",
        "snapshot",
        "random",
        "heuristic",
        "heuristic_city",
        "heuristic_expansion",
        "heuristic_devcard",
        "checkpoint",
    ]
    weight: float = 1.0
    path: str | None = None
    """Checkpoint file for kind == "checkpoint"."""
    name: str | None = None
    """Label of a "checkpoint" opponent in the episode journal (default: the file stem)."""


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
class StrategyConfig:
    """Strategy statistic z (see `carla.envs.strategy`): conditioning, pseudo-rewards, selection.

    z holds an opening build order and cumulative build counts. A z is either prescribed
    (sampled from a library of teacher games), chosen by the policy's selector head, or absent.
    The strategy pseudo-reward is a separate objective with its own critic; it is not
    potential-based shaping and deliberately changes what the policy optimizes.
    """

    enabled: bool = False
    library: str | None = None
    """z library (.npz from `carla strategy-bc`); None: the one stored in the init checkpoint."""
    archetypes: tuple[str, ...] = ("city", "expansion", "devcard")
    selection: Literal["prescribed", "learned", "mixed", "none"] = "prescribed"
    """How the learner's z is set each episode. mixed: prescribed with `prescribed_prob`, else
    learned by the selector head."""
    prescribed_prob: float = 0.5
    none_prob: float = 0.1
    """Probability that an episode plays without z (all zeros), so the policy also plays
    without an external strategy."""
    reconsider_turns: int = 8
    """A learned selection is reconsidered at the seat's roll every this many own turns."""
    switch_cost: float = 0.05
    """Strategy pseudo-reward charged when the selector switches to another archetype."""
    pseudo_reward_prob: float = 0.5
    """Probability that an episode with z pays the strategy pseudo-reward (AlphaStar: 25%)."""
    pseudo_reward_coef: float = 0.5
    """Weight of the strategy advantage in the policy gradient (outcome advantage weight 1)."""
    build_order_weight: float = 1.0
    statistics_weight: float = 1.0
    decay_turns: tuple[int, ...] = (40, 80, 120)
    """The pseudo-reward halves at each listed game turn and stops after the last."""
    selector_entropy_coef: float = 0.01
    sl_zero_prob: float = 0.1
    """Behaviour cloning: probability that a demonstration game is trained without its z."""


@dataclass(frozen=True)
class LeagueConfig:
    """AlphaStar-style league (`carla league`); see `carla.rl.league`."""

    main_agents: int = 1
    main_exploiters: int = 1
    league_exploiters: int = 1
    schedule: tuple[float, ...] = (0.5, 0.25, 0.25)
    """Share of updates for main agents, main exploiters and league exploiters."""
    updates_per_turn: int = 10
    """Consecutive PPO updates of one learning player before the next one trains."""
    teachers: tuple[str, ...] = (
        "heuristic",
        "heuristic_city",
        "heuristic_expansion",
        "heuristic_devcard",
    )
    """Bots that are permanent (protected) league members."""
    seed_checkpoints: tuple[OpponentSpec, ...] = ()
    """Fixed policy checkpoints added as protected historical players."""
    init_checkpoint: str | None = None
    """Initial (supervised) weights of every learning player and exploiter resets."""
    variants: tuple[str, ...] = ("select", "none", "city", "expansion", "devcard")
    """Strategy variants a strategy-conditioned policy opponent may play (payoff per variant)."""
    main_branches: tuple[float, ...] = (0.35, 0.5, 0.15)
    """Main agent matchmaking: self-play, PFSP over the league, verification."""
    pfsp_main: Literal["squared", "linear", "variance", "linear_capped"] = "squared"
    pfsp_exploiter: Literal["squared", "linear", "variance", "linear_capped"] = "variance"
    pfsp_league_exploiter: Literal["squared", "linear", "variance", "linear_capped"] = (
        "linear_capped"
    )
    payoff_decay: float = 0.99
    selfplay_floor: float = 0.3
    """Self-play falls back to the opponent's history when the pairwise rate is below this."""
    exploiter_target: float = 0.2
    """A main exploiter plays the current main agents while its pairwise rate exceeds this."""
    beat_threshold: float = 0.7
    forgotten_threshold: float = 0.7
    exploit_threshold: float = 0.3
    main_snapshot_steps: int = 1_000_000
    main_snapshot_max_steps: int = 2_000_000
    exploiter_min_steps: int = 1_000_000
    exploiter_max_steps: int = 2_000_000
    exploiter_min_win_rate: float = 0.3
    """A main exploiter is frozen early only if it also wins this share of its own games."""
    league_exploiter_reset_prob: float = 0.25
    exploiter_objective: Literal["win", "win_and_rank", "main_loses"] = "win"
    """win: own outcome only. win_and_rank: plus `rank_bonus` per main seat finished ahead of.
    main_loses: +1 whenever no main agent wins (kingmaking allowed; for comparison only)."""
    rank_bonus: float = 0.25
    exploiter_strategy: Literal["select", "random_archetype"] = "random_archetype"
    """Exploiter z mode after each reset (a random fixed archetype diversifies exploiters)."""
    max_historical: int = 24
    """Unprotected historical players kept; the least useful are pruned beyond this."""
    tables_per_plan: int = 256
    """Concrete tables sampled per matchmaking plan; envs draw one per episode."""
    total_env_steps: int = 30_000_000
    """Learner decisions over all learning players."""


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
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
