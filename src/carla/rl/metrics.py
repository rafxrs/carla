"""Training episode accounting and observation-conditioned policy diagnostics."""

from dataclasses import dataclass, field

import numpy as np
import torch

from carla.engine.actions import ActionType
from carla.engine.state import Phase
from carla.envs.action_space import MASK_SHAPES
from carla.envs.strategy import STAT_NAMES
from carla.models.distributions import TradeDistribution

HEAD_METRICS = tuple(f"head_{key}_entropy" for key in MASK_SHAPES)
DIAGNOSTIC_FIELDS = (
    "main_choice_decisions",
    "main_type_entropy",
    "main_type_max_probability",
    "main_buildable_decisions",
    "main_end_turn_probability_when_buildable",
    "main_end_turn_rate_when_buildable",
    *HEAD_METRICS,
)
PPO_FIELDS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "grad_norm",
    "auxiliary_loss",
    "explained_variance",
    "epochs_stopped_early",
    "optimizer_steps",
    "advantage_mean",
    "advantage_std",
    "return_mean",
    "return_std",
    "value_mean",
    "value_std",
)
STRATEGY_UPDATE_FIELDS = (
    "strategy_value_loss",
    "strategy_return_mean",
    "strategy_advantage_std",
    "selector_entropy",
    "selections",
    "upgo_loss",
)
"""Strategy pseudo-reward critic, selector and UPGO statistics (empty when unused)."""
UPDATE_FIELDS = (
    "update",
    "env_steps",
    "samples",
    "optimizer_steps_total",
    "episodes_completed",
    "terminal_outcomes",
    "truncations",
    "nonzero_rewards",
    "collection_seconds",
    "update_seconds",
    "training_seconds",
    "evaluation_seconds",
    "wall_seconds",
    "decisions_per_second",
    *PPO_FIELDS,
    *DIAGNOSTIC_FIELDS,
    "reference_kl",  # Appended last: older journals gain it on resume (see CsvJournal).
    *STRATEGY_UPDATE_FIELDS,
)
STRATEGY_EPISODE_FIELDS = (
    "strategy",
    "strategy_choices",
    "strategy_switches",
    "strategy_adherence",
    "strategy_return",
    *(f"stat_{name}" for name in STAT_NAMES),
)
"""Per-episode strategy diagnostics: the learner's variant at the start, the selector's choices
(archetype indices joined by '-'), switches, final distance to z, summed pseudo-reward, and the
learner's public build statistics."""
EPISODE_FIELDS = (
    "update",
    "env_steps",
    "env_id",
    "episode_id",
    "game_seed",
    "seat",
    "opponent",
    "outcome",
    "winner",
    "return",
    "learner_decisions",
    "turns",
    "victory_points",
    *STRATEGY_EPISODE_FIELDS,
)
EVALUATION_FIELDS = (
    "update",
    "env_steps",
    "evaluation_target",
    "training_seconds",
    "wall_seconds",
    "checkpoint",
    "split",
    "eval_seed",
    "board",
    "game_seed",
    "seat",
    "opponent",
    "mode",
    "outcome",
    "winner",
    "return",
    "learner_decisions",
    "turns",
    "victory_points",
    "strategy",
    "strategy_choices",
    *(f"stat_{name}" for name in STAT_NAMES),
)


@dataclass
class EpisodeTracker:
    episode_id: int = 0
    reward: float = 0.0
    decisions: int = 0
    partial: bool = False
    """Legacy checkpoints lack episode accumulators; omit that first incomplete record."""
    strategy_return: float = 0.0

    def step(
        self,
        reward: float,
        env,
        done: bool,
        env_id: int,
        env_steps: int,
        strategy_reward: float = 0.0,
    ) -> dict | None:
        self.reward += reward
        self.strategy_return += strategy_reward
        self.decisions += 1
        if not done:
            return None
        winner = env.game.winner
        record = {
            "env_steps": env_steps,
            "env_id": env_id,
            "episode_id": self.episode_id,
            "game_seed": env.game.seed,
            "seat": env.learner,
            "opponent": ",".join(agent.name for agent in env.opponents),
            "outcome": (
                "truncated"
                if env.game.is_truncated
                else ("win" if winner == env.learner else "loss")
            ),
            "winner": -1 if winner is None else winner,
            "return": self.reward,
            "learner_decisions": self.decisions,
            "turns": env.game.turn,
            "victory_points": env.game.victory_points(env.learner),
            **strategy_record(env, self.strategy_return),
        }
        partial = self.partial
        self.episode_id += 1
        self.reward, self.decisions, self.partial = 0.0, 0, False
        self.strategy_return = 0.0
        return None if partial else record


def strategy_record(env, strategy_return: float = 0.0) -> dict:
    """Strategy diagnostics of the learner's finished episode in `env` (see the fields)."""
    stats = env.aec.stats[env.learner]
    controller = getattr(env, "strategy", None)
    out = {f"stat_{name}": float(v) for name, v in zip(STAT_NAMES, stats.counts, strict=True)}
    if controller is None:
        out.update(
            {
                "strategy": "",
                "strategy_choices": "",
                "strategy_switches": "",
                "strategy_adherence": "",
                "strategy_return": "",
            }
        )
        return out
    out.update(
        {
            "strategy": controller.mode if controller.mode != "prescribed" else controller.label,
            "strategy_choices": "-".join(str(c) for c in controller.choices),
            "strategy_switches": controller.switches,
            "strategy_adherence": controller.adherence(stats),
            "strategy_return": strategy_return,
        }
    )
    return out


@dataclass
class RolloutDiagnostics:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "RolloutDiagnostics") -> None:
        for key, value in other.sums.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
        for key, value in other.counts.items():
            self.counts[key] = self.counts.get(key, 0) + value

    def add(self, key: str, values: np.ndarray) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + float(values.sum())
        self.counts[key] = self.counts.get(key, 0) + len(values)

    def observe(self, dist, batch: dict, heads: torch.Tensor) -> None:
        keys = list(MASK_SHAPES)
        entropies = (
            torch.stack(
                [
                    -(lp.exp() * torch.where(torch.isfinite(lp), lp, 0.0)).sum(1)
                    for lp in (dist.logp[key] for key in keys)
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )
        masks = {
            key: value.reshape(len(heads), -1).cpu().numpy()
            for key, value in batch["masks"].items()
        }
        for i, key in enumerate(keys):
            if key == "trade" and isinstance(dist, TradeDistribution):
                # The one-for-one head is unused; report entropy along sampled offer prefixes.
                chosen = heads[:, 0] == ActionType.PROPOSE_TRADE
                self.add("head_trade_entropy", dist._conditional_entropy[chosen].cpu().numpy())
                continue
            # Dummy uniform distributions for unavailable heads must not enter the mean.
            self.add(f"head_{key}_entropy", entropies[masks[key].any(1), i])
        main = batch["globals"][:, Phase.MAIN].cpu().numpy().astype(bool)
        choices = main & (masks["type"].sum(1) > 1)
        probabilities = dist.logp["type"].exp().cpu().numpy()
        self.add("main_type_entropy", entropies[choices, keys.index("type")])
        self.add("main_type_max_probability", probabilities[choices].max(1))
        buildable = main & masks["type"][
            :, [ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY]
        ].any(1)
        self.add(
            "main_end_turn_probability_when_buildable",
            probabilities[buildable, ActionType.END_TURN],
        )
        self.add(
            "main_end_turn_rate_when_buildable",
            (heads[:, 0].cpu().numpy()[buildable] == ActionType.END_TURN).astype(float),
        )

    def result(self) -> dict:
        result = {
            key: self.sums[key] / count if count else "" for key, count in self.counts.items()
        }
        result["main_choice_decisions"] = self.counts.get("main_type_entropy", 0)
        result["main_buildable_decisions"] = self.counts.get(
            "main_end_turn_probability_when_buildable", 0
        )
        return result
