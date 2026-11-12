"""Training episode accounting and observation-conditioned policy diagnostics."""

from dataclasses import dataclass, field

import numpy as np
import torch

from carla.engine.actions import ActionType
from carla.engine.state import Phase
from carla.envs.action_space import MASK_SHAPES

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
)
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
)


@dataclass
class EpisodeTracker:
    episode_id: int = 0
    reward: float = 0.0
    decisions: int = 0
    partial: bool = False
    """Legacy checkpoints lack episode accumulators; omit that first incomplete record."""

    def step(self, reward: float, env, done: bool, env_id: int, env_steps: int) -> dict | None:
        self.reward += reward
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
        }
        partial = self.partial
        self.episode_id += 1
        self.reward, self.decisions, self.partial = 0.0, 0, False
        return None if partial else record


@dataclass
class RolloutDiagnostics:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

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
