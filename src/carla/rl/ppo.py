"""PPO learner (TODO: rl workstream).

Loss per minibatch, with advantages from `compute_gae` (normalised per batch):
  policy  = -mean(min(ratio * A, clip(ratio, 1 - clip_coef, 1 + clip_coef) * A)),
            ratio = exp(dist.log_prob(actions) - old_log_probs)
  value   = value_coef * mean((V - returns)^2)            (optionally clipped)
  entropy = -entropy_coef * mean(dist.entropy())
  belief  = model.belief_loss_coef * NLL(belief predictions, hidden_truth)   (belief models)
Updates: `ppo.epochs` passes over shuffled minibatches (whole sequences of
`ppo.sequence_length` steps for recurrent models), grad-norm clipping, optional early stop on
`ppo.target_kl`. Stale batches (version older than `ppo.max_policy_lag`) are dropped.
"""

from carla.config.schema import PPOConfig
from carla.models.base import PolicyNetwork
from carla.rl.buffer import RolloutBatch


class PPOLearner:
    def __init__(self, policy: PolicyNetwork, config: PPOConfig, device: str = "cpu") -> None:
        self.policy = policy.to(device)
        self.config = config
        self.device = device
        self.version = 0

    def update(self, batches: list[RolloutBatch]) -> dict[str, float]:
        """One PPO update on `batches`; returns training statistics. TODO(rl)."""
        raise NotImplementedError("rl workstream: PPO update")
