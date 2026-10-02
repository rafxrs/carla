"""Build the policy network selected by a `ModelConfig`."""

from carla.config.schema import ModelConfig
from carla.models.base import PolicyNetwork
from carla.models.gnn import GNNPolicy
from carla.models.mlp import MLPPolicy


def build_policy(config: ModelConfig) -> PolicyNetwork:
    """TODO(models): wire set attention, belief module and trade decoder into both encoders."""
    if config.belief != "none" or config.set_attention or config.trade_decoder != "none":
        raise NotImplementedError(
            "belief / set attention / trade decoder are not implemented yet (models workstream)"
        )
    if config.encoder == "mlp":
        return MLPPolicy(config)
    if config.encoder == "gnn":
        return GNNPolicy(config)
    raise ValueError(f"unknown encoder {config.encoder!r}")
