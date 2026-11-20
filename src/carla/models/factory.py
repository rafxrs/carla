"""Build the policy network selected by a `ModelConfig`."""

from carla.config.schema import ModelConfig
from carla.models.base import PolicyNetwork
from carla.models.gnn import GNNPolicy
from carla.models.mlp import MLPPolicy


def build_policy(config: ModelConfig) -> PolicyNetwork:
    """Build either board encoder with optional GRU trade decoding."""
    if config.belief != "none" or config.set_attention or config.trade_decoder == "transformer":
        raise NotImplementedError(
            "belief, set attention, and transformer trade decoding are not implemented"
        )
    if config.max_trade_cards_per_side < 1:
        raise ValueError("model.max_trade_cards_per_side must be positive")
    if config.encoder == "mlp":
        return MLPPolicy(config)
    if config.encoder == "gnn":
        return GNNPolicy(config)
    raise ValueError(f"unknown encoder {config.encoder!r}")
