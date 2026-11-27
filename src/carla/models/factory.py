"""Build the policy network selected by a `ModelConfig`."""

from carla.config.schema import ModelConfig
from carla.models.base import PolicyNetwork
from carla.models.gnn import GNNPolicy
from carla.models.mlp import MLPPolicy
from carla.models.strategy_net import StrategyNet


def build_policy(config: ModelConfig) -> PolicyNetwork:
    """Build either board encoder with optional GRU trade decoding.

    Set attention and the belief GRU are implemented for the GNN encoder only.
    """
    if config.trade_decoder == "transformer":
        raise NotImplementedError("transformer trade decoding is not implemented")
    if config.belief == "lstm":
        raise NotImplementedError("only the GRU belief module is implemented")
    if config.encoder == "mlp" and (config.belief != "none" or config.set_attention):
        raise NotImplementedError("belief and set attention require model.encoder=gnn")
    if config.encoder != "strategy" and (
        config.core != "none"
        or config.strategy_conditioning
        or config.strategy_selector
        or config.privileged_value
    ):
        raise NotImplementedError(
            "the recurrent core, strategy conditioning, selector and privileged critic "
            "require model.encoder=strategy"
        )
    if config.strategy_selector and not config.strategy_conditioning:
        raise ValueError("model.strategy_selector needs model.strategy_conditioning")
    if config.max_trade_cards_per_side < 1:
        raise ValueError("model.max_trade_cards_per_side must be positive")
    if config.encoder == "mlp":
        return MLPPolicy(config)
    if config.encoder == "gnn":
        return GNNPolicy(config)
    if config.encoder == "strategy":
        return StrategyNet(config)
    raise ValueError(f"unknown encoder {config.encoder!r}")
