"""Autoregressive domestic-trade decoder (TODO: models workstream; engine support pending).

Emits a token sequence: give-resources (masked so that the running total never exceeds the
proposer's hand), then receive-resources (masked to types not given), then STOP. Tokens:
0..4 = one card of a resource, 5 = switch from give to receive, 6 = stop. The result is a
`carla.engine.actions.TradeOffer` for a PROPOSE_TRADE action; its log-probability is the sum
over tokens and is added to the type head's log-probability.
"""

from torch import Tensor, nn


class TradeDecoder(nn.Module):
    """context (B, d) + own hand (B, 5) -> sampled offers, token log-probs. TODO(models)."""

    def __init__(self, dim: int, kind: str = "gru", max_tokens: int = 12) -> None:
        super().__init__()
        raise NotImplementedError("models workstream: autoregressive trade decoder")

    def forward(self, context: Tensor, hand: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
