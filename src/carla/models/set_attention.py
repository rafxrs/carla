"""Set-attention modules for variable-size sets (TODO: models workstream).

Uses: a seat's development cards (held + played), the three opponents (relative seats, from the
`players` array), and the resources being composed into a trade offer. Intended block: multi-head
self-attention over set elements with a key-padding mask, then attention pooling by a learned
query (Set Transformer style, `model.attention_heads` heads).
"""

from torch import Tensor, nn


class SetAttentionPool(nn.Module):
    """(B, N, d) elements + (B, N) validity mask -> (B, d) pooled vector. TODO(models)."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        raise NotImplementedError("models workstream: set attention pooling")

    def forward(self, elements: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError
