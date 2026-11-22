"""Set-attention pooling for variable-size sets.

Uses: the four seats (relative order, from the `players` array plus their trade responses) and a
seat's development cards. Block: multi-head self-attention over the elements with a key-padding
mask, a feed-forward layer, then attention pooling by a learned query (Set Transformer style,
`model.attention_heads` heads).
"""

import torch
from torch import Tensor, nn


class SetAttentionPool(nn.Module):
    """(B, N, d) elements + (B, N) validity mask -> (B, d) pooled vector.

    Rows without any valid element pool to zeros.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must be divisible by attention_heads")
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * dim**-0.5)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, elements: Tensor, mask: Tensor) -> Tensor:
        valid = mask.bool()
        empty = ~valid.any(dim=1)
        # All-masked rows would yield NaN attention weights; attend to everything, then zero.
        padding = ~(valid | empty[:, None])
        attended, _ = self.attention(
            elements, elements, elements, key_padding_mask=padding, need_weights=False
        )
        x = self.norm1(elements + attended)
        x = self.norm2(x + self.feed_forward(x))
        query = self.query.expand(elements.shape[0], -1, -1)
        pooled, _ = self.pool(query, x, x, key_padding_mask=padding, need_weights=False)
        return pooled.squeeze(1).masked_fill(empty[:, None], 0.0)
