"""GNN board encoder and pointer-network policy (TODO: models workstream).

Plan:
- Nodes: corners (54) and optionally tiles (19); road edges (72) are edge features on
  corner-corner links (or nodes of their own). Inputs: the `corners`, `edges`, `tiles` arrays of
  the observation; no coordinates, so the encoder is rotation/reflection invariant by design.
- `gnn_layers` rounds of hand-rolled message passing over `board_graph()` (mean/sum aggregation
  with `index_add_`, residual + LayerNorm), giving per-corner, per-edge and per-tile embeddings.
- A global context (pooled nodes + `players`, `hand`, `dev_cards`, `globals`) feeds the type head
  and the value head.
- Pointer heads: settlement/city logits = score(corner embedding, context), road logits =
  score(edge embedding, context), robber logits = score(tile embedding, context) + victim term.
"""

from torch import Tensor

from carla.config.schema import ModelConfig
from carla.models.base import PolicyNetwork, PolicyOutput


class MessagePassingLayer:
    """One round of corner/edge/tile message passing. TODO(models)."""


class GNNPolicy(PolicyNetwork):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("models workstream: GNN encoder with pointer heads")

    def forward(self, obs: dict[str, Tensor], state: Tensor | None = None) -> PolicyOutput:
        raise NotImplementedError
