"""Native PyTorch board GNN with context-conditioned spatial policy heads.

Shared entity encoders and mean aggregation are equivariant to consistent relabeling of
features AND topology; graph pooling is invariant. With the canonical topology buffers,
this also covers its rotation/reflection automorphisms. There are no coordinates or learned
board IDs. Relative player/resource slots retain the observation's fixed semantic ordering.
"""

import math

import torch
from torch import Tensor, nn

from carla.config.schema import ModelConfig
from carla.engine.constants import NUM_DEV_CARD_TYPES, NUM_PLAYERS, NUM_RESOURCES
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, NUM_TILES
from carla.envs.action_space import MASK_SHAPES
from carla.envs.observation import (
    CORNER_FEATURES,
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    PLAYER_FEATURES,
    TILE_FEATURES,
    TRADE_FEATURES,
)
from carla.models.base import PolicyNetwork, PolicyOutput
from carla.models.graph import board_graph

NON_BOARD_KEYS = ("players", "hand", "dev_cards", "globals", "trade_context")
NON_BOARD_SIZE = (
    NUM_PLAYERS * PLAYER_FEATURES
    + NUM_RESOURCES
    + 2 * NUM_DEV_CARD_TYPES
    + GLOBAL_FEATURES
    + TRADE_FEATURES
)
_SPATIAL_HEADS = ("settlement", "city", "road", "robber")


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
    )


def _mean_aggregate(messages: Tensor, targets: Tensor, degree: Tensor) -> Tensor:
    """Scatter (B, incidences, H) messages into (B, destinations, H) means."""
    totals = messages.new_zeros(messages.shape[0], degree.shape[0], messages.shape[-1])
    totals.index_add_(1, targets, messages)
    return totals / degree[None, :, None]


class MessagePassingLayer(nn.Module):
    """One corner/edge update and, optionally, bidirectional tile/corner communication.

    Topology tensors and precomputed degrees are registered once by `GNNPolicy` and shared
    by all rounds. Edges use sums and absolute differences of their endpoint embeddings,
    so exchanging endpoints cannot change a road embedding.
    """

    def __init__(self, hidden_dim: int, tile_nodes: bool = True) -> None:
        super().__init__()
        self.tile_nodes = tile_nodes
        self.corner_message = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
        self.corner_update = _mlp((3 if tile_nodes else 2) * hidden_dim, hidden_dim)
        self.corner_norm = nn.LayerNorm(hidden_dim)
        self.edge_update = _mlp(3 * hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        if tile_nodes:
            self.tile_message = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.tile_update = _mlp(2 * hidden_dim, hidden_dim)
            self.tile_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, corners: Tensor, edges: Tensor, tiles: Tensor, graph: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Map (B, 54, H), (B, 72, H), (B, 19, H) to embeddings of the same shapes."""
        messages = self.corner_message(
            torch.cat((corners[:, graph["source"]], edges[:, graph["link_edge"]]), dim=-1)
        )
        neighbors = _mean_aggregate(messages, graph["target"], graph["corner_degree"])
        inputs = [corners, neighbors]
        if self.tile_nodes:
            tile_messages = self.tile_message(tiles)[:, graph["tile"]]
            inputs.append(
                _mean_aggregate(tile_messages, graph["tile_corner"], graph["corner_tile_degree"])
            )
        corners = self.corner_norm(corners + self.corner_update(torch.cat(inputs, dim=-1)))

        u, v = corners[:, graph["edge_u"]], corners[:, graph["edge_v"]]
        edges = self.edge_norm(
            edges + self.edge_update(torch.cat((edges, u + v, (u - v).abs()), dim=-1))
        )
        if self.tile_nodes:
            adjacent = _mean_aggregate(
                corners[:, graph["tile_corner"]], graph["tile"], graph["tile_degree"]
            )
            tiles = self.tile_norm(tiles + self.tile_update(torch.cat((tiles, adjacent), dim=-1)))
        return corners, edges, tiles


class SpatialHead(nn.Module):
    """Shared additive pointer scorer, with optional channels for joint tile/victim logits."""

    def __init__(self, hidden_dim: int, channels: int = 1) -> None:
        super().__init__()
        self.entity = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, channels)

    def forward(self, entities: Tensor, context: Tensor) -> Tensor:
        # (B, entities, channels); a nonlinear interaction lets context change rankings.
        return self.score(torch.tanh(self.entity(entities) + self.context(context)[:, None, :]))


class GNNPolicy(PolicyNetwork):
    """Feed-forward policy/value network implementing the existing factored PPO contract.

    With `gnn_tile_nodes=False`, projected tile features are mean-folded into corners once;
    tiles skip recurrent message updates and are read out from their own projection plus
    their six encoded corners. Per-tile information is retained for pooling and robber logits.
    Event history is unused, matching the feed-forward MLP baseline.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_dim <= 0 or config.gnn_layers <= 0:
            raise ValueError("GNN hidden_dim and gnn_layers must be positive")
        self.tile_nodes = config.gnn_tile_nodes
        hidden = config.hidden_dim
        self.corner_input = nn.Sequential(nn.Linear(CORNER_FEATURES, hidden), nn.ReLU())
        self.edge_input = nn.Sequential(nn.Linear(EDGE_FEATURES, hidden), nn.ReLU())
        self.tile_input = nn.Sequential(nn.Linear(TILE_FEATURES, hidden), nn.ReLU())

        canonical = board_graph()
        # edge_corner enumerates each canonical edge's two endpoint incidences.
        endpoints = canonical["edge_corner"][1].reshape(NUM_EDGES, 2).T
        tile, tile_corner = canonical["tile_corner"]
        source = torch.cat((endpoints[0], endpoints[1]))
        target = torch.cat((endpoints[1], endpoints[0]))
        graph = {
            "edge_u": endpoints[0],
            "edge_v": endpoints[1],
            "source": source,
            "target": target,
            "link_edge": torch.arange(NUM_EDGES).repeat(2),
            "tile": tile,
            "tile_corner": tile_corner,
            "corner_degree": torch.bincount(target, minlength=NUM_CORNERS).float().clamp_min(1),
            "corner_tile_degree": torch.bincount(tile_corner, minlength=NUM_CORNERS)
            .float()
            .clamp_min(1),
            "tile_degree": torch.bincount(tile, minlength=NUM_TILES).float().clamp_min(1),
        }
        self._graph_keys = tuple(graph)
        for key, tensor in graph.items():
            self.register_buffer(key, tensor)

        if not self.tile_nodes:
            self.tile_fold = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
            self.tile_readout = _mlp(2 * hidden, hidden)
            self.tile_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden, self.tile_nodes) for _ in range(config.gnn_layers)
        )
        self.context = nn.Sequential(
            nn.Linear(3 * hidden + NON_BOARD_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.spatial_heads = nn.ModuleDict(
            {
                key: SpatialHead(hidden, NUM_PLAYERS if key == "robber" else 1)
                for key in _SPATIAL_HEADS
            }
        )
        self.heads = nn.ModuleDict(
            {
                f"logits_{key}": nn.Linear(hidden, math.prod(shape))
                for key, shape in MASK_SHAPES.items()
                if key not in _SPATIAL_HEADS
            }
        )
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: dict[str, Tensor], state: Tensor | None = None) -> PolicyOutput:
        batch = obs["globals"].shape[0]
        graph = {key: getattr(self, key) for key in self._graph_keys}
        corners = self.corner_input(obs["corners"].float())  # (B, 54, H)
        edges = self.edge_input(obs["edges"].float())  # (B, 72, H)
        tiles = self.tile_input(obs["tiles"].float())  # (B, 19, H)
        if not self.tile_nodes:
            folded = _mean_aggregate(
                tiles[:, graph["tile"]], graph["tile_corner"], graph["corner_tile_degree"]
            )
            corners = self.tile_fold(torch.cat((corners, folded), dim=-1))
        for layer in self.layers:
            corners, edges, tiles = layer(corners, edges, tiles, graph)
        if not self.tile_nodes:
            adjacent = _mean_aggregate(
                corners[:, graph["tile_corner"]], graph["tile"], graph["tile_degree"]
            )
            tiles = self.tile_norm(tiles + self.tile_readout(torch.cat((tiles, adjacent), dim=-1)))

        pooled = [entities.mean(dim=1) for entities in (corners, edges, tiles)]  # 3 * (B, H)
        extra = [obs[key].reshape(batch, -1).float() for key in NON_BOARD_KEYS]
        context = self.context(torch.cat((*pooled, *extra), dim=-1))  # (B, H)
        spatial = {"settlement": corners, "city": corners, "road": edges, "robber": tiles}
        logits = {
            key: (
                self.spatial_heads[key](spatial[key], context)
                if key in spatial
                else self.heads[f"logits_{key}"](context)
            ).reshape(batch, *shape)
            for key, shape in MASK_SHAPES.items()
        }
        return PolicyOutput(logits=logits, value=self.value(context).squeeze(1))
