"""MLP baseline (control condition): flattened features, no structural inductive bias."""

import math

import torch
from torch import Tensor, nn

from carla.config.schema import ModelConfig
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

FLAT_KEYS = (
    "tiles",
    "corners",
    "edges",
    "players",
    "hand",
    "dev_cards",
    "globals",
    "trade_context",
)
FLAT_SIZE = (
    19 * TILE_FEATURES
    + 54 * CORNER_FEATURES
    + 72 * EDGE_FEATURES
    + 4 * PLAYER_FEATURES
    + 5
    + 10
    + GLOBAL_FEATURES
    + TRADE_FEATURES
)


class MLPPolicy(PolicyNetwork):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = FLAT_SIZE
        for _ in range(config.mlp_layers):
            layers += [nn.Linear(width, config.hidden_dim), nn.ReLU()]
            width = config.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(  # "type" would shadow nn.Module.type, hence the prefix
            {f"logits_{k}": nn.Linear(width, math.prod(s)) for k, s in MASK_SHAPES.items()}
        )
        self.value = nn.Linear(width, 1)

    def forward(self, obs: dict[str, Tensor], state: Tensor | None = None) -> PolicyOutput:
        batch = obs["globals"].shape[0]
        x = torch.cat([obs[k].reshape(batch, -1).float() for k in FLAT_KEYS], dim=1)
        h = self.trunk(x)
        logits = {
            k: self.heads[f"logits_{k}"](h).reshape(batch, *shape)
            for k, shape in MASK_SHAPES.items()
        }
        return PolicyOutput(logits=logits, value=self.value(h).squeeze(1))
