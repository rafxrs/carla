"""The static board graph as tensors.

The topology is identical in every game, so a batch of boards shares one graph: message passing
is a fixed gather/scatter (or a dense 54x54 adjacency matmul) broadcast over the batch, with no
graph library and no per-sample batching.
"""

import torch
from torch import Tensor

from carla.engine.topology import EDGE_CORNERS, NUM_CORNERS, TILE_CORNERS


def board_graph(device: torch.device | str = "cpu") -> dict[str, Tensor]:
    """Index tensors (int64):

    corner_corner (2, 144): directed corner->corner links (both directions of each edge)
    edge_corner   (2, 144): (edge id, corner id) incidences
    tile_corner   (2, 114): (tile id, corner id) incidences
    corner_adjacency (54, 54): dense 0/1 float adjacency
    """
    pairs = torch.tensor(EDGE_CORNERS, dtype=torch.long)
    corner_corner = torch.cat([pairs.T, pairs.flip(1).T], dim=1)
    edges = torch.arange(len(EDGE_CORNERS)).repeat_interleave(2)
    edge_corner = torch.stack([edges, pairs.reshape(-1)])
    tiles = torch.arange(len(TILE_CORNERS)).repeat_interleave(6)
    tile_corner = torch.stack([tiles, torch.tensor(TILE_CORNERS).reshape(-1)])
    adjacency = torch.zeros(NUM_CORNERS, NUM_CORNERS)
    adjacency[corner_corner[0], corner_corner[1]] = 1.0
    graph = {
        "corner_corner": corner_corner,
        "edge_corner": edge_corner,
        "tile_corner": tile_corner,
        "corner_adjacency": adjacency,
    }
    return {k: v.to(device) for k, v in graph.items()}
