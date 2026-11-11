from copy import deepcopy

import pytest
import torch

from carla.config.schema import ModelConfig
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS, CORNER_TILES
from carla.envs.action_space import MASK_SHAPES, decode_action
from carla.envs.observation import ObservationEncoder
from carla.models import GNNPolicy, MLPPolicy, batch_observations, build_policy


@pytest.fixture
def observations():
    encoder = ObservationEncoder()
    return [encoder.encode(game, game.decision_player) for game in (Game(seed=0), Game(seed=1))]


@pytest.mark.parametrize("tile_nodes", [True, False])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gnn_forward_contract_and_batch_independence(observations, tile_nodes, batch_size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = build_policy(
        ModelConfig(encoder="gnn", hidden_dim=16, gnn_layers=2, gnn_tile_nodes=tile_nodes)
    ).to(device)
    assert isinstance(policy, GNNPolicy)
    assert policy.initial_state(batch_size, device=device) is None
    selected = [observations[i % len(observations)] for i in range(batch_size)]
    batch = batch_observations(selected, device=device)
    out = policy(batch)
    reference = MLPPolicy(ModelConfig(hidden_dim=16, mlp_layers=1)).to(device)(batch)
    assert out.state is None and out.belief is None
    assert out.value.shape == (batch_size,) and torch.isfinite(out.value).all()
    assert set(out.logits) == set(reference.logits) == set(MASK_SHAPES)
    for key, shape in MASK_SHAPES.items():
        assert out.logits[key].shape == reference.logits[key].shape == (batch_size, *shape)
        assert torch.isfinite(out.logits[key]).all()
    assert all(buffer.device.type == device for buffer in policy.buffers())
    for i, obs in enumerate(selected):
        single = policy(batch_observations([obs], device=device))
        torch.testing.assert_close(out.value[i], single.value[0])
        for key in MASK_SHAPES:
            torch.testing.assert_close(out.logits[key][i], single.logits[key][0])


@pytest.mark.parametrize("tile_nodes", [True, False])
def test_gnn_masked_distribution_samples_legal_actions_across_game(tile_nodes):
    torch.manual_seed(3)
    policy = build_policy(
        ModelConfig(encoder="gnn", hidden_dim=16, gnn_layers=2, gnn_tile_nodes=tile_nodes)
    )
    encoder, game, rng = ObservationEncoder(), Game(seed=3), Rng(3)
    checked = 0
    while not game.is_over and game.num_steps < 100:
        if game.num_steps % 5 == 0:
            actor, legal = game.decision_player, game.legal_actions()
            batch = batch_observations([encoder.encode(game, actor)] * 2)
            dist = policy(batch).distribution(batch["masks"])
            for deterministic in (False, True):
                heads = dist.sample(deterministic=deterministic)
                assert all(decode_action(h.tolist(), actor) in legal for h in heads)
                assert torch.isfinite(dist.log_prob(heads)).all()
            assert torch.isfinite(dist.entropy()).all()
            checked += 1
        game.step(rng.choice(game.legal_actions()))
    assert checked >= 2


@pytest.mark.parametrize("tile_nodes", [True, False])
def test_gnn_backward_reaches_all_parameters(observations, tile_nodes):
    torch.manual_seed(5)
    policy = build_policy(
        ModelConfig(encoder="gnn", hidden_dim=16, gnn_layers=2, gnn_tile_nodes=tile_nodes)
    )
    batch = batch_observations(observations)
    out = policy(batch)
    dist = out.distribution(batch["masks"])
    # Include every raw head, since a legal setup decision only selects settlement/road heads.
    loss = sum(logits.square().mean() for logits in out.logits.values()) + out.value.square().mean()
    loss = loss - dist.log_prob(dist.sample()).mean() - 0.01 * dist.entropy().mean()
    loss.backward()
    for name, parameter in policy.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for module in (policy.corner_input, policy.edge_input, policy.tile_input, policy.layers):
        assert any(parameter.grad.abs().sum() > 0 for parameter in module.parameters())


def test_messages_carry_neighbor_edge_and_tile_information():
    torch.manual_seed(6)
    policy = GNNPolicy(ModelConfig(encoder="gnn", hidden_dim=16, gnn_layers=1))
    corners = torch.randn(1, 54, 16, requires_grad=True)
    edges = torch.randn(1, 72, 16, requires_grad=True)
    tiles = torch.randn(1, 19, 16, requires_grad=True)
    graph = {key: getattr(policy, key) for key in policy._graph_keys}
    encoded_corners, encoded_edges, encoded_tiles = policy.layers[0](corners, edges, tiles, graph)
    encoded_corners[0, 0, 0].backward(retain_graph=True)
    assert corners.grad[0, list(CORNER_NEIGHBORS[0])].abs().sum() > 0
    assert edges.grad[0, list(CORNER_EDGES[0])].abs().sum() > 0
    assert tiles.grad[0, list(CORNER_TILES[0])].abs().sum() > 0
    # The road and tile readouts also receive corner information.
    for output in (encoded_edges, encoded_tiles):
        gradient = torch.autograd.grad(output[0, 0, 0], corners, retain_graph=True)[0]
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("tile_nodes", [True, False])
def test_spatial_equivariance_and_global_invariance_under_relabeling(observations, tile_nodes):
    torch.manual_seed(7)
    policy = GNNPolicy(
        ModelConfig(encoder="gnn", hidden_dim=16, gnn_layers=2, gnn_tile_nodes=tile_nodes)
    )
    relabeled = deepcopy(policy)
    order = {key: torch.randperm(n) for key, n in (("corners", 54), ("edges", 72), ("tiles", 19))}
    inverse = {key: indices.argsort() for key, indices in order.items()}
    batch = batch_observations(observations)
    permuted = {
        key: value[:, order[key]] if key in order else value for key, value in batch.items()
    }

    # New array slot i represents old slot order[i]; transform the incidence tables with it.
    relabeled.edge_u = inverse["corners"][policy.edge_u][order["edges"]]
    relabeled.edge_v = inverse["corners"][policy.edge_v][order["edges"]]
    relabeled.source = torch.cat((relabeled.edge_u, relabeled.edge_v))
    relabeled.target = torch.cat((relabeled.edge_v, relabeled.edge_u))
    relabeled.tile = inverse["tiles"][policy.tile]
    relabeled.tile_corner = inverse["corners"][policy.tile_corner]
    relabeled.corner_degree = policy.corner_degree[order["corners"]]
    relabeled.corner_tile_degree = policy.corner_tile_degree[order["corners"]]
    relabeled.tile_degree = policy.tile_degree[order["tiles"]]
    expected, actual = policy(batch), relabeled(permuted)
    torch.testing.assert_close(actual.value, expected.value)
    entity = {"settlement": "corners", "city": "corners", "road": "edges", "robber": "tiles"}
    for key in MASK_SHAPES:
        logits = expected.logits[key]
        if key in entity:
            logits = logits[:, order[entity[key]]]
        torch.testing.assert_close(actual.logits[key], logits)


def test_road_endpoints_are_unordered(observations):
    policy = GNNPolicy(ModelConfig(encoder="gnn", hidden_dim=16, gnn_layers=2))
    flipped = deepcopy(policy)
    flipped.edge_u, flipped.edge_v = policy.edge_v.clone(), policy.edge_u.clone()
    flipped.source, flipped.target = policy.target.clone(), policy.source.clone()
    batch = batch_observations(observations)
    expected, actual = policy(batch), flipped(batch)
    torch.testing.assert_close(actual.value, expected.value)
    for key in MASK_SHAPES:
        torch.testing.assert_close(actual.logits[key], expected.logits[key])


@pytest.mark.parametrize(("hidden", "layers"), [(0, 2), (-1, 2), (16, 0), (16, -1)])
def test_gnn_rejects_invalid_dimensions(hidden, layers):
    with pytest.raises(ValueError, match="must be positive"):
        build_policy(ModelConfig(encoder="gnn", hidden_dim=hidden, gnn_layers=layers))
