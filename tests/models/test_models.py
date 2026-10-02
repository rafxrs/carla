import pytest
import torch

from carla.config.schema import ModelConfig
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.envs.action_space import (
    MASK_SHAPES,
    build_masks,
    decode_action,
    encode_action,
)
from carla.envs.catan_env import CatanEnv
from carla.envs.observation import ObservationEncoder
from carla.models import (
    FactoredDistribution,
    MLPPolicy,
    batch_observations,
    board_graph,
    build_policy,
)


def sampled_positions(n_games: int = 2, every: int = 7):
    """(legal actions, actor) at regularly spaced decisions of random games."""
    out = []
    for seed in range(n_games):
        game, rng = Game(seed=seed), Rng(seed)
        while not game.is_over:
            if game.num_steps % every == 0:
                out.append((game.legal_actions(), game.decision_player))
            game.step(rng.choice(game.legal_actions()))
    return out


def masks_and_logits(legal, actor, generator):
    masks = {k: torch.as_tensor(v)[None] for k, v in build_masks(legal, actor).items()}
    logits = {k: torch.randn((1, *s), generator=generator) * 3 for k, s in MASK_SHAPES.items()}
    return masks, logits


def expand(tensors, n):
    return {k: v.expand(n, *v.shape[1:]) for k, v in tensors.items()}


def test_distribution_is_normalised_over_legal_actions_and_entropy_is_exact():
    generator = torch.Generator().manual_seed(0)
    for legal, actor in sampled_positions():
        masks, logits = masks_and_logits(legal, actor, generator)
        heads = torch.stack([torch.as_tensor(encode_action(a, actor)) for a in legal])
        logp = FactoredDistribution(expand(logits, len(legal)), expand(masks, len(legal))).log_prob(
            heads
        )
        assert torch.logsumexp(logp, 0).item() == pytest.approx(0.0, abs=1e-5)
        entropy = FactoredDistribution(logits, masks).entropy().item()
        assert entropy == pytest.approx(-(logp.exp() * logp).sum().item(), abs=1e-4)


def test_samples_are_always_legal():
    generator = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    for legal, actor in sampled_positions(every=3):
        masks, logits = masks_and_logits(legal, actor, generator)
        dist = FactoredDistribution(logits, masks)
        for deterministic in (False, True):
            heads = dist.sample(deterministic=deterministic)
            assert decode_action(heads[0].tolist(), actor) in legal
            assert torch.isfinite(dist.log_prob(heads)).all()


def test_mlp_policy_runs_on_env_observations():
    env = CatanEnv()
    obs, _ = env.reset(seed=0)
    batch = batch_observations([obs, obs])
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=2))
    assert isinstance(policy, MLPPolicy) and policy.initial_state(2) is None
    out = policy(batch)
    assert out.value.shape == (2,)
    for key, shape in MASK_SHAPES.items():
        assert out.logits[key].shape == (2, *shape)
    heads = out.distribution(batch["masks"]).sample()
    assert decode_action(heads[0].tolist(), env.learner) in env.game.legal_actions()


def test_unimplemented_variants_fail_loudly():
    for cfg in (
        ModelConfig(encoder="gnn"),
        ModelConfig(belief="gru"),
        ModelConfig(set_attention=True),
        ModelConfig(trade_decoder="gru"),
    ):
        with pytest.raises(NotImplementedError):
            build_policy(cfg)


def test_board_graph():
    g = board_graph()
    assert g["corner_corner"].shape == (2, 144) and g["edge_corner"].shape == (2, 144)
    assert g["tile_corner"].shape == (2, 114)
    adjacency = g["corner_adjacency"]
    assert torch.equal(adjacency, adjacency.T) and adjacency.sum().item() == 144


def test_observation_batch_dtypes():
    encoder = ObservationEncoder()
    game = Game(seed=0)
    batch = batch_observations([encoder.encode(game, 0)])
    assert batch["corners"].dtype == torch.float32 and batch["masks"]["type"].dtype  # type: ignore[index] == torch.bool
