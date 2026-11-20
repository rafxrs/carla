from itertools import product

import numpy as np
import pytest
import torch

from carla.config.schema import ModelConfig, PPOConfig
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.envs.action_space import HEAD_INDEX, decode_action
from carla.envs.observation import ObservationEncoder
from carla.envs.trade_tokens import offer_tokens, tokens_offer
from carla.models.base import batch_observations
from carla.models.factory import build_policy
from carla.models.trade_decoder import TradeDecoder
from carla.rl.buffer import RolloutBatch
from carla.rl.ppo import PPOLearner
from carla.rl.train import _upgrade_trade_observation
from tests.engine.scenario import give, post_setup_game


def test_offer_codec_is_canonical_and_lossless():
    offer = TradeOffer((2, 0, 1, 0, 0), (0, 1, 0, 0, 2))
    assert tokens_offer(offer_tokens(offer, 4)) == offer
    with pytest.raises(ValueError, match="sorted"):
        tokens_offer((1, 0, 5, 2, 6))
    with pytest.raises(ValueError, match="same resource"):
        tokens_offer((0, 5, 0, 6))


@pytest.mark.parametrize(("cap", "engine_cap"), [(1, 1), (4, 1), (4, 4), (7, -1)])
def test_decoder_samples_legal_offers_and_rescores_exactly(cap, engine_cap):
    torch.manual_seed(18)
    decoder = TradeDecoder(16, cap)
    context = torch.randn(32, 16, requires_grad=True)
    hand = torch.tensor([[2, 1, 0, 3, 1]]).expand(32, -1)
    limit = torch.full((32,), engine_cap)
    for deterministic in (False, True):
        tokens, _ = decoder.sample(context, hand, limit, deterministic)
        score, entropy = decoder.score(context, hand, limit, tokens)
        assert torch.isfinite(score).all() and torch.isfinite(entropy).all()
        for row in tokens.tolist():
            offer = tokens_offer(row[: row.index(6) + 1])
            assert all(g <= h for g, h in zip(offer.give, hand[0].tolist(), strict=True))
            effective_cap = min(cap, engine_cap) if engine_cap > 0 else cap
            assert sum(offer.give) <= effective_cap and sum(offer.receive) <= effective_cap
        (-score.mean()).backward(retain_graph=True)
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert decoder.output.weight.grad is not None


def test_structured_policy_offer_is_engine_legal_and_hidden_hand_independent():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 1, 0, 0, 0))
    obs = ObservationEncoder().encode(game, 0)
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=2, trade_decoder="gru"))
    batch = batch_observations([obs] * 32)
    out = policy(batch)
    dist = out.distribution(batch["masks"])
    actions = dist.sample()
    logp = dist.log_prob(actions)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(dist.entropy()).all()
    for row in actions.tolist():
        action = decode_action(row, 0)
        assert game.is_legal(action)
    # Force the type head to propose; opponent hand contents never enter its mask.
    batch["masks"]["type"][:] = False
    batch["masks"]["type"][:, ActionType.PROPOSE_TRADE] = True
    offers = policy(batch).distribution(batch["masks"]).sample()
    assert all(game.is_legal(decode_action(row, 0)) for row in offers.tolist())
    assert (offers[:, HEAD_INDEX["resource"]] == -1).all()
    assert (offers[:, HEAD_INDEX["resource2"]] == -1).all()


def test_trade_entropy_is_available_before_scoring_and_matches_saved_tokens():
    torch.manual_seed(24)
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 1, 0, 0, 0))
    batch = batch_observations([ObservationEncoder().encode(game, 0)] * 8)
    batch["masks"]["type"][:] = False
    batch["masks"]["type"][:, ActionType.PROPOSE_TRADE] = True
    policy = build_policy(ModelConfig(hidden_dim=16, mlp_layers=1, trade_decoder="gru"))
    dist = policy(batch).distribution(batch["masks"])
    standalone_entropy = dist.entropy()
    assert torch.isfinite(standalone_entropy).all()
    assert (standalone_entropy > 0).all()
    standalone_entropy.mean().backward()
    assert policy.trade_decoder.output.weight.grad is not None

    dist = policy(batch).distribution(batch["masks"])
    actions = dist.sample()
    sampled_entropy = dist.entropy()
    dist.log_prob(actions)
    assert torch.allclose(sampled_entropy, dist.entropy(), atol=1e-6)


def test_counteroffer_uses_responder_hand_and_perspective():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 0, 0, 0, 0))
    game.step(Action(ActionType.PROPOSE_TRADE, trade=TradeOffer((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))))
    responder = game.decision_player
    give(game, responder, (0, 2, 0, 0, 0))
    obs = ObservationEncoder().encode(game, responder)
    assert obs["own_hand_counts"][1] >= 2
    batch = batch_observations([obs] * 12)
    batch["masks"]["type"][:] = False
    batch["masks"]["type"][:, ActionType.PROPOSE_TRADE] = True
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1, trade_decoder="gru"))
    actions = policy(batch).distribution(batch["masks"]).sample()
    for row in actions.tolist():
        counter = decode_action(row, responder)
        assert game.is_legal(counter)
        assert all(
            g <= h for g, h in zip(counter.trade.give, game.state.hands[responder], strict=True)
        )


def test_ppo_updates_from_saved_trade_tokens():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 0, 0, 0, 0))
    obs = ObservationEncoder().encode(game, 0)
    obs["masks"]["type"][:] = False
    obs["masks"]["type"][ActionType.PROPOSE_TRADE] = True
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1, trade_decoder="gru"))
    batch = batch_observations([obs])
    with torch.no_grad():
        output = policy(batch)
        dist = output.distribution(batch["masks"])
        action = dist.sample()
        old_logp = dist.log_prob(action)
    assert game.is_legal(decode_action(action[0].tolist(), 0))
    arrays = {
        key: (
            {k: v.cpu().numpy() for k, v in value.items()}
            if isinstance(value, dict)
            else value.cpu().numpy()
        )
        for key, value in batch.items()
    }
    rollout = RolloutBatch(
        observations=arrays,
        actions=action.numpy(),
        log_probs=old_logp.numpy(),
        values=output.value.numpy(),
        rewards=np.array([1.0], dtype=np.float32),
        dones=np.array([True]),
        last_value=0.0,
    )
    learner = PPOLearner(policy, PPOConfig(epochs=1, minibatch_size=1))
    stats = learner.update([rollout])
    assert np.isfinite(stats["policy_loss"])
    assert policy.trade_decoder.output.weight.grad is not None


def test_one_card_decoder_probability_sums_to_one():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 1, 1, 1, 1))
    obs = ObservationEncoder().encode(game, 0)
    obs["masks"]["type"][:] = False
    obs["masks"]["type"][ActionType.PROPOSE_TRADE] = True
    policy = build_policy(
        ModelConfig(hidden_dim=32, mlp_layers=1, trade_decoder="gru", max_trade_cards_per_side=1)
    )
    batch = batch_observations([obs] * 20)
    actions = torch.full((20, 12), -1, dtype=torch.long)
    actions[:, 0] = ActionType.PROPOSE_TRADE
    for row, (g, r) in enumerate((g, r) for g in range(5) for r in range(5) if g != r):
        actions[row, 8:] = torch.tensor((g, 5, r, 6))
    scores = policy(batch).distribution(batch["masks"]).log_prob(actions)
    assert torch.logsumexp(scores, 0).item() == pytest.approx(0.0, abs=1e-5)


def test_multi_card_decoder_probability_sums_to_one():
    decoder = TradeDecoder(16, max_cards=2)
    sides = [counts for counts in product(range(3), repeat=5) if 1 <= sum(counts) <= 2]
    offers = [
        TradeOffer(give, receive)
        for give in sides
        for receive in sides
        if not any(g and r for g, r in zip(give, receive, strict=True))
    ]
    tokens = [offer_tokens(offer, 2) for offer in offers]
    padded = torch.tensor([t + (-1,) * (decoder.max_tokens - len(t)) for t in tokens])
    context = torch.randn(1, 16).expand(len(offers), -1)
    hand = torch.full((len(offers), 5), 2)
    limit = torch.full((len(offers),), 2)
    scores, _ = decoder.score(context, hand, limit, padded)
    assert torch.logsumexp(scores, 0).item() == pytest.approx(0.0, abs=1e-5)


def test_offer_observation_ignores_opponent_resource_composition():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 0, 0, 0, 0))
    game.state.hands[1] = [2, 0, 0, 0, 0]
    before = ObservationEncoder().encode(game, 0)
    game.state.hands[1] = [0, 2, 0, 0, 0]
    after = ObservationEncoder().encode(game, 0)
    for key in before:
        if key == "masks":
            assert all(np.array_equal(before[key][name], after[key][name]) for name in before[key])
        else:
            assert np.array_equal(before[key], after[key]), key


def test_pre_decoder_checkpoint_observation_upgrade_uses_own_hand():
    obs = {"hand": np.asarray([0.2, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)}
    _upgrade_trade_observation(obs, None)
    assert obs["own_hand_counts"].tolist() == [2, 0, 1, 0, 0]
    assert obs["trade_card_limit"].tolist() == [-1]


def test_gnn_uses_same_trade_decoder_contract():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 0, 0, 0, 0))
    obs = ObservationEncoder().encode(game, 0)
    obs["masks"]["type"][:] = False
    obs["masks"]["type"][ActionType.PROPOSE_TRADE] = True
    batch = batch_observations([obs, obs])
    policy = build_policy(
        ModelConfig(encoder="gnn", hidden_dim=32, gnn_layers=1, trade_decoder="gru")
    )
    distribution = policy(batch).distribution(batch["masks"])
    actions = distribution.sample()
    assert all(game.is_legal(decode_action(row, 0)) for row in actions.tolist())
    assert torch.isfinite(distribution.log_prob(actions)).all()
