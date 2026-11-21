import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.utils.passive_env_checker import (
    env_reset_passive_checker,
    env_step_passive_checker,
)

from carla.config.schema import EnvConfig, ModelConfig
from carla.engine.actions import ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.state import Phase
from carla.envs import CatanEnv, ObservationEncoder, TradeActionWrapper, decode_action
from carla.envs.trade_tokens import offer_tokens
from carla.models.base import batch_observations
from carla.models.factory import build_policy
from tests.engine.scenario import give, post_setup_game


def test_wrapper_contains_decoder_offers():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 1, 0, 0, 0))
    obs = ObservationEncoder().encode(game, 0)
    batch = batch_observations([obs] * 16)
    batch["masks"]["type"][:] = False
    batch["masks"]["type"][:, ActionType.PROPOSE_TRADE] = True
    policy = build_policy(ModelConfig(hidden_dim=32, mlp_layers=1, trade_decoder="gru"))
    with torch.no_grad():
        actions = policy(batch).distribution(batch["masks"]).sample().numpy()

    wrapped = TradeActionWrapper(CatanEnv(game.config), max_cards=4)
    assert isinstance(wrapped.action_space, spaces.MultiDiscrete)
    assert all(wrapped.action_space.contains(action) for action in actions)
    assert not wrapped.action_space.contains(actions[0, :-1])
    assert not TradeActionWrapper(CatanEnv(game.config), max_cards=3).action_space.contains(
        actions[0]
    )
    invalid = actions[0].copy()
    invalid[-1] = 7
    assert not wrapped.action_space.contains(invalid)


def test_wrapper_samples_legal_actions_and_steps():
    wrapped = TradeActionWrapper(CatanEnv(EngineConfig(allow_domestic_trade=True, max_turns=8)))
    wrapped.action_space.seed(23)
    _, info = wrapped.reset(seed=13)
    for _ in range(500):
        action = wrapped.action_space.sample()
        assert wrapped.action_space.contains(action)
        assert decode_action(action, wrapped.unwrapped.learner) in info["legal_actions"]
        _, _, terminated, truncated, info = wrapped.step(action)
        if terminated or truncated:
            break
    else:
        pytest.fail("sampled episode did not finish")


def test_wrapper_samples_one_card_trade_as_tokens():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 0, 0, 0, 0))
    base = CatanEnv(game.config)
    base.aec._game = game
    wrapped = TradeActionWrapper(base)
    wrapped.action_space.seed(17)
    offers = [
        action
        for _ in range(30)
        if (action := wrapped.action_space.sample())[0] == ActionType.PROPOSE_TRADE
    ]
    assert offers
    assert all(wrapped.action_space.contains(action) for action in offers)
    assert all(decode_action(action, 0) in game.legal_actions() for action in offers)


def test_wrapper_steps_a_multi_card_offer():
    config = EngineConfig(allow_domestic_trade=True)
    base = CatanEnv(config, EnvConfig(learner_seat=0))
    wrapped = TradeActionWrapper(base)
    wrapped.reset(seed=3)
    game = post_setup_game(config=config)
    give(game, 0, (2, 0, 0, 0, 0))
    base.aec._game = game
    offer = TradeOffer((2, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    tokens = offer_tokens(offer, 4)
    action = np.full(wrapped.action_space.shape, -1, dtype=np.int64)
    action[0] = ActionType.PROPOSE_TRADE
    action[8 : 8 + len(tokens)] = tokens
    assert wrapped.action_space.contains(action)
    wrapped.step(action)
    assert game.phase == Phase.TRADE_CONFIRM
    assert game.state.pending_trade == offer


def test_wrapper_passes_gymnasium_passive_checker():
    wrapped = TradeActionWrapper(CatanEnv(EngineConfig(allow_domestic_trade=True)))
    env_reset_passive_checker(wrapped)
    env_step_passive_checker(wrapped, wrapped.action_space.sample())


def test_wrapper_rejects_nonpositive_card_cap():
    with pytest.raises(ValueError, match="positive"):
        TradeActionWrapper(CatanEnv(), max_cards=0)
