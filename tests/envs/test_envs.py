import itertools

import numpy as np
import pytest
from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker

from carla.config.schema import EnvConfig
from carla.engine import actions as A
from carla.engine.config import EngineConfig
from carla.engine.game import Game, IllegalActionError
from carla.engine.rng import Rng
from carla.envs import (
    HEAD_NAMES,
    CatanAECEnv,
    CatanEnv,
    ObservationEncoder,
    build_masks,
    decode_action,
    encode_action,
    sample_heads,
)
from carla.envs.action_space import HEAD_INDEX, HEAD_SIZES, heads_used
from carla.envs.observation import TRADE_BASE_FEATURES
from tests.engine.scenario import give, one_for_one, post_setup_game


def all_mask_decisions(masks, actor):
    """Every action the masks allow, by brute force over the heads each type uses."""
    out = set()
    for t in np.flatnonzero(masks["type"]).tolist():
        for dev in range(5) if heads_used(t, 0) == ("dev",) else [-1]:
            free = [h for h in heads_used(t, dev) if h != "dev"]
            for values in itertools.product(*(range(HEAD_SIZES[h]) for h in free)):
                vector = np.full(len(HEAD_NAMES), -1)
                vector[0], vector[HEAD_INDEX["dev"]] = t, dev
                for h, v in zip(free, values, strict=True):
                    vector[HEAD_INDEX[h]] = v
                if mask_allows(masks, vector):
                    out.add(decode_action(vector, actor))
    return out


def mask_allows(masks, h) -> bool:
    t, corner, edge, tile, player, dev, r1, r2 = (int(x) for x in h)
    name = {2: "settlement", 3: "city"}.get(t)
    if name:
        return masks[name][corner]
    checks = {
        4: lambda: masks["road"][edge],
        7: lambda: masks["robber"][tile, player],
        8: lambda: masks["discard"][r1],
        9: lambda: masks["maritime"][r1, r2],
        10: lambda: masks["trade"][r1, r2],
        13: lambda: masks["partner"][player],
    }
    if t in checks:
        return bool(checks[t]())
    if t == 6:
        if not masks["dev"][dev]:
            return False
        if dev == 2:
            return bool(masks["year_of_plenty"][r1, r2])
        if dev == 3:
            return bool(masks["monopoly"][r1])
        return True
    return True


@pytest.mark.parametrize("seed", [0, 1])
def test_masks_describe_exactly_the_legal_actions(seed):
    game, rng = Game(seed=seed), Rng(seed)
    while not game.is_over:
        legal, actor = game.legal_actions(), game.decision_player
        masks = build_masks(legal, actor)
        if game.num_steps % 5 == 0:
            assert all_mask_decisions(masks, actor) == set(legal)
        for action in legal:
            assert decode_action(encode_action(action, actor), actor) == action
        game.step(rng.choice(legal))


def test_trade_masks_cover_the_baseline_offers_and_responses():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 0, 0, 0, 0))
    give(game, 1, (0, 1, 0, 0, 0))
    legal = game.legal_actions()
    masks = build_masks(legal, game.decision_player)
    assert all_mask_decisions(masks, game.decision_player) == set(legal)
    assert masks["trade"][0, 1] and not masks["trade"][0, 0]
    larger = A.Action(
        A.ActionType.PROPOSE_TRADE,
        trade=A.TradeOffer((1, 0, 0, 0, 0), (0, 2, 0, 0, 0)),
    )
    assert game.is_legal(larger)
    with pytest.raises(ValueError, match="one-for-one"):
        build_masks((larger,), 0)
    with pytest.raises(ValueError, match="one-for-one"):
        encode_action(larger, 0)
    game.step(one_for_one(0, 1))
    obs = ObservationEncoder().encode(game, 0)
    assert obs["trade_context"][0] == obs["trade_context"][6] == 0.25
    assert not obs["masks"]["type"].any()  # the proposer is waiting for responses
    responder_masks = build_masks(game.legal_actions(), game.decision_player)
    assert responder_masks["type"][A.ActionType.ACCEPT_TRADE]
    game.step(A.ACCEPT_TRADE)
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    confirm_masks = build_masks(game.legal_actions(), game.decision_player)
    assert all_mask_decisions(confirm_masks, game.decision_player) == set(game.legal_actions())


def test_invitation_and_counteroffer_masks_and_public_observation():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 0, 0, 0, 0))
    give(game, 1, (0, 1, 0, 0, 0))
    masks = build_masks(game.legal_actions(), 0)
    assert masks["type"][A.ActionType.INVITE_TRADE]
    game.step(A.INVITE_TRADE)
    legal = game.legal_actions()
    responder_masks = build_masks(legal, 1)
    assert responder_masks["type"][A.ActionType.PROPOSE_TRADE]
    assert not responder_masks["type"][A.ActionType.ACCEPT_TRADE]
    assert all_mask_decisions(responder_masks, 1) == set(legal)
    game.step(one_for_one(1, 0))
    obs = ObservationEncoder().encode(game, 0)
    start = TRADE_BASE_FEATURES + 2 * 5  # responder seat 1 relative to observer seat 0
    assert obs["trade_context"][start + 1] == 0.25
    assert obs["trade_context"][start + 5] == 0.25


def run_episode(env, seed, sampler_seed):
    rng = np.random.default_rng(sampler_seed)
    obs, info = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        assert env.observation_space.contains(obs)
        action = sample_heads(obs["masks"], rng)
        assert decode_action(action, env.learner) in info["legal_actions"]
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps, info, env.game.state_hash()


def test_gym_env_plays_full_episodes():
    env = CatanEnv()
    for seed in range(3):
        total, steps, info, _ = run_episode(env, seed, seed)
        assert steps > 10
        assert total == (1.0 if info["winner"] == env.learner else -1.0)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(len(HEAD_NAMES), dtype=int))


@pytest.mark.parametrize("allow_trade", [False, True])
def test_gym_action_space_samples_current_legal_actions(allow_trade):
    env = CatanEnv(EngineConfig(allow_domestic_trade=allow_trade, max_turns=12))
    _, info = env.reset(seed=13)
    while not env.game.is_over:
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        assert decode_action(action, env.learner) in info["legal_actions"]
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


@pytest.mark.parametrize("allow_trade", [False, True])
def test_gymnasium_passive_api_checker(allow_trade):
    env = CatanEnv(EngineConfig(allow_domestic_trade=allow_trade))
    env_reset_passive_checker(env)
    env_step_passive_checker(env, env.action_space.sample())


def test_gym_env_is_reproducible():
    a = run_episode(CatanEnv(), seed=7, sampler_seed=1)
    b = run_episode(CatanEnv(), seed=7, sampler_seed=1)
    assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3]


def test_truncation_and_shaping_rewards():
    env = CatanEnv(
        EngineConfig(max_turns=8),
        EnvConfig(learner_seat=2, vp_reward=0.5, truncation_reward=-0.25),
    )
    total, _, info, _ = run_episode(env, seed=3, sampler_seed=3)
    assert env.learner == 2 and info["winner"] is None and env.game.is_truncated
    assert total == pytest.approx(0.5 * env.game.victory_points(2, include_hidden=False) - 0.25)


def test_illegal_head_choice_raises():
    env = CatanEnv()
    obs, _ = env.reset(seed=0)
    illegal = np.flatnonzero(~obs["masks"]["type"])[0]
    with pytest.raises(IllegalActionError):
        env.step(np.array([illegal, 0, 0, 0, 0, 0, 0, 0]))


def test_aec_self_play_rewards_and_hidden_truth():
    env = CatanAECEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=5)
    totals = np.zeros(4)
    while True:
        obs = env.observe()
        result = env.step(sample_heads(obs["masks"], rng))
        totals += result.rewards
        if result.terminated or result.truncated:
            break
    winner = result.info["winner"]
    assert totals[winner] == 1.0 and sorted(totals) == [-1.0, -1.0, -1.0, 1.0]
    truth = env.hidden_truth(0)
    assert truth["resources"].shape == (3, 5) and truth["dev_cards"].shape == (3, 5)
    assert truth["resources"].sum() == sum(map(sum, env.game.state.hands[1:]))


def test_observation_does_not_leak_hidden_cards():
    game, rng = Game(seed=2), Rng(2)
    for _ in range(400):
        game.step(rng.choice(game.legal_actions()))
    encoder = ObservationEncoder()
    observer = game.decision_player
    before = encoder.encode(game, observer)
    s = game.state
    for p in range(4):
        if p != observer:
            s.hands[p] = s.hands[p][1:] + s.hands[p][:1]  # same size, other composition
            s.dev_hands[p] = s.dev_hands[p][::-1]
    game.set_state(s)
    after = encoder.encode(game, observer)
    for key in ("tiles", "corners", "edges", "players", "hand", "dev_cards", "events"):
        np.testing.assert_array_equal(before[key], after[key])
