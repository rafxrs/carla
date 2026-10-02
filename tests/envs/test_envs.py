import itertools

import numpy as np
import pytest

from carla.config.schema import EnvConfig
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
