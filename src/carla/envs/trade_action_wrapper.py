"""Gymnasium action space for fixed-width, decoder-enabled policy actions."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carla.engine.actions import ActionType
from carla.envs.action_space import HEAD_INDEX, HEAD_NAMES, HEAD_SIZES, encode_action
from carla.envs.catan_env import CatanEnv
from carla.envs.trade_tokens import PAD, STOP, offer_tokens


class _TradeActionSpace(spaces.MultiDiscrete):
    """Eight action heads plus room for give/SWITCH/receive/STOP tokens."""

    def __init__(self, env: CatanEnv, max_cards: int) -> None:
        self.env = env
        self.max_cards = max_cards
        self.max_tokens = 2 * max_cards + 2
        # -1 is the padding value for unused heads and tokens.
        start = [0] + [-1] * (len(HEAD_NAMES) - 1 + self.max_tokens)
        nvec = (
            [HEAD_SIZES["type"]]
            + [HEAD_SIZES[name] + 1 for name in HEAD_NAMES[1:]]
            + [STOP + 2] * self.max_tokens
        )
        super().__init__(nvec, start=start)

    def sample(self, mask=None, probability=None):
        """Sample a legal current decision after reset."""
        if mask is not None or probability is not None:
            raise NotImplementedError("per-head Gymnasium masks cannot express conditional actions")
        if self.env.aec._game is None:
            return super().sample()
        game = self.env.game
        if game.is_over or game.decision_player != self.env.learner:
            return super().sample()
        legal = game.legal_actions()
        action = legal[int(self.np_random.integers(len(legal)))]
        heads = encode_action(action, self.env.learner)
        tokens = np.full(self.max_tokens, PAD, dtype=np.int64)
        if action.type == ActionType.PROPOSE_TRADE:
            assert action.trade is not None
            sequence = offer_tokens(action.trade, self.max_cards)
            tokens[: len(sequence)] = sequence
            heads[HEAD_INDEX["resource"]] = PAD
            heads[HEAD_INDEX["resource2"]] = PAD
        return np.concatenate((heads, tokens))


class TradeActionWrapper(gym.Wrapper):
    """Expose decoder actions as a standard fixed-width Gymnasium action space.

    Set `max_cards` to the policy's `model.max_trade_cards_per_side`. The engine
    still checks trade rules and the acting player's hand when `step()` runs.
    """

    def __init__(self, env: CatanEnv, max_cards: int = 4) -> None:
        if max_cards < 1:
            raise ValueError("decoder card cap must be positive")
        super().__init__(env)
        self.action_space = _TradeActionSpace(env, max_cards)
