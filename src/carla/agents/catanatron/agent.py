"""Original-code Catanatron bots on sanitized CARLA worlds; strict conversion."""

import json
import warnings

from carla.agents.api import AgentView
from carla.agents.base import Agent
from carla.agents.catanatron.codec import moves, reconstruct, representable
from carla.engine.actions import CANCEL_TRADE, REJECT_TRADE
from carla.engine.rng import derive_seed
from carla.engine.state import Phase
from carla.planning import sample_world
from carla.utils.seeding import fixed_hash_seed


class CatanatronAgent(Agent):
    def __init__(self, algorithm="F", depth=2):
        if algorithm not in ("F", "AB") or depth < 1:
            raise ValueError("expected F or AB with positive depth")
        self.algorithm, self.depth = algorithm, depth
        self.name = "catanatron-f" if algorithm == "F" else f"catanatron-ab{depth}"
        self.decisions = 0

    def reset(self, game, seat, seed):
        if fixed_hash_seed() is None:
            warnings.warn(
                "Catanatron decisions depend on Python string hashing: set PYTHONHASHSEED "
                "(e.g. 0) for reproducible games",
                RuntimeWarning,
                stacklevel=2,
            )
        from catanatron.models.player import Color
        from catanatron.players.minimax import AlphaBetaPlayer
        from catanatron.players.value import ValueFunctionPlayer

        self.seed, self.decisions = seed, 0
        self.bot = (
            ValueFunctionPlayer(tuple(Color)[seat])
            if self.algorithm == "F"
            else AlphaBetaPlayer(tuple(Color)[seat], AlphaBetaPlayer.Params(depth=self.depth))
        )

    def act(self, game, seat, legal):
        if game.state.phase == Phase.TRADE_RESPONSE:
            return REJECT_TRADE
        if game.state.phase == Phase.TRADE_CONFIRM:
            return CANCEL_TRADE
        actions = representable(legal)
        if len(actions) == 1:
            return actions[0]
        seed = derive_seed(self.seed, "decision", game.num_steps)
        self.decisions += 1
        world = sample_world(AgentView.from_game(game, seat), seed)
        native = reconstruct(world, seed)
        from carla.agents.catanatron.vendor.osp_game import Encoder

        def canonical(move):
            return json.dumps(move, sort_keys=True)

        by_move = {canonical(m): a for a, m in zip(actions, moves(world, actions), strict=True)}
        candidates = {}
        encoder = Encoder(native)
        for action in native.playable_actions:
            if game.state.phase == Phase.DISCARD:
                resource = {
                    "WOOD": "lumber",
                    "BRICK": "brick",
                    "SHEEP": "wool",
                    "WHEAT": "grain",
                    "ORE": "ore",
                }[action.value]
                move = ["discard", {resource: 1}]
            else:
                move = encoder.move(action)
            if (mapped := by_move.get(canonical(move))) is not None:
                candidates[action] = mapped
        if not candidates:
            raise ValueError(f"no common legal moves in {game.state.phase.name}")
        # AB ignores its passed actions and reads this property instead.
        native.playable_actions = list(candidates)
        chosen = self.bot.decide(native, native.playable_actions)
        if chosen not in candidates:
            raise ValueError(f"Catanatron selected an unrepresented action: {chosen}")
        return candidates[chosen]
