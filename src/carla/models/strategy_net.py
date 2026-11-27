"""AlphaStar-style policy for Catan (`model.encoder: strategy`).

Encoders (feed-forward, batched over all rows of an unroll):
- board: the GNN of `carla.models.gnn` over corners, edges and tiles, with the planning
  features (`env.strategic_features`) added to corners and edges;
- seats: each relative seat's public stats, trade response, counteroffer and planning
  features, mixed by self-attention (a small transformer, like AlphaStar's entity encoder);
  the per-seat embeddings are kept for the robber-victim and trade-partner pointers;
- development cards (set attention) and scalars (hand, globals, own planning features).

Memory: the optional GRU belief folds the public event stream into a belief over opponents'
cards; an LSTM core (`model.core: lstm`) integrates the encoded observation over the seat's
decisions. Both states form one flat recurrent state: [belief | h_1..h_L | c_1..c_L].

Conditioning: the strategy statistic z (`carla.envs.strategy`) is embedded after the core and
conditions every head through a context MLP and a sigmoid gate (AlphaStar's GLU gated by
`scalar_context`), the trade decoder (its context is the gated context) and both critics. The
selector head reads only the core output and scalars, never z, so choosing a strategy and
acting under it are separate decisions. Spatial heads are pointers over corner, edge, tile and
seat embeddings, so spatial and opponent-specific information reaches the action choice.

Critics: the outcome value and the strategy pseudo-reward value. With `model.privileged_value`
they also read the opponents' hidden cards (training only, as AlphaStar's baselines read the
opponent's observation); policy logits never see them.
"""

import math

import torch
from torch import Tensor, nn

from carla.config.schema import ModelConfig
from carla.engine.constants import NUM_PLAYERS, NUM_RESOURCES
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, NUM_TILES
from carla.envs.action_space import MASK_SHAPES
from carla.envs.observation import CORNER_FEATURES, EDGE_FEATURES, EVENT_FEATURES, TILE_FEATURES
from carla.envs.strategy import STRATEGY_KEY, Z_DIM
from carla.envs.strategy_features import (
    CORNER_STRATEGY_FEATURES,
    EDGE_STRATEGY_FEATURES,
    OWN_STRATEGY_FEATURES,
    SEAT_STRATEGY_FEATURES,
)
from carla.models.base import STATE_KEY, TARGET_KEY, PolicyNetwork, PolicyOutput
from carla.models.belief import TARGET_SHAPES, BeliefModule, belief_loss
from carla.models.gnn import (
    DEV_ELEMENT_FEATURES,
    SEAT_ELEMENT_FEATURES,
    SET_FLAT_SIZE,
    MessagePassingLayer,
    SpatialHead,
    _dev_elements,
    _seat_elements,
    _set_flat,
)
from carla.models.graph import board_graph
from carla.models.set_attention import SetAttentionPool
from carla.models.trade_decoder import TradeDecoder

_POINTER_HEADS = ("settlement", "city", "road", "robber", "partner")
PRIVILEGED_SIZE = sum(math.prod(shape) for shape in TARGET_SHAPES.values())


class PairHead(nn.Module):
    """Joint (tile, victim seat) pointer: tile and seat embeddings meet the context."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.tile = nn.Linear(hidden, hidden)
        self.seat = nn.Linear(hidden, hidden, bias=False)
        self.context = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1)

    def forward(self, tiles: Tensor, seats: Tensor, context: Tensor) -> Tensor:
        joint = (
            self.tile(tiles)[:, :, None, :]
            + self.seat(seats)[:, None, :, :]
            + self.context(context)[:, None, None, :]
        )
        return self.score(torch.tanh(joint)).squeeze(-1)  # (B, tiles, seats)


class StrategyNet(PolicyNetwork):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        if hidden <= 0 or config.gnn_layers <= 0 or hidden % config.attention_heads:
            raise ValueError("strategy net needs positive sizes divisible by attention_heads")
        self.config = config
        self.corner_input = nn.Sequential(
            nn.Linear(CORNER_FEATURES + CORNER_STRATEGY_FEATURES, hidden), nn.ReLU()
        )
        self.edge_input = nn.Sequential(
            nn.Linear(EDGE_FEATURES + EDGE_STRATEGY_FEATURES, hidden), nn.ReLU()
        )
        self.tile_input = nn.Sequential(nn.Linear(TILE_FEATURES, hidden), nn.ReLU())
        canonical = board_graph()
        endpoints = canonical["edge_corner"][1].reshape(NUM_EDGES, 2).T
        tile, tile_corner = canonical["tile_corner"]
        source = torch.cat((endpoints[0], endpoints[1]))
        target = torch.cat((endpoints[1], endpoints[0]))
        graph = {
            "edge_u": endpoints[0],
            "edge_v": endpoints[1],
            "source": source,
            "target": target,
            "link_edge": torch.arange(NUM_EDGES).repeat(2),
            "tile": tile,
            "tile_corner": tile_corner,
            "corner_degree": torch.bincount(target, minlength=NUM_CORNERS).float().clamp_min(1),
            "corner_tile_degree": torch.bincount(tile_corner, minlength=NUM_CORNERS)
            .float()
            .clamp_min(1),
            "tile_degree": torch.bincount(tile, minlength=NUM_TILES).float().clamp_min(1),
        }
        self._graph_keys = tuple(graph)
        for key, tensor in graph.items():
            self.register_buffer(key, tensor)
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden, True) for _ in range(config.gnn_layers)
        )

        self.seat_input = nn.Sequential(
            nn.Linear(SEAT_ELEMENT_FEATURES + SEAT_STRATEGY_FEATURES, hidden), nn.ReLU()
        )
        layer = nn.TransformerEncoderLayer(
            hidden,
            config.attention_heads,
            dim_feedforward=2 * hidden,
            dropout=0.0,
            batch_first=True,
        )
        self.seat_transformer = (
            nn.TransformerEncoder(layer, config.seat_attention_layers, enable_nested_tensor=False)
            if config.seat_attention_layers > 0
            else None
        )
        self.seat_pool = SetAttentionPool(hidden, config.attention_heads)
        self.dev_input = nn.Sequential(nn.Linear(DEV_ELEMENT_FEATURES, hidden), nn.ReLU())
        self.dev_pool = SetAttentionPool(hidden, config.attention_heads)
        self.scalar_input = nn.Sequential(
            nn.Linear(SET_FLAT_SIZE + OWN_STRATEGY_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.belief = (
            BeliefModule(EVENT_FEATURES, config.belief_dim, config.belief)
            if config.belief != "none"
            else None
        )
        self.belief_dim = config.belief_dim if self.belief is not None else 0
        self.belief_loss_coef = config.belief_loss_coef
        embed_inputs = 6 * hidden + self.belief_dim
        self.embed = nn.Sequential(nn.Linear(embed_inputs, hidden), nn.ReLU())
        self.core_layers = config.core_layers if config.core == "lstm" else 0
        self.core_dim = config.core_dim
        if self.core_layers:
            self.cells = nn.ModuleList(
                nn.LSTMCell(hidden if i == 0 else self.core_dim, self.core_dim)
                for i in range(self.core_layers)
            )
            core_out = self.core_dim
        else:
            self.cells = nn.ModuleList()
            core_out = hidden
        self.state_dim = self.belief_dim + 2 * self.core_layers * self.core_dim

        self.uses_strategy = config.strategy_conditioning
        z_dim = config.strategy_dim if self.uses_strategy else 0
        self.z_embed = (
            nn.Sequential(nn.Linear(Z_DIM, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim), nn.ReLU())
            if self.uses_strategy
            else None
        )
        self.context = nn.Sequential(
            nn.Linear(core_out + z_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.gate = nn.Linear(z_dim + hidden, hidden)
        self.pointer = nn.ModuleDict(
            {key: SpatialHead(hidden) for key in ("settlement", "city", "road", "partner")}
        )
        self.robber = PairHead(hidden)
        self.heads = nn.ModuleDict(
            {
                f"logits_{key}": nn.Linear(hidden, math.prod(shape))
                for key, shape in MASK_SHAPES.items()
                if key not in _POINTER_HEADS
            }
        )
        self.privileged = config.privileged_value
        value_inputs = hidden + (hidden // 2 if self.privileged else 0)
        self.privileged_input = (
            nn.Sequential(nn.Linear(PRIVILEGED_SIZE, hidden // 2), nn.ReLU())
            if self.privileged
            else None
        )
        self.value_trunk = nn.Sequential(nn.Linear(value_inputs, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.strategy_value = nn.Linear(hidden, 1) if self.uses_strategy else None
        self.num_strategies = config.num_strategies if config.strategy_selector else 0
        self.selector = (
            nn.Sequential(
                nn.Linear(core_out + hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.num_strategies + 1),
            )
            if self.num_strategies
            else None
        )
        self.trade_decoder = (
            TradeDecoder(hidden, config.max_trade_cards_per_side)
            if config.trade_decoder == "gru"
            else None
        )

    # ------------------------------------------------------------------ encoders

    def _encode(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = obs["globals"].shape[0]
        graph = {key: getattr(self, key) for key in self._graph_keys}
        corners = self.corner_input(
            torch.cat((obs["corners"].float(), obs["corner_strategy"].float()), dim=-1)
        )
        edges = self.edge_input(
            torch.cat((obs["edges"].float(), obs["edge_strategy"].float()), dim=-1)
        )
        tiles = self.tile_input(obs["tiles"].float())
        for layer in self.layers:
            corners, edges, tiles = layer(corners, edges, tiles, graph)
        seats = self.seat_input(
            torch.cat((_seat_elements(obs), obs["seat_strategy"].float()), dim=-1)
        )
        if self.seat_transformer is not None:
            seats = self.seat_transformer(seats)
        all_seats = torch.ones(batch, NUM_PLAYERS, dtype=torch.bool, device=seats.device)
        dev_elements, dev_mask = _dev_elements(obs)
        scalar = self.scalar_input(torch.cat((_set_flat(obs), obs["own_strategy"].float()), dim=-1))
        pooled = torch.cat(
            (
                corners.mean(1),
                edges.mean(1),
                tiles.mean(1),
                self.seat_pool(seats, all_seats),
                self.dev_pool(self.dev_input(dev_elements), dev_mask),
                scalar,
            ),
            dim=-1,
        )
        return {
            "corners": corners,
            "edges": edges,
            "tiles": tiles,
            "seats": seats,
            "scalar": scalar,
            "pooled": pooled,
        }

    def _split(self, state: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        b = self.belief_dim
        belief = state[:, :b].contiguous()
        hs, cs = [], []
        for i in range(self.core_layers):
            start = b + 2 * i * self.core_dim
            hs.append(state[:, start : start + self.core_dim].contiguous())
            cs.append(state[:, start + self.core_dim : start + 2 * self.core_dim].contiguous())
        return belief, hs, cs

    def _join(self, belief: Tensor, hs: list[Tensor], cs: list[Tensor]) -> Tensor:
        parts = [belief]
        for h, c in zip(hs, cs, strict=True):
            parts.extend((h, c))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------ forward

    def forward(self, obs: dict[str, Tensor], state: Tensor | None = None) -> PolicyOutput:
        if state is None and self.recurrent and STATE_KEY in obs:
            state = obs[STATE_KEY].float()
        return self.unroll(obs, state, None, 1)

    def unroll(
        self, obs: dict[str, Tensor], state: Tensor | None, resets: Tensor | None, steps: int
    ) -> PolicyOutput:
        total = obs["globals"].shape[0]
        if total % steps:
            raise ValueError("unroll rows must be a multiple of the step count")
        width = total // steps
        device = obs["globals"].device
        encoded = self._encode(obs)
        if self.recurrent and state is None:
            state = torch.zeros(width, self.state_dim, device=device)
        core_rows, belief_rows = [], []
        for t in range(steps):
            rows = slice(t * width, (t + 1) * width)
            features = encoded["pooled"][rows]
            if self.recurrent:
                assert state is not None
                if resets is not None:
                    state = torch.where(resets[t][:, None], torch.zeros_like(state), state)
                belief_state, hs, cs = self._split(state)
            if self.belief is not None:
                belief_features, prediction, belief_state = self.belief(
                    obs["events"][rows], obs["event_mask"][rows], belief_state
                )
                belief_rows.append(prediction)
                features = torch.cat((features, belief_features), dim=-1)
            x = self.embed(features)
            if self.core_layers:
                new_h, new_c = [], []
                for cell, h, c in zip(self.cells, hs, cs, strict=True):
                    h, c = cell(x, (h, c))
                    new_h.append(h)
                    new_c.append(c)
                    x = h
                hs, cs = new_h, new_c
            if self.recurrent:
                state = self._join(
                    belief_state if self.belief is not None else x.new_zeros(width, 0), hs, cs
                )
            core_rows.append(x)
        core = torch.cat(core_rows) if steps > 1 else core_rows[0]
        belief = (
            {key: torch.cat([b[key] for b in belief_rows]) for key in belief_rows[0]}
            if belief_rows
            else None
        )
        return self._heads(obs, encoded, core, state, belief)

    def _heads(self, obs, encoded, core, state, belief) -> PolicyOutput:
        batch = core.shape[0]
        scalar = encoded["scalar"]
        if self.z_embed is not None:
            z = obs.get(STRATEGY_KEY)
            z = core.new_zeros(batch, Z_DIM) if z is None else z.float()
            z_features = self.z_embed(z)
            context = self.context(torch.cat((core, z_features, scalar), dim=-1))
            gate = torch.sigmoid(self.gate(torch.cat((z_features, scalar), dim=-1)))
        else:
            context = self.context(torch.cat((core, scalar), dim=-1))
            gate = torch.sigmoid(self.gate(scalar))
        gated = gate * context
        logits = {}
        for key, shape in MASK_SHAPES.items():
            if key in ("settlement", "city"):
                value = self.pointer[key](encoded["corners"], gated)
            elif key == "road":
                value = self.pointer[key](encoded["edges"], gated)
            elif key == "partner":
                value = self.pointer[key](encoded["seats"], gated)
            elif key == "robber":
                value = self.robber(encoded["tiles"], encoded["seats"], gated)
            else:
                value = self.heads[f"logits_{key}"](gated)
            logits[key] = value.reshape(batch, *shape)
        value_input = context
        if self.privileged_input is not None:
            targets = obs.get(TARGET_KEY)
            if targets is None:
                hidden = core.new_zeros(batch, PRIVILEGED_SIZE)
            else:
                hidden = torch.cat(
                    [targets[k].float().reshape(batch, -1) / NUM_RESOURCES for k in TARGET_SHAPES],
                    dim=-1,
                )
            value_input = torch.cat((context, self.privileged_input(hidden)), dim=-1)
        trunk = self.value_trunk(value_input)
        selector = (
            self.selector(torch.cat((core, scalar), dim=-1)) if self.selector is not None else None
        )
        return PolicyOutput(
            logits=logits,
            value=self.value(trunk).squeeze(1),
            state=state,
            belief=belief,
            trade_decoder=self.trade_decoder,
            trade_context=gated if self.trade_decoder is not None else None,
            trade_hand=obs["own_hand_counts"] if self.trade_decoder is not None else None,
            trade_limit=obs["trade_card_limit"] if self.trade_decoder is not None else None,
            strategy_value=(
                self.strategy_value(trunk).squeeze(1) if self.strategy_value is not None else None
            ),
            strategy_logits=selector,
        )

    def auxiliary_loss(self, output: PolicyOutput, obs: dict[str, Tensor]) -> Tensor | None:
        if output.belief is None or TARGET_KEY not in obs or not self.belief_loss_coef:
            return None
        targets: dict[str, Tensor] = obs[TARGET_KEY]  # type: ignore[assignment]
        return self.belief_loss_coef * belief_loss(output.belief, targets)
