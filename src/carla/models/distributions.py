"""Masked factored action distribution: the contract between policy outputs and env masks.

A policy emits one logit tensor per mask key (`carla.envs.action_space.MASK_SHAPES`, plus the
batch dimension). Two-dimensional masks (robber, year_of_plenty, maritime) are sampled as joint
categoricals, so dependent heads (victim given tile, receive given give) are exact. A decision
is: type -> its sub-decision -> (for development cards) the card's sub-decision.
"""

import torch
from torch import Tensor

from carla.engine.actions import ActionType
from carla.engine.constants import DevCard
from carla.envs.action_space import HEAD_INDEX, HEAD_NAMES, MASK_SHAPES
from carla.envs.trade_tokens import PAD
from carla.models.trade_decoder import TradeDecoder

_T = ActionType
# sub-decision key -> (selecting type, selecting dev card or None, heads written)
_SUBS: dict[str, tuple[int, int | None, tuple[str, ...]]] = {
    "settlement": (_T.BUILD_SETTLEMENT, None, ("corner",)),
    "city": (_T.BUILD_CITY, None, ("corner",)),
    "road": (_T.BUILD_ROAD, None, ("edge",)),
    "robber": (_T.MOVE_ROBBER, None, ("tile", "player")),
    "dev": (_T.PLAY_DEV_CARD, None, ("dev",)),
    "year_of_plenty": (_T.PLAY_DEV_CARD, DevCard.YEAR_OF_PLENTY, ("resource", "resource2")),
    "monopoly": (_T.PLAY_DEV_CARD, DevCard.MONOPOLY, ("resource",)),
    "discard": (_T.DISCARD, None, ("resource",)),
    "maritime": (_T.MARITIME_TRADE, None, ("resource", "resource2")),
    "partner": (_T.CONFIRM_TRADE, None, ("player",)),
    "trade": (_T.PROPOSE_TRADE, None, ("resource", "resource2")),
}


class FactoredDistribution:
    """Distribution over complete decisions, as head-index tensors (B, len(HEAD_NAMES))."""

    def __init__(self, logits: dict[str, Tensor], masks: dict[str, Tensor]) -> None:
        self.batch = logits["type"].shape[0]
        self.logp: dict[str, Tensor] = {}
        for key, shape in MASK_SHAPES.items():
            flat_logits = logits[key].reshape(self.batch, -1)
            if flat_logits.shape[1] != _numel(shape):
                raise ValueError(f"logits[{key!r}] must have shape (batch, *{shape})")
            mask = masks[key].reshape(self.batch, -1).bool()
            # Unavailable sub-decisions get a dummy uniform mask: their probability is 0 anyway.
            mask = mask | ~mask.any(dim=1, keepdim=True)
            self.logp[key] = torch.log_softmax(flat_logits.masked_fill(~mask, -torch.inf), dim=1)

    def sample(self, deterministic: bool = False) -> Tensor:
        picks = {
            key: (lp.argmax(1) if deterministic else torch.multinomial(lp.exp(), 1).squeeze(1))
            for key, lp in self.logp.items()
        }
        heads = torch.full(
            (self.batch, len(HEAD_NAMES)), -1, dtype=torch.long, device=picks["type"].device
        )
        heads[:, 0] = picks["type"]
        dev = picks["dev"]
        for key, (action_type, card, head_names) in _SUBS.items():
            chosen = picks["type"] == action_type
            if card is not None:
                chosen &= dev == card
            values = _unflatten(picks[key], key)
            for name, value in zip(head_names, values, strict=True):
                col = HEAD_INDEX[name]
                heads[:, col] = torch.where(chosen, value, heads[:, col])
        return heads

    def log_prob(self, heads: Tensor, *, include_trade: bool = True) -> Tensor:
        heads = heads.long()
        action_type, dev = heads[:, 0], heads[:, HEAD_INDEX["dev"]]
        total = _gather(self.logp["type"], action_type)
        for key, (t, card, head_names) in _SUBS.items():
            if key == "trade" and not include_trade:
                continue
            chosen = action_type == t
            if card is not None:
                chosen &= dev == card
            index = _flatten([heads[:, HEAD_INDEX[n]].clamp(min=0) for n in head_names], key)
            total = total + torch.where(chosen, _gather(self.logp[key], index), 0.0)
        return total

    def entropy(self) -> Tensor:
        """Exact entropy of the baseline factored distribution."""

        def h(key: str) -> Tensor:
            lp = self.logp[key]
            safe_lp = torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))
            return -(lp.exp() * safe_lp).sum(1)

        p_type, p_dev = self.logp["type"].exp(), self.logp["dev"].exp()
        dev_h = h("dev")
        dev_h = dev_h + p_dev[:, DevCard.YEAR_OF_PLENTY] * h("year_of_plenty")
        dev_h = dev_h + p_dev[:, DevCard.MONOPOLY] * h("monopoly")
        total = h("type")
        for key, (t, card, _) in _SUBS.items():
            if card is None:
                total = total + p_type[:, t] * (dev_h if key == "dev" else h(key))
        return total


class TradeDistribution(FactoredDistribution):
    """Eight action heads plus padded offer tokens when the type is PROPOSE_TRADE.

    The old one-for-one subhead is ignored for offers. Scoring uses the decoder's
    conditional token probabilities, including for counteroffers.
    """

    def __init__(
        self,
        logits: dict[str, Tensor],
        masks: dict[str, Tensor],
        decoder: TradeDecoder,
        context: Tensor,
        hand: Tensor,
        limit: Tensor,
    ) -> None:
        super().__init__(logits, masks)
        self.decoder = decoder
        self.context = context
        self.hand = hand.long()
        self.limit = limit.long().reshape(-1)
        self._conditional_entropy = context.new_zeros(self.batch)
        self._entropy_ready = False

    def sample(self, deterministic: bool = False) -> Tensor:
        heads = super().sample(deterministic)
        tokens = torch.full(
            (self.batch, self.decoder.max_tokens), PAD, dtype=torch.long, device=heads.device
        )
        selected = heads[:, 0] == _T.PROPOSE_TRADE
        conditional = self.context.new_zeros(self.batch)
        if selected.any():
            # Offer terms live in the tokens; clear the unused one-for-one head values.
            heads[selected, HEAD_INDEX["resource"]] = PAD
            heads[selected, HEAD_INDEX["resource2"]] = PAD
            sampled, token_entropy = self.decoder.sample(
                self.context[selected], self.hand[selected], self.limit[selected], deterministic
            )
            tokens[selected] = sampled
            conditional = conditional.index_copy(
                0, selected.nonzero(as_tuple=True)[0], token_entropy
            )
        self._conditional_entropy = conditional
        self._entropy_ready = True
        actions = torch.cat((heads, tokens), dim=1)
        # Remember the sampled offers' log-probabilities: scoring them again is redundant.
        token_logp = self.context.new_zeros(self.batch)
        if selected.any():
            token_logp = token_logp.index_copy(
                0, selected.nonzero(as_tuple=True)[0], self.decoder.last_sample_logp
            )
        self._sampled = (actions, token_logp)
        return actions

    def log_prob(self, actions: Tensor) -> Tensor:
        if actions.shape != (self.batch, len(HEAD_NAMES) + self.decoder.max_tokens):
            raise ValueError("incorrect structured action shape")
        heads = actions[:, : len(HEAD_NAMES)]
        selected = heads[:, 0] == _T.PROPOSE_TRADE
        total = super().log_prob(heads, include_trade=False)
        cached = getattr(self, "_sampled", None)
        if cached is not None and cached[0] is actions:
            return total + cached[1]
        if selected.any():
            # Add the exact sequence probability; the one-for-one subhead is unused.
            token_logp, token_entropy = self.decoder.score(
                self.context[selected],
                self.hand[selected],
                self.limit[selected],
                actions[selected, len(HEAD_NAMES) :],
            )
            replacement = total.new_zeros(self.batch).index_copy(
                0, selected.nonzero(as_tuple=True)[0], token_logp
            )
            conditional = total.new_zeros(self.batch).index_copy(
                0, selected.nonzero(as_tuple=True)[0], token_entropy
            )
            total = total + replacement
            self._conditional_entropy = conditional
        else:
            self._conditional_entropy = total.new_zeros(self.batch)
        self._entropy_ready = True
        return total

    def entropy(self) -> Tensor:
        """Exact head entropy plus sampled token entropy along offer prefixes."""
        return self.head_entropy() + self.token_entropy()

    def head_entropy(self) -> Tensor:
        """Exact entropy of the action heads, without the unused one-for-one trade subhead."""
        trade_lp = self.logp["trade"]
        safe = torch.where(torch.isfinite(trade_lp), trade_lp, 0.0)
        old_trade_entropy = -(trade_lp.exp() * safe).sum(1)
        return super().entropy() - self.logp["type"].exp()[:, _T.PROPOSE_TRADE] * old_trade_entropy

    def token_entropy(self) -> Tensor:
        """Offer-token entropy summed along stored (or freshly sampled) offer prefixes: a
        sampled regularizer, not the exact entropy over every possible offer; zero for rows
        that are not offers."""
        if not self._entropy_ready:
            self.sample()  # The token term uses a sampled prefix; draw one if none exists yet.
        return self._conditional_entropy


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _gather(logp: Tensor, index: Tensor) -> Tensor:
    return logp.gather(1, index.unsqueeze(1)).squeeze(1)


def _unflatten(index: Tensor, key: str) -> tuple[Tensor, ...]:
    shape = MASK_SHAPES[key]
    return (index,) if len(shape) == 1 else (index // shape[1], index % shape[1])


def _flatten(values: list[Tensor], key: str) -> Tensor:
    shape = MASK_SHAPES[key]
    return values[0] if len(shape) == 1 else values[0] * shape[1] + values[1]
