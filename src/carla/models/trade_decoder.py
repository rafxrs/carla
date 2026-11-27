"""Masked GRU generating canonical domestic-trade offers."""

import torch
from torch import Tensor, nn

from carla.engine.constants import NUM_RESOURCES
from carla.envs.trade_tokens import PAD, STOP, SWITCH

START = STOP + 1


class TradeDecoder(nn.Module):
    """Generate sorted give/SWITCH/receive/STOP tokens from an actor's context."""

    def __init__(self, dim: int, max_cards: int = 4) -> None:
        super().__init__()
        if max_cards < 1:
            raise ValueError("decoder card cap must be positive")
        self.max_cards = max_cards
        self.max_tokens = 2 * max_cards + 2
        self.embedding = nn.Embedding(START + 1, dim)
        self.cell = nn.GRUCell(2 * dim, dim)
        self.initial = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, STOP + 1)

    def _mask(self, hand, limit, give, receive, stage, last):
        """Legal next tokens for each prefix; stage 0 gives and stage 1 requests."""
        cap = torch.where(limit > 0, limit.clamp(max=self.max_cards), self.max_cards)
        ids = torch.arange(NUM_RESOURCES, device=hand.device)[None, :]
        give_total, receive_total = give.sum(1), receive.sum(1)
        # Giving the last unused resource type would leave nothing distinct to request.
        other_type_left = (give == 0).sum(1, keepdim=True) - (give == 0).long()
        give_cards = (
            (give < hand)
            & (ids >= last[:, None])
            & (give_total < cap)[:, None]
            & (other_type_left > 0)
        )
        receive_cards = (give == 0) & (ids >= last[:, None]) & (receive_total < cap)[:, None]
        cards = torch.where(stage[:, None].bool(), receive_cards, give_cards)
        return torch.cat(
            (
                cards,
                ((stage == 0) & (give_total > 0))[:, None],
                ((stage == 1) & (receive_total > 0))[:, None],
            ),
            dim=1,
        )

    @staticmethod
    def _advance(token, give, receive, stage, last):
        """Update counts and the smallest allowed next resource ID."""
        card = token < NUM_RESOURCES
        increment = torch.nn.functional.one_hot(token.clamp(0, NUM_RESOURCES - 1), NUM_RESOURCES)
        give = give + increment * (card & (stage == 0))[:, None]
        receive = receive + increment * (card & (stage == 1))[:, None]
        last = torch.where(card, token, last)
        last = torch.where(token == SWITCH, 0, last)
        stage = torch.where(token == SWITCH, 1, stage)
        return give, receive, stage, last

    def _start(self, context):
        n = len(context)
        hidden = torch.tanh(self.initial(context))
        previous = torch.full((n,), START, dtype=torch.long, device=context.device)
        counts = torch.zeros((n, NUM_RESOURCES), dtype=torch.long, device=context.device)
        stage = torch.zeros(n, dtype=torch.long, device=context.device)
        return hidden, previous, counts, counts.clone(), stage, stage.clone()

    def _logits(self, hidden, previous, context, mask):
        hidden = self.cell(torch.cat((self.embedding(previous), context), dim=1), hidden)
        logp = torch.log_softmax(self.output(hidden).masked_fill(~mask, -torch.inf), dim=1)
        safe = torch.where(torch.isfinite(logp), logp, 0.0)
        return hidden, logp, -(logp.exp() * safe).sum(1)

    def sample(self, context: Tensor, hand: Tensor, limit: Tensor, deterministic=False):
        """Return (padded tokens, sampled prefix entropy) for a batch of actors."""
        hidden, previous, give, receive, stage, last = self._start(context)
        tokens = torch.full(
            (len(context), self.max_tokens), PAD, dtype=torch.long, device=context.device
        )
        entropy_sum = context.new_zeros(len(context))
        self.last_sample_logp = context.new_zeros(len(context))
        active = torch.ones(len(context), dtype=torch.bool, device=context.device)
        for step in range(self.max_tokens):
            mask = self._mask(hand, limit, give, receive, stage, last)
            mask = torch.where(
                active[:, None],
                mask,
                torch.nn.functional.one_hot(torch.full_like(previous, STOP), STOP + 1).bool(),
            )
            hidden, logp, entropy = self._logits(hidden, previous, context, mask)
            token = logp.argmax(1) if deterministic else torch.multinomial(logp.exp(), 1).squeeze(1)
            tokens[:, step] = torch.where(active, token, PAD)
            entropy_sum = entropy_sum + torch.where(active, entropy, 0.0)
            self.last_sample_logp = self.last_sample_logp + torch.where(
                active, logp.gather(1, token[:, None]).squeeze(1), 0.0
            )
            give, receive, stage, last = self._advance(token, give, receive, stage, last)
            active = active & (token != STOP)
            previous = token
            if not active.any():
                break
        if active.any():
            raise RuntimeError("trade decoder exceeded its token bound")
        return tokens, entropy_sum

    def score(self, context: Tensor, hand: Tensor, limit: Tensor, tokens: Tensor):
        """Teacher-force stored tokens; return exact sequence log probability and prefix entropy."""
        hidden, previous, give, receive, stage, last = self._start(context)
        logp_sum = context.new_zeros(len(context))
        entropy_sum = logp_sum.clone()
        done = torch.zeros(len(context), dtype=torch.bool, device=context.device)
        if tokens.shape != (len(context), self.max_tokens):
            raise ValueError("incorrect trade token shape")
        if ((tokens < PAD) | (tokens > STOP)).any():
            raise ValueError("trade token outside the vocabulary")
        for step in range(self.max_tokens):
            token = tokens[:, step].long()
            active = ~done
            if ((token == PAD) != done).any():
                raise ValueError("trade tokens must stop exactly once before padding")
            mask = self._mask(hand, limit, give, receive, stage, last)
            selected = torch.where(active, token, STOP)
            if (active & ~mask.gather(1, selected[:, None]).squeeze(1)).any():
                raise ValueError("illegal or noncanonical trade token")
            # Finished rows need one valid dummy token so log_softmax stays finite.
            mask = torch.where(
                active[:, None],
                mask,
                torch.nn.functional.one_hot(torch.full_like(previous, STOP), STOP + 1).bool(),
            )
            hidden, logp, entropy = self._logits(hidden, previous, context, mask)
            logp_sum = logp_sum + torch.where(
                active, logp.gather(1, selected[:, None]).squeeze(1), 0.0
            )
            entropy_sum = entropy_sum + torch.where(active, entropy, 0.0)
            give, receive, stage, last = self._advance(selected, give, receive, stage, last)
            done = done | (token == STOP)
            previous = selected
        if not done.all():
            raise ValueError("trade tokens lack STOP")
        return logp_sum, entropy_sum
