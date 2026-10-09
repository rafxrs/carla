# Domestic trading: rules, implementation, and extension guide

Domestic trading is an optional negotiation between the active player and the other
three players, separate from bank and port trading. After rolling, the active player
proposes resource terms or invites offers. Each opponent responds once; the active
player confirms one exchange or cancels. Cards move only on confirmation.

The engine and Arcade GUI accept multi-card, multi-resource offers. The current
factored policy and baseline bots generate one-for-one offers. That smaller policy
action space is not the engine's rule limit. The engine leaves trading off by default;
the runnable MLP example `configs/experiments/domestic_trade.yaml` enables it, and
`carla gui` turns it on unless given `--no-trading`. `full_trade.yaml`
describes planned model work, not a completed training configuration.

## Offer meaning

`TradeOffer(give, receive)` holds two vectors in the project's five-resource order.
Each vector is written from the **offer maker's** perspective: what they give and
what they want. For an active player's proposal, `give` leaves the active player's
hand. For a responder's counteroffer, `give` leaves the responder's hand. The engine
reverses that perspective when the active player confirms a counteroffer.

For example, an active player offering two brick for one ore puts two brick in
`give` and one ore in `receive`. A responder countering with one grain for one
lumber puts grain in their `give` vector and lumber in their `receive` vector.
The confirmation dialog restates the selected terms from the active player's
perspective.

## Configurable experiment limits

| `EngineConfig` field | Default | Effect |
| --- | ---: | --- |
| `allow_domestic_trade` | `False` | Enables the protocol in the main phase (on in the GUI unless `--no-trading`). |
| `max_trade_proposals_per_turn` | 3 | Negotiation rounds started by the active player, including invitations and cancelled rounds. `None` removes the cap; `0` disables rounds. |
| `max_domestic_trades_per_turn` | 3 | Completed exchanges on the active player's turn. `None` removes the cap; `0` prevents new rounds. |
| `max_trade_cards_per_side` | 4 | Cards given or requested in one proposal or counteroffer, at least 1. `None` removes the cap. |

These are configurable experiment controls. The [CATAN rulebook](https://www.catan.com/sites/default/files/2025-03/CN3081%20CATAN%E2%80%93The%20Game%20Rulebook%20secure%20%281%29.pdf)
does not impose these numerical limits. Counters do not consume another round slot; all three other players get one response
per round, so an invitation cannot produce an unbounded response loop. The two
per-turn counters reset when the active player ends their turn. A round also requires
the `MAIN` phase, the active player as decision maker, and at least one card in their
hand.

## Negotiation protocol

| Phase | Who acts | Trade decisions | Result |
| --- | --- | --- | --- |
| `MAIN` | Active player | Propose terms or invite offers | Start a round and increment the negotiation count. |
| `TRADE_RESPONSE` | Other seats, one at a time in turn order | Accept an affordable initial proposal, reject, or counter from their own hand | Store one public answer per seat and advance to the next opponent. An invitation has no initial terms, so it can only be rejected or countered. |
| `TRADE_CONFIRM` | Active player | Confirm one available partner or cancel | Recheck both hands, exchange cards or cancel, and return to `MAIN`. |

An acceptance signals willingness, not an immediate transfer. A counteroffer replaces
that responder's acceptance and does not start another response loop. Exactly one
partner may trade in a round; responders cannot trade with each other on the active
player's turn. Cancellation is always available if nobody offers useful terms. The
bank is unaffected by domestic exchanges.

### Offer validity and information boundaries

- `give` and `receive` are five-element tuples of nonnegative integers, each with at
  least one card. Boolean counts and mutable-list vectors are rejected.
- One resource cannot appear on both sides of the same offer. Each side respects
  `max_trade_cards_per_side` unless it is `None`.
- The offer maker must hold everything they promise. The engine does not reject a
  proposal merely because an opponent lacks requested cards; doing so would expose
  a hidden hand. Acceptance checks only the responder's own hand.
- Hands cannot change during a negotiation, so these checks still hold when the
  active player concludes; taking a counteroffer also requires that player to hold
  the cards it asks for. `Game.is_legal()` checks any submitted offer with
  `rules.legal_trade_offer()`, including the action type and unused fields.
- Offers, answers, and executed terms are public. Other players' resource
  compositions and development-card identities remain hidden from observations
  and masks. Retrospective evaluation can inspect hidden hands in recorded games.

## How the implementation fits together

| Layer | Main files | Responsibility |
| --- | --- | --- |
| Data and configuration | `engine/actions.py`, `engine/config.py`, `engine/state.py` | Defines offers, action kinds, phases, negotiation state, limits, and JSON-compatible snapshots. |
| Rules and transitions | `engine/rules.py`, `engine/game.py` | Lists baseline legal actions, validates arbitrary submitted offers, enforces hand and phase checks, transfers cards, and clears round state. |
| Events and replay | `engine/events.py`, `engine/text.py`, game records | Publishes invitations, proposals, responses, counters, executions, and cancellations, with readable logs and exact replay. |
| Environment and model | `envs/action_space.py`, `envs/observation.py`, `models/` | Adds trade action types and masks, public relative-seat context, and one-for-one policy heads. |
| Human and bot play | `gui/controller.py`, `gui/trade.py`, `gui/session.py`, `agents/` | Builds offers from hand cards in Arcade, shows every offer and answer as cards, and supplies baseline decisions. |
| Evaluation | `eval/metrics.py`, `cli.py` | Reports invitations, counters, feasibility, acceptance, and execution from recorded games and tournaments. |

`GameState` stores the pending offer (`None` for an invitation), acceptances,
counteroffers, response progress, and per-turn counts. Snapshots and serialized state
preserve these fields. Public trade events appear in records, so saved games can be
replayed to the same state. `current_player` remains the turn owner while
`decision_player` moves through the opponents during responses. In `rules.py`, each
trade action has its own handler, like every other action; `_open_negotiation`,
`_next_answer` and `_close_negotiation` move a round between the phases.

Without domestic trade, `GameState.to_dict()` leaves out the negotiation fields and
`EngineConfig.to_dict()` the limits added with domestic trade. Such games serialize,
and hash, exactly as before domestic trade existed, so records saved earlier still
replay.

`legal_actions()` enumerates the fixed one-for-one subset that the current policy
can represent. `Game.is_legal()` additionally accepts any offer that
`legal_trade_offer()` allows: a valid offer (`valid_trade_offer()`) that opens a
negotiation or answers one. An engine or GUI offer can therefore be legal without
appearing in the enumerable policy list. `build_masks()` and `encode_action()`
explicitly reject multi-card offers instead of dropping extra cards.

The `trade_context` observation encodes public pending terms, acceptances,
counteroffers by seat relative to the observer, and normalized remaining-limit
indicators. Trade events also carry public terms. An opponent's resource composition
does not enter these features or the legal-action mask. An unlimited (`None`)
per-turn limit has a remaining-limit indicator of `1`.

## Evaluation and interpretation

`trade_metrics` requires recorded games. It returns proposal, invitation, and
counteroffer counts, and these rates:

| Metric | Numerator / denominator |
| --- | --- |
| Proposal feasibility | Initial proposals for which at least one opponent held all requested cards / initial proposals. |
| Acceptance | Initial proposals with at least one acceptance / initial proposals. |
| Counter feasibility | Counteroffers whose requested cards were in the active player's hand / counteroffers. |
| Execution | Completed exchanges / all rounds, including invitations. |

Invitation rounds have no initial proposal, so they are excluded from proposal
feasibility and acceptance. Feasibility uses hidden hands **after the fact** only;
it is never an acting-player feature. A zero denominator gives a zero rate. These
measures describe opportunities and behavior, not trade quality or win-rate gains.

## Arcade flow and how to run it

Trades start from the hand. Clicking a resource card opens the
trade panel above the hand and moves that card into **You give**; each further click
adds another. Clicking one of the five resources on the right adds it to **You get**,
and clicking a card in the panel takes it back. One offer then goes to the
**bank** (each resource given in a multiple of its rate, one card per batch, played
as consecutive maritime trades) or to the **players**. **Ask for offers** starts a
round without terms. `T` opens an empty offer and `Esc` closes it.

Offers and answers show their cards. A responder sees the proposal under **You give**
and **You get** and can decline, accept, or counter; clicking a hand card also starts
a counteroffer. While the others answer, the proposer sees each seat's status, then
picks a deal with its **Trade** button or cancels. Spectators see the same table
read-only.

```bash
carla gui --play                       # player trading on, default limits
carla gui --play --no-trading          # bank and port trades only
carla gui --play --set engine.max_trade_cards_per_side=2
```

With `--config`, the GUI takes the limits from the file; whether players trade at
all is set by `--no-trading`, not by the file.

The README has simulation and tournament commands for non-GUI runs.

## Building on this system

1. **Variable-length neural offers.** Add a decoder that chooses legal `give` and
   `receive` count vectors using the acting hand and public state. Keep
   `legal_trade_offer()` as the final engine check. Update model sampling, action
   encoding, and tests together; the current `(give resource, receive resource)`
   mask cannot represent extra cards without losing information.
2. **Better trading agents.** Use the existing response and confirmation phases,
   public terms, and own-hand checks. Evaluate whether terms improve a build plan
   or expected outcome. The baseline heuristic uses simple resource deficits and
   is not an optimized negotiator.
3. **Belief and PPO work.** Infer opponents' holdings from public events. Keep
   inferred state separate from ground-truth auxiliary targets, and do not feed
   hidden hands into policy inputs. The MLP example runs on one-for-one terms;
   PPO updates, the opponent pool, belief module, and advanced encoders are
   separate project work.
4. **New rules or limits.** Change typed `EngineConfig`, the initiation checks,
   and the offer validator together, then expose the setting in an experiment
   YAML. Preserve the one-response-per-seat bound and the hand checks if the
   protocol changes. Label experiment controls as such.
5. **New measurements or UI choices.** Derive outcome metrics from recorded
   games and public events; keep private-hand analysis retrospective. Use the
   existing picker and theme helpers for UI additions.

## Verification and present boundary

The complete local suite passed with 604 tests and `CARLA_REQUIRE_GUI=1`, so
Arcade window tests fail instead of silently skipping when a display is absent.
Targeted tests cover multi-card terms, limits, hidden information, action
encoding, replay, metrics, bots, and the GUI. A 32-seed stress audit covered
1,704 proposals, 4,816 counters, 55 invitations, and 569 completed exchanges;
resource invariants held after each decision and recorded games replayed exactly.
Ruff, Black, and patch checks passed locally. Remote CI awaits a push.

The engine and human GUI can make multi-card offers, while neural policy heads
represent one-for-one terms. Gymnasium's generic full `check_env` probe may reuse
an action after resetting to a different board where that action is illegal;
CARLA intentionally rejects illegal actions. Its passive API checks, legal-action
sampling, and seeded full episodes passed.
