"""Domestic trade transitions, limits, conservation, replay, and private information."""

import random
from typing import Any

import pytest

from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.constants import NUM_RESOURCES, RESOURCE_CARDS_PER_TYPE
from carla.engine.events import EventKind
from carla.engine.game import Game, IllegalActionError
from carla.engine.record import GameRecord
from carla.engine.simulate import play_random_game
from carla.engine.state import GameState, Phase
from carla.eval.metrics import trade_metrics
from carla.eval.runner import GameResult
from tests.engine.scenario import (
    check_invariants,
    finish_setup,
    give,
    one_for_one,
    post_setup_game,
)


def offer(give: tuple[int, ...], receive: tuple[int, ...]) -> Action:
    return Action(ActionType.PROPOSE_TRADE, trade=TradeOffer(give, receive))


def trade_game(config: EngineConfig | None = None) -> Game:
    game = post_setup_game(config=config or EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 0, 0, 0, 0))
    give(game, 1, (0, 2, 0, 0, 0))
    give(game, 2, (0, 1, 0, 0, 0))
    return game


def recordable_trade_game() -> Game:
    """Reach a deterministic main phase through recorded setup and dice steps."""
    game = Game(EngineConfig(allow_domestic_trade=True), seed=0)
    finish_setup(game)
    game.step(A.ROLL, outcome=(2, 3))
    return game


def answer_all(game: Game, *answers: Action) -> None:
    for answer in answers:
        game.step(answer)
    assert game.phase == Phase.TRADE_CONFIRM and game.decision_player == 0


def test_offer_rules_and_hidden_hands_do_not_affect_proposer_legality():
    game = trade_game()
    valid = offer((1, 0, 0, 0, 0), (0, 0, 0, 0, 1))  # nobody has ore
    assert game.is_legal(valid)
    for invalid in (
        offer((0,) * 5, (0, 1, 0, 0, 0)),
        offer((1, 0, 0, 0, 0), (0,) * 5),
        offer((1, 0, 0, 0, 0), (1, 0, 0, 0, 0)),
        offer((3, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
        offer((1, 0, 0, 0, 0), (0, 5, 0, 0, 0)),
        offer((1, 0, 0, 0), (0, 1, 0, 0, 0)),
        Action(10, trade=valid.trade),  # type: ignore[arg-type]
        Action(ActionType.PROPOSE_TRADE, player=1, trade=valid.trade),
        Action(
            ActionType.PROPOSE_TRADE,
            trade=TradeOffer([1, 0, 0, 0, 0], (0, 1, 0, 0, 0)),  # type: ignore[arg-type]
        ),
    ):
        assert not game.is_legal(invalid)
        with pytest.raises(IllegalActionError):
            game.step(invalid)
    game.step(valid)
    assert game.phase == Phase.TRADE_RESPONSE
    counter = one_for_one(1, 0)
    assert counter is not None and game.is_legal(counter)
    assert not game.is_legal(counter._replace(dev=0))
    for _ in range(3):
        assert A.REJECT_TRADE in game.legal_actions()
        assert A.ACCEPT_TRADE not in game.legal_actions()
        game.step(A.REJECT_TRADE)
    assert game.legal_actions() == (A.CANCEL_TRADE,)


def test_responses_confirm_and_resource_conservation():
    game = trade_game()
    before = [game.state.bank[r] + sum(h[r] for h in game.state.hands) for r in range(5)]
    terms = TradeOffer((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    events = game.step(offer(*terms))
    assert events[0].kind == EventKind.TRADE_PROPOSED
    assert events[0].give == terms.give
    assert game.decision_player == 1
    answer_all(game, A.ACCEPT_TRADE, A.ACCEPT_TRADE, A.REJECT_TRADE)
    assert set(game.legal_actions()) == {
        A.CONFIRM_TRADE_WITH[1],
        A.CONFIRM_TRADE_WITH[2],
        A.CANCEL_TRADE,
    }
    events = game.step(A.CONFIRM_TRADE_WITH[2])
    assert events[0].kind == EventKind.TRADE_EXECUTED and events[0].target == 2
    assert game.state.hands[0] == [1, 1, 0, 0, 0]
    assert game.state.hands[2] == [1, 0, 0, 0, 0]
    assert game.state.hands[1] == [0, 2, 0, 0, 0]
    assert game.state.pending_trade is None and game.phase == Phase.MAIN
    assert game.state.domestic_trades_this_turn == 1
    assert [game.state.bank[r] + sum(h[r] for h in game.state.hands) for r in range(5)] == before
    assert before == [RESOURCE_CARDS_PER_TYPE] * NUM_RESOURCES


def test_limits_cancellation_and_turn_reset():
    game = trade_game(
        EngineConfig(
            allow_domestic_trade=True,
            max_trade_proposals_per_turn=3,
            max_domestic_trades_per_turn=1,
        )
    )
    proposal = one_for_one(0, 1)
    game.step(proposal)
    answer_all(game, A.ACCEPT_TRADE, A.REJECT_TRADE, A.REJECT_TRADE)
    game.step(A.CANCEL_TRADE)
    assert game.state.trade_proposals_this_turn == 1
    game.step(proposal)
    answer_all(game, A.ACCEPT_TRADE, A.REJECT_TRADE, A.REJECT_TRADE)
    game.step(A.CONFIRM_TRADE_WITH[1])
    assert game.state.trade_proposals_this_turn == 2
    assert game.state.domestic_trades_this_turn == 1
    assert not any(a.type == ActionType.PROPOSE_TRADE for a in game.legal_actions())
    game.step(A.END_TURN)
    assert game.state.trade_proposals_this_turn == game.state.domestic_trades_this_turn == 0

    limited = trade_game(
        EngineConfig(
            allow_domestic_trade=True,
            max_trade_proposals_per_turn=1,
            max_domestic_trades_per_turn=3,
        )
    )
    limited.step(proposal)
    answer_all(limited, A.REJECT_TRADE, A.REJECT_TRADE, A.REJECT_TRADE)
    limited.step(A.CANCEL_TRADE)
    assert not any(a.type == ActionType.PROPOSE_TRADE for a in limited.legal_actions())


def test_multi_card_offer_is_validated_and_exchanged_atomically():
    game = trade_game()
    give(game, 0, (0, 0, 1, 0, 0))
    proposal = offer((1, 0, 1, 0, 0), (0, 2, 0, 0, 0))
    assert game.is_legal(proposal)
    game.step(proposal)
    answer_all(game, A.ACCEPT_TRADE, A.REJECT_TRADE, A.REJECT_TRADE)
    game.step(A.CONFIRM_TRADE_WITH[1])
    assert game.state.hands[0] == [1, 2, 0, 0, 0]
    assert game.state.hands[1] == [1, 0, 1, 0, 0]


def test_responder_counteroffer_is_public_and_executes_with_active_player():
    game = trade_game()
    game.step(one_for_one(0, 4))  # ask for unavailable ore
    terms = TradeOffer((0, 1, 0, 0, 0), (2, 0, 0, 0, 0))
    counter = offer(*terms)
    assert counter not in game.legal_actions() and game.is_legal(counter)
    events = game.step(counter)
    assert events[0].kind == EventKind.TRADE_COUNTERED
    assert events[0].player == 1 and events[0].target == 0
    assert events[0].give == terms.give
    state_copy = game.state.from_dict(game.state.to_dict())
    assert state_copy.trade_counteroffers[1] == terms
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    assert A.CONFIRM_TRADE_WITH[1] in game.legal_actions()
    executed = game.step(A.CONFIRM_TRADE_WITH[1])[0]
    assert executed.kind == EventKind.TRADE_EXECUTED
    assert executed.give == terms.receive
    assert executed.receive == terms.give
    assert game.state.hands[0] == [0, 1, 0, 0, 0]
    assert game.state.hands[1] == [2, 1, 0, 0, 0]
    assert game.state.trade_counteroffers == [None] * 4


def test_counteroffer_replays_from_the_initial_position():
    game = recordable_trade_game()
    game.step(one_for_one(3, 1))  # seat 1 has no brick
    game.step(one_for_one(4, 2))  # seat 1 offers ore for seat 0's wool
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    game.step(A.CONFIRM_TRADE_WITH[1])
    record = GameRecord.from_game(game)
    assert record.replay().state_hash() == game.state_hash()


def test_counteroffer_legality_does_not_reveal_active_players_hand():
    game = trade_game()
    game.step(A.INVITE_TRADE)
    counter = offer((0, 1, 0, 0, 0), (0, 0, 0, 0, 1))
    assert game.is_legal(counter)  # the responder owns brick; the active player has no ore
    game.step(counter)
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    assert game.legal_actions() == (A.CANCEL_TRADE,)
    game.step(A.CANCEL_TRADE)
    assert game.state.trade_proposals_this_turn == 1


def test_invitation_accepts_an_unsolicited_offer_and_records_metrics():
    game = recordable_trade_game()
    game.step(A.INVITE_TRADE)
    assert game.state.pending_trade is None
    assert A.ACCEPT_TRADE not in game.legal_actions()
    game.step(one_for_one(4, 2))
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    game.step(A.CONFIRM_TRADE_WITH[1])
    record = GameRecord.from_game(game)
    result = GameResult(
        seed=0,
        agents=("a", "b", "c", "d"),
        winner=None,
        victory_points=tuple(game.victory_points(p) for p in range(4)),
        turns=game.turn,
        steps=game.num_steps,
        record=record,
    )
    assert trade_metrics([result]) == {
        "proposals": 0.0,
        "invitations": 1.0,
        "counteroffers": 1.0,
        "feasibility_rate": 0.0,
        "acceptance_rate": 0.0,
        "counter_feasibility_rate": 1.0,
        "execution_rate": 1.0,
    }
    assert record.replay().state_hash() == game.state_hash()
    assert "declines to trade with P0" in record.to_text()


def test_unlimited_trade_limits_are_configurable():
    game = trade_game(
        EngineConfig(
            allow_domestic_trade=True,
            max_trade_proposals_per_turn=None,
            max_domestic_trades_per_turn=None,
        )
    )
    for _ in range(5):
        game.step(A.INVITE_TRADE)
        for _ in range(3):
            game.step(A.REJECT_TRADE)
        game.step(A.CANCEL_TRADE)
    assert game.state.trade_proposals_this_turn == 5
    assert A.INVITE_TRADE in game.legal_actions()


@pytest.mark.parametrize("seed", range(4))
def test_seeded_games_keep_invariants_with_multicard_offers_and_replay(seed):
    rng = random.Random(seed)
    game = Game(EngineConfig(allow_domestic_trade=True, max_turns=20), seed)
    custom_moves = 0
    while not game.is_over:
        legal = game.legal_actions()
        action = rng.choice(legal)
        if game.phase in (Phase.MAIN, Phase.TRADE_RESPONSE) and rng.random() < 0.65:
            hand = game.state.hands[game.decision_player]
            available = [r for r, n in enumerate(hand) if n]
            if available and any(a.type == ActionType.PROPOSE_TRADE for a in legal):
                give_type = rng.choice(available)
                get_type = rng.choice([r for r in range(NUM_RESOURCES) if r != give_type])
                give, receive = [0] * NUM_RESOURCES, [0] * NUM_RESOURCES
                give[give_type] = rng.randint(1, min(4, hand[give_type]))
                receive[get_type] = rng.randint(1, 4)
                action = offer(tuple(give), tuple(receive))
                assert game.is_legal(action)
                custom_moves += 1
        game.step(action)
        check_invariants(game)
    assert custom_moves > 0
    assert GameRecord.from_game(game).replay().state_hash() == game.state_hash()


def test_snapshot_state_dict_and_exact_record_replay():
    game = Game(EngineConfig(allow_domestic_trade=True), seed=0)
    finish_setup(game)
    game.step(A.ROLL, outcome=(2, 3))
    giver = next(r for r, n in enumerate(game.state.hands[0]) if n)
    partner, wanted = next(
        (p, r) for p in range(1, 4) for r, n in enumerate(game.state.hands[p]) if n and r != giver
    )
    proposal = one_for_one(giver, wanted)
    game.step(proposal)
    snapshot = game.snapshot()
    state_copy = game.state.from_dict(game.state.to_dict())
    assert state_copy.pending_trade == proposal.trade
    for p in range(1, 4):
        game.step(A.ACCEPT_TRADE if p == partner else A.REJECT_TRADE)
    game.step(A.CONFIRM_TRADE_WITH[partner])
    expected_hash = game.state_hash()
    record = GameRecord.from_game(game)
    assert record.replay().state_hash() == expected_hash
    assert "trades" in record.to_text()
    result = GameResult(
        seed=0,
        agents=("a", "b", "c", "d"),
        winner=None,
        victory_points=tuple(game.victory_points(p) for p in range(4)),
        turns=game.turn,
        steps=game.num_steps,
        record=record,
    )
    assert trade_metrics([result]) == {
        "proposals": 1.0,
        "invitations": 0.0,
        "counteroffers": 0.0,
        "feasibility_rate": 1.0,
        "acceptance_rate": 1.0,
        "counter_feasibility_rate": 0.0,
        "execution_rate": 1.0,
    }
    game.restore(snapshot)
    assert game.phase == Phase.TRADE_RESPONSE and game.state.pending_trade == proposal.trade


def test_trade_config_limits_must_be_nonnegative():
    for field in (
        "max_trade_proposals_per_turn",
        "max_domestic_trades_per_turn",
        "max_trade_cards_per_side",
    ):
        limits: dict[str, Any] = {field: -1}
        with pytest.raises(ValueError, match="non-negative integer"):
            EngineConfig(**limits)


def test_games_without_domestic_trade_serialize_as_before_it_existed():
    record = GameRecord.from_game(play_random_game(5))
    assert record.hashes is not None  # computed by the engine before domestic trade was added:
    assert (len(record.hashes), record.hashes[0], record.hashes[100], record.hashes[-1]) == (
        609,
        "cce7db5bb979f437",
        "cc933945038f6622",
        "7e6eb11e67f22075",
    )
    plain = record.replay().state.to_dict()
    assert "trade_responses" not in plain and "max_trade_cards_per_side" not in plain["config"]
    assert GameState.from_dict(plain).to_dict() == plain
    trading = Game(EngineConfig(allow_domestic_trade=True)).state.to_dict()
    assert trading["trade_responses"] == 0 and trading["config"]["max_trade_cards_per_side"] == 4
