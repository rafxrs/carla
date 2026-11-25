import pytest

from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.constants import Building, DevCard
from carla.engine.events import EventKind
from carla.engine.game import Game
from carla.engine.state import Phase
from carla.engine.topology import CORNER_NEIGHBORS
from carla.eval.diagnostics import DiagnosticRecorder, diagnose_game
from tests.engine.scenario import give, give_dev, post_setup_game, put_settlement


def trade_position():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (5, 0, 1, 1, 0))
    give(game, 1, (0, 3, 0, 0, 0))
    return game


def offer(give, receive):
    return Action(ActionType.PROPOSE_TRADE, trade=TradeOffer(tuple(give), tuple(receive)))


def respond(recorder, game, accepted=False):
    recorder.step(game, A.ACCEPT_TRADE if accepted else A.REJECT_TRADE)
    recorder.step(game, A.REJECT_TRADE)
    recorder.step(game, A.REJECT_TRADE)


def test_executed_trade_counts_both_players_and_cost_only_affordability():
    game = trade_position()
    put_settlement(game, 0, 0)
    recorder = DiagnosticRecorder()
    recorder.step(game, offer((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    respond(recorder, game, accepted=True)
    recorder.step(game, A.CONFIRM_TRADE_WITH[1])
    assert not any(action.type == ActionType.BUILD_SETTLEMENT for action in game.legal_actions())
    road = next(action for action in game.legal_actions() if action.type == ActionType.BUILD_ROAD)
    recorder.step(game, road)
    result = recorder.finish(game)
    learner, partner = result.players[:2]
    assert learner["completed_negotiations"] == learner["completed_trades_initiated"] == 1
    assert partner["completed_trades_as_partner"] == 1
    assert learner["negotiations_with_direct_acceptance"] == learner["opening_proposals"] == 1
    assert learner["completed_via_counteroffer"] == 0
    assert learner["trades_unlocking_settlement"] == 1  # Cost affordability, no legal corner.
    assert learner["trades_unlocking_road"] == 1
    assert learner["purchased_roads"] == learner["roads"] == 1 and learner["setup_roads"] == 0
    assert learner["road_legal_decisions"] == learner["construction_selected_when_possible"] == 1
    assert partner["acceptances"] == partner["required_trade_responses"] == 1
    assert result.trades[0]["give"] == result.trades[1]["receive"] == (1, 0, 0, 0, 0)
    assert result.trades[0]["receive"] == result.trades[1]["give"] == (0, 1, 0, 0, 0)
    assert result.trades[0]["same_turn_construction_after"] == ["road"]
    assert learner["domestic_cards_given"] == partner["domestic_cards_received"] == 1
    assert learner["first_settlement_turn"] is None and learner["vp_3_turn"] is None


def test_invitation_counteroffer_is_not_an_opening_proposal():
    game = trade_position()
    recorder = DiagnosticRecorder()
    recorder.step(game, A.INVITE_TRADE)
    recorder.step(game, offer((0, 1, 0, 0, 0), (1, 0, 0, 0, 0)))
    recorder.step(game, A.REJECT_TRADE)
    recorder.step(game, A.REJECT_TRADE)
    recorder.step(game, A.CONFIRM_TRADE_WITH[1])
    result = recorder.finish(game)
    assert result.players[0]["invitations"] == 1
    assert result.players[0]["opening_proposals"] == 0
    assert result.players[1]["counteroffers"] == 1
    assert result.players[0]["completed_via_counteroffer"] == 1
    assert result.players[0]["negotiations_with_direct_acceptance"] == 0
    assert result.players[0]["all_opponents_rejected"] == 0
    assert result.players[0]["opening_direct_acceptance_rate"] is None
    assert result.trades[0]["via_counteroffer"] and result.offers[0]["offer_kind"] == "counteroffer"


def test_failed_and_repeated_offers_within_and_across_turns():
    game = trade_position()
    recorder = DiagnosticRecorder()
    terms = offer((1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    for _ in range(2):
        recorder.step(game, terms)
        respond(recorder, game)
        recorder.step(game, A.CANCEL_TRADE)
    recorder.step(game, A.END_TURN)
    for _ in range(3):
        recorder.step(game, A.ROLL, (1, 1))
        recorder.step(game, A.END_TURN)
    recorder.step(game, A.ROLL, (1, 1))
    recorder.step(game, terms)
    respond(recorder, game)
    recorder.step(game, A.CANCEL_TRADE)
    row = recorder.finish(game).players[0]
    assert row["failed_negotiations"] == row["all_opponents_rejected"] == 3
    assert row["unaffordable_opening_offers"] == 3
    assert row["repeated_failed_offers_same_turn"] == 1
    assert row["repeated_identical_offers_across_turns"] == 1
    assert row["negotiation_completion_rate"] == row["opening_direct_acceptance_rate"] == 0


def test_reverse_resource_directions_do_not_require_identical_quantities():
    game = trade_position()
    recorder = DiagnosticRecorder()
    for terms in (offer((2, 0, 0, 0, 0), (0, 1, 0, 0, 0)), offer((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))):
        recorder.step(game, terms)
        respond(recorder, game, accepted=True)
        recorder.step(game, A.CONFIRM_TRADE_WITH[1])
    rows = recorder.finish(game).players
    assert rows[0]["same_turn_reverse_exchanges"] == rows[1]["same_turn_reverse_exchanges"] == 1


def test_setup_and_dice_production_are_counted_from_events():
    game = Game(EngineConfig(max_turns=2), seed=7)
    recorder = DiagnosticRecorder(trajectories=True)
    all_events = []
    while not game.is_over:
        all_events.extend(recorder.step(game, game.legal_actions()[0]))
    result = recorder.finish(game)
    assert all(row["setup_settlements"] == row["setup_roads"] == 2 for row in result.players)
    assert all(row["settlements"] == row["roads"] == row["cities"] == 0 for row in result.players)
    assert sum(row["completed_turns"] for row in result.players) == game.turn
    assert sum(row["decisions"] for row in result.players) == len(game.history)
    for seat, row in enumerate(result.players):
        produced = [
            event
            for event in all_events
            if event.kind == EventKind.RESOURCES_PRODUCED and event.player == seat
        ]
        assert row["dice_resources"] == sum(sum(event.counts) for event in produced)
        assert row["dice_resources_by_resource"] == [
            sum(event.counts[r] for event in produced) for r in range(5)
        ]
        assert row["setup_production"] is not None and row["production_samples"] == game.turn
        assert row["first_city_turn"] is None
        if row["completed_turns"] == 0:
            assert row["dice_resources_per_player_turn"] is None
    assert {row["point"] for row in result.trajectories} == {"setup_end", "turn_end", "terminal"}


def test_production_diagnostics_are_raw_uncapped_capacity():
    game = post_setup_game()
    game.state.corner_owner = [0] * len(game.state.corner_owner)
    game.state.corner_building = [Building.CITY] * len(game.state.corner_building)
    game.state.robber_tile = next(
        t for t, resource in enumerate(game.board.tile_resource) if resource >= 0
    )
    row = DiagnosticRecorder().finish(game).players[0]
    assert row["final_production_capacity"] > row["final_production"] > 2
    assert sum(row["final_capacity_by_resource"]) == pytest.approx(row["final_production_capacity"])


def test_recorded_history_replays_state_and_rng_exactly():
    game = Game(EngineConfig(max_turns=3, allow_domestic_trade=True), seed=17)
    while not game.is_over:
        game.step(game.legal_actions()[0])
    before = game.to_dict()
    result = diagnose_game(game)
    assert sum(row["decisions"] for row in result.players) == len(game.history)
    assert game.to_dict() == before
    game.state.bank[0] -= 1
    with pytest.raises(ValueError, match="final state/RNG"):
        diagnose_game(game)


def test_trade_affordability_losses_and_public_leader_attribution():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (1, 1, 1, 1, 0))
    give(game, 1, (0, 0, 0, 0, 1))
    put_settlement(game, 1, 0)
    recorder = DiagnosticRecorder()
    recorder.step(game, offer((1, 0, 0, 0, 0), (0, 0, 0, 0, 1)))
    respond(recorder, game, accepted=True)
    recorder.step(game, A.CONFIRM_TRADE_WITH[1])
    row = recorder.finish(game).players[0]
    assert row["trades_removing_road"] == row["trades_removing_settlement"] == 1
    assert row["trades_unlocking_dev"] == row["trades_with_ahead_public_leader"] == 1
    assert row["settlement_legal_decisions"] == 0


def test_free_roads_maritime_exchange_and_development_events():
    game = post_setup_game()
    put_settlement(game, 0, 0)
    give(game, 0, (8, 0, 2, 3, 3))
    give_dev(game, 0, DevCard.ROAD_BUILDING)
    recorder = DiagnosticRecorder()
    recorder.step(game, A.PLAY_ROAD_BUILDING)
    while game.phase == Phase.ROAD_BUILDING:
        recorder.step(game, game.legal_actions()[0])
    events = recorder.step(game, A.MARITIME[0][1])
    maritime = next(event for event in events if event.kind == EventKind.MARITIME_TRADED)
    recorder.step(
        game,
        next(action for action in game.legal_actions() if action.type == ActionType.BUILD_ROAD),
    )
    recorder.step(game, A.BUY_DEV, DevCard.KNIGHT)
    row = recorder.finish(game).players[0]
    assert row["free_roads"] == 2 and row["purchased_roads"] == 1 and row["roads"] == 3
    assert row["dev_cards_bought"] == row["dev_played_road_building"] == 1
    assert row["maritime_cards_given"] == -min(maritime.counts)
    assert row["maritime_cards_received"] == 1 and row["dice_resources"] == 0
    assert row["road_when_city_legal"] == row["dev_when_city_legal"] == 1
    assert row["trading_or_end_when_city_legal"] == 1


def test_immediate_city_and_victory_milestones_are_zero_not_missing():
    game = post_setup_game()
    put_settlement(game, 0, 0)
    other = next(
        c for c in range(len(game.state.corner_owner)) if c != 0 and c not in CORNER_NEIGHBORS[0]
    )
    put_settlement(game, 0, other)
    give(game, 0, (0, 0, 0, 2, 3))
    recorder = DiagnosticRecorder()
    recorder.step(game, A.CITY_AT[0])
    give(game, 0, (0, 0, 0, 2, 3))
    recorder.step(game, A.END_TURN)
    row = recorder.finish(game).players[0]
    assert row["first_city_turn"] == row["vp_3_turn"] == 0
    assert row["vp_4_turn"] is None and row["first_settlement_turn"] is None
    assert row["city_when_city_legal"] == row["trading_or_end_when_settlement_or_city_legal"] == 1
    assert row["settlement_or_city_legal_decisions"] == 2


def test_unselected_counteroffers_are_remembered_as_failed_offers():
    game = trade_position()
    recorder = DiagnosticRecorder()
    counter = offer((0, 1, 0, 0, 0), (1, 0, 0, 0, 0))
    for completed in (False, True):
        recorder.step(game, A.INVITE_TRADE)
        recorder.step(game, counter)
        recorder.step(game, A.REJECT_TRADE)
        recorder.step(game, A.REJECT_TRADE)
        recorder.step(game, A.CONFIRM_TRADE_WITH[1] if completed else A.CANCEL_TRADE)
    result = recorder.finish(game)
    assert result.players[1]["repeated_failed_offers_same_turn"] == 1
    assert result.offers[1]["repeated_failed_same_turn"]
    assert result.trades[0]["negotiation_decision"] == result.offers[1]["negotiation_decision"]


def test_setup_production_is_recorded_when_setup_ends_the_game():
    game = Game(EngineConfig(vp_to_win=2), seed=0)
    recorder = DiagnosticRecorder()
    while not game.is_over:
        recorder.step(game, game.legal_actions()[0])
    assert all(row["setup_production"] is not None for row in recorder.finish(game).players)
