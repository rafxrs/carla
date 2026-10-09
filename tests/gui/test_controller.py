from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.constants import GENERIC_PORT, DevCard
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.state import Phase
from carla.engine.topology import EDGE_CORNERS, PORT_CORNERS, TILE_CORNERS
from carla.gui.controller import Controller, negotiation_panel
from carla.gui.theme import RESOURCE_LABELS
from tests.engine.scenario import BOARD, give, give_dev, post_setup_game, put_settlement
from tests.gui.clicks import card_script_for, drive, script_for


def controller_for(game: Game) -> Controller:
    controller = Controller(game.decision_player)
    controller.reset(game.state, game.legal_actions())
    return controller


def check_every_action(game: Game) -> set[ActionType]:
    """Every legal action of the position is produced, exactly, by its click script."""
    legal = game.legal_actions()
    for action in legal:
        controller = controller_for(game)
        assert drive(controller, script_for(action, legal, game.state)) == action
        if action.type == ActionType.PLAY_DEV_CARD:  # or by clicking the card itself
            controller = controller_for(game)
            assert action.dev in controller.playable_cards()
            assert drive(controller, card_script_for(action)) == action
    return {a.type for a in legal}


def test_click_scripts_reach_every_legal_action_in_random_games():
    seen: set[ActionType] = set()
    for seed in range(4):
        game, rng = Game(EngineConfig(allow_domestic_trade=seed % 2 == 1), seed), Rng(seed)
        while not game.is_over and game.num_steps < 1500:
            seen |= check_every_action(game)
            legal = game.legal_actions()
            kinds = sorted({a.type for a in legal})  # random by type, like RandomAgent
            kind = kinds[rng.randbelow(len(kinds))]
            game.step(rng.choice([a for a in legal if a.type == kind]))
    assert seen == set(ActionType)


def test_development_cards_through_the_picker():
    game = post_setup_game(rolled=False)
    for card in DevCard:
        give_dev(game, 0, card)
    put_settlement(game, 0, 20)
    controller = controller_for(game)
    assert controller.prompt().startswith("Roll the dice")
    controller.press("play_dev")
    picker = controller.picker
    assert picker is not None and picker.kind == "dev"
    enabled = {o.key: o.enabled for o in picker.options}
    assert enabled == {0: True, 1: True, 2: True, 3: True, 4: False}  # VP cards are never played
    assert controller.choose(DevCard.KNIGHT) is A.PLAY_KNIGHT

    controller.press("play_dev")
    assert controller.choose(DevCard.YEAR_OF_PLENTY) is None
    assert controller.choose(3) is None  # first card: grain
    assert controller.picker is not None and "Grain" in controller.picker.title
    assert controller.choose(1) is A.YEAR_OF_PLENTY[1][3]  # either order gives the same action

    controller.press("play_dev")
    controller.choose(DevCard.MONOPOLY)
    assert controller.choose(4) is A.MONOPOLY_ON[4]
    controller.press("play_dev")
    assert controller.choose(DevCard.ROAD_BUILDING) is A.PLAY_ROAD_BUILDING


def test_year_of_plenty_only_offers_what_the_bank_has():
    game = post_setup_game()
    give_dev(game, 0, DevCard.YEAR_OF_PLENTY)
    game.state.bank[:] = [0, 1, 2, 5, 5]
    game.set_state(game.state)
    controller = controller_for(game)
    controller.press("play_dev")
    controller.choose(DevCard.YEAR_OF_PLENTY)
    assert controller.picker is not None
    assert [o.enabled for o in controller.picker.options] == [False, True, True, True, True]
    controller.choose(1)  # brick: the bank has a single one, so brick+brick is not offered
    assert controller.picker is not None
    assert [o.enabled for o in controller.picker.options] == [False, False, True, True, True]
    assert controller.choose(1) is None and controller.choose(2) is A.YEAR_OF_PLENTY[1][2]


def test_bank_trades_follow_the_port_rates():
    slot = next(i for i, kind in enumerate(BOARD.port_types) if kind != GENERIC_PORT)
    resource = BOARD.port_types[slot]
    other = (resource + 1) % 5
    game = post_setup_game()
    put_settlement(game, 0, PORT_CORNERS[slot][0])
    give(game, 0, [2 if r == resource else 3 for r in range(5)])
    controller = controller_for(game)
    assert controller.offerable_cards() == set(range(5)) and controller.trade_panel() is None
    for _ in range(3):
        controller.offer_card(other)
    panel = controller.trade_panel()
    assert panel is not None and panel.kind == "offer" and panel.give[other] == 3
    assert [b.key for b in panel.buttons] == ["bank"]  # no domestic trade by default
    assert f"1 card for 4 {RESOURCE_LABELS[other].lower()}" in panel.hint
    assert other not in controller.offerable_cards()  # all three are in the offer
    controller.choose_trade(("want", resource))
    assert controller.choose_trade("bank") is None  # 3 cards at 4:1
    assert controller.cancel() and controller.trade_panel() is None
    controller.offer_card(resource)
    controller.offer_card(resource)
    controller.choose_trade(("want", other))
    assert controller.choose_trade("bank") is A.MARITIME[resource][other]  # 2:1 port
    assert controller.take_queued() == [] and controller.trade_panel() is None


def test_one_bank_trade_click_can_make_several_trades():
    game = post_setup_game()
    give(game, 0, (8, 0, 0, 0, 1))
    controller = controller_for(game)
    for _ in range(8):
        controller.offer_card(0)
    assert controller.offerable_cards() == {4}  # no lumber left in the hand
    controller.choose_trade(("want", 0))  # lumber is being given, so it cannot be asked for
    panel = controller.trade_panel()
    assert panel is not None and panel.get == (0, 0, 0, 0, 0) and not panel.wants[0]
    controller.choose_trade(("want", 3))
    controller.choose_trade(("want", 1))
    panel = controller.trade_panel()
    assert panel is not None and not any(panel.wants)  # 8 lumber buy two cards at most
    assert controller.choose_trade("bank") is A.MARITIME[0][1]
    assert controller.take_queued() == [A.MARITIME[0][3]] and controller.take_queued() == []


def test_offer_cards_go_back_and_forth():
    game = post_setup_game()
    give(game, 0, (4, 1, 0, 0, 0))
    controller = controller_for(game)
    controller.toggle_trade()  # the T key: an empty offer
    panel = controller.trade_panel()
    assert panel is not None and not any(panel.give) and "Click your cards" in panel.hint
    controller.offer_card(0)
    controller.offer_card(0)
    controller.choose_trade(("want", 3))
    assert controller.offer == ((2, 0, 0, 0, 0), (0, 0, 0, 1, 0))
    assert controller.choose_trade(("give", 0)) is None
    assert controller.choose_trade(("get", 3)) is None
    assert controller.choose_trade(("get", 3)) is None  # nothing left to take off
    assert controller.offer == ((1, 0, 0, 0, 0), (0,) * 5)
    assert 3 not in controller.offerable_cards() and 1 in controller.offerable_cards()
    assert controller.press("end_turn") is A.END_TURN and controller.trade_panel() is None
    controller.toggle_trade()
    controller.toggle_trade()  # the T key again: closed
    assert controller.trade_panel() is None
    assert controller.prompt().startswith("Build on a highlighted spot, click your cards")


def test_discards_are_one_card_at_a_time_and_cannot_be_cancelled():
    game = post_setup_game(rolled=False)
    give(game, 0, (3, 0, 2, 2, 2))
    game.step(A.ROLL, outcome=(3, 4))
    assert game.phase == Phase.DISCARD and game.decision_player == 0
    controller = controller_for(game)
    picker = controller.picker
    assert picker is not None and picker.kind == "discard" and not picker.cancellable
    assert picker.title == "Discard 4 more cards"
    assert not controller.cancel() and controller.picker is picker
    assert not any(b.enabled for b in controller.buttons())
    assert controller.choose(1) is None  # no brick in hand
    for left in (4, 3, 2, 1):
        controller = controller_for(game)
        assert controller.picker is not None
        assert controller.picker.title.startswith(f"Discard {left} more card")
        action = controller.choose(0) or controller.choose(2)
        assert action is not None and action in game.legal_actions()
        game.step(action)
    assert game.phase == Phase.MOVE_ROBBER


def test_robber_victim_choice():
    tile = 0 if BOARD.desert_tile != 0 else 1
    game = post_setup_game(rolled=False)
    corners = TILE_CORNERS[tile]
    put_settlement(game, 1, corners[0])
    put_settlement(game, 2, corners[2])
    put_settlement(game, 3, corners[4])
    give(game, 1, (1, 0, 0, 0, 0))
    give(game, 2, (0, 1, 1, 0, 0))  # seat 3 has no cards: it cannot be robbed
    game.step(A.ROLL, outcome=(6, 1))
    controller = controller_for(game)
    assert set(controller.tile_targets()) == set(range(19)) - {BOARD.desert_tile}
    assert controller.click_tile(BOARD.desert_tile) is None  # the robber has to move
    assert controller.click_tile(tile) is None
    picker = controller.picker
    assert picker is not None and picker.kind == "victim"
    assert [(o.key, o.detail) for o in picker.options] == [(1, "1 card"), (2, "2 cards")]
    assert controller.tile_targets() == {}  # board targets are hidden behind the picker
    assert controller.cancel() and controller.picker is None
    controller.click_tile(tile)
    assert controller.choose(2) is A.ROBBER_TO[tile][3]
    targets = controller.tile_targets()
    lone = next(t for t, moves in targets.items() if moves == [A.ROBBER_TO[t][0]])
    assert controller.click_tile(lone) is A.ROBBER_TO[lone][0]  # nobody to rob: no picker


def test_board_clicks_follow_the_legal_targets():
    game = post_setup_game()
    put_settlement(game, 0, 20)
    give(game, 0, (2, 2, 1, 3, 3))
    controller = controller_for(game)
    corners, edges = controller.corner_targets(), controller.edge_targets()
    assert corners == {20: A.CITY_AT[20]}  # no road yet, so no settlement site
    assert set(edges) == {e for e in range(72) if A.ROAD_AT[e] in game.legal_actions()}
    assert controller.click_corner(20) is A.CITY_AT[20]
    assert controller.click_corner(0) is None and controller.click_edge(0) is None
    assert controller.prompt().startswith("Build")


def test_buttons_follow_legality_and_inactive_controller_ignores_input():
    game = post_setup_game(rolled=False)
    controller = controller_for(game)
    states = {b.id: b.enabled for b in controller.buttons()}
    assert states == {
        "roll": True,
        "end_turn": False,
        "buy_dev": False,
        "play_dev": False,
    }
    assert controller.press("end_turn") is None and controller.press("roll") is A.ROLL
    idle = Controller(seat=1)
    idle.reset(game.state)  # seat 0 is to act: nothing for seat 1
    assert not idle.active and idle.prompt() == ""
    assert not any(b.enabled for b in idle.buttons())
    assert idle.press("roll") is None and idle.click_corner(3) is None
    idle.reset(None)
    assert idle.choose(0) is None and not idle.cancel()


def test_opening_a_picker_blocks_board_clicks_until_cancelled():
    game = post_setup_game()
    put_settlement(game, 0, 20)
    give(game, 0, (4, 0, 0, 2, 3))
    give_dev(game, 0, DevCard.MONOPOLY)
    controller = controller_for(game)
    controller.offer_card(0)  # an offer leaves the board and the buttons usable
    assert controller.corner_targets() == {20: A.CITY_AT[20]}
    controller.press("play_dev")  # a picker replaces the offer
    assert controller.click_corner(20) is None and controller.corner_targets() == {}
    assert controller.picker is not None and controller.picker.kind == "dev"
    assert controller.offer == ((0,) * 5, (0,) * 5)
    assert controller.choose_index(3) is None  # Monopoly, then a resource
    assert controller.choose_index(0) is A.MONOPOLY_ON[0]
    controller.press("play_dev")
    assert controller.cancel() and controller.click_corner(20) is A.CITY_AT[20]


def test_setup_prompts_and_targets():
    game = Game(seed=3)
    controller = controller_for(game)
    assert controller.prompt() == "Place a settlement"
    assert len(controller.corner_targets()) == 54 and not controller.edge_targets()
    action = controller.click_corner(10)
    assert action is not None and action is A.SETTLEMENT_AT[10]
    game.step(action)
    controller = controller_for(game)
    assert controller.prompt().startswith("Place a road")
    assert controller.edge_targets() and not controller.corner_targets()
    assert all(10 in EDGE_CORNERS[e] for e in controller.edge_targets())


# ------------------------------------------------------------------ domestic trade

TRADING = EngineConfig(allow_domestic_trade=True)
_ORE = (0, 0, 0, 0, 1)


def step(game: Game, action: Action | None) -> None:
    assert action is not None
    game.step(action)


def test_offer_to_players_needs_valid_terms():
    game = post_setup_game(config=TRADING)
    give(game, 0, (5, 1, 0, 0, 0))
    controller = controller_for(game)
    controller.offer_card(0)
    panel = controller.trade_panel()
    assert panel is not None and [b.key for b in panel.buttons] == ["invite", "bank", "propose"]
    assert controller.choose_trade("propose") is None  # nothing asked for yet
    for _ in range(4):
        controller.offer_card(0)
    controller.choose_trade(("want", 4))
    panel = controller.trade_panel()
    assert panel is not None and "players: 4 cards a side at most" in panel.hint
    assert controller.choose_trade("propose") is None  # five cards: over the limit
    controller.choose_trade(("give", 0))
    action = controller.choose_trade("propose")
    assert action == Action(ActionType.PROPOSE_TRADE, trade=TradeOffer((4, 0, 0, 0, 0), _ORE))
    assert action is not None and game.is_legal(action) and controller.trade_panel() is None


def test_answering_an_offer_shows_its_cards():
    game = post_setup_game(config=TRADING)
    give(game, 1, (1, 0, 0, 0, 0))
    give(game, 0, (0, 0, 0, 0, 1))
    s = game.state
    s.current_player = s.decision_player = 1
    game.set_state(s)
    game.step(Action(ActionType.PROPOSE_TRADE, trade=TradeOffer((1, 0, 0, 0, 0), _ORE)))
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    controller = controller_for(game)  # seat 0 answers last
    panel = controller.trade_panel()
    assert panel is not None and panel.kind == "respond" and panel.title == "P1 offers a trade"
    (row,) = panel.rows
    assert (row.give, row.get) == (_ORE, (1, 0, 0, 0, 0))  # from the answering seat's side
    assert [(b.key, b.enabled) for b in panel.buttons] == [
        ("reject", True),
        ("counter", True),
        ("accept", True),
    ]
    assert not controller.cancel() and controller.prompt() == "P1 offers a trade"
    assert controller.choose_index(2) is A.ACCEPT_TRADE  # the number keys follow the buttons
    controller.reset(game.state, game.legal_actions())
    controller.offer_card(4)  # a hand card starts a counteroffer
    panel = controller.trade_panel()
    assert panel is not None and panel.kind == "counter" and panel.title == "Counteroffer to P1"
    assert controller.cancel()
    panel = controller.trade_panel()
    assert panel is not None and panel.kind == "respond"


def test_a_trade_completes_through_the_panels():
    game = post_setup_game(config=TRADING)
    give(game, 0, (1, 0, 0, 0, 0))
    give(game, 1, (0, 1, 0, 0, 0))
    proposer = controller_for(game)
    assert drive(proposer, [("offer_card", 0), ("choose_trade", ("want", 1))]) is None
    step(game, proposer.choose_trade("propose"))
    assert game.state.phase == Phase.TRADE_RESPONSE
    waiting = negotiation_panel(game.state, viewer=0)
    assert waiting is not None and waiting.title == "Your trade offer"
    assert [r.status for r in waiting.rows] == ["offers", "deciding…", "waiting", "waiting"]
    for answer in ("accept", "reject", "reject"):
        step(game, controller_for(game).choose_trade(answer))
    confirm = controller_for(game)
    panel = confirm.trade_panel()
    assert panel is not None and panel.kind == "responses"
    assert [(r.label, r.status, r.button is not None) for r in panel.rows] == [
        ("You", "offers", False),
        ("P1", "accepts", True),
        ("P2", "declines", False),
        ("P3", "declines", False),
    ]
    assert confirm.choose_trade(("confirm", 2)) is None  # P2 declined
    step(game, confirm.choose_trade(("confirm", 1)))
    assert game.state.hands[0] == [0, 1, 0, 0, 0]
    assert game.state.hands[1] == [1, 0, 0, 0, 0]


def test_an_invitation_gets_a_counteroffer_shown_as_cards():
    game = post_setup_game(config=TRADING)
    give(game, 0, (2, 0, 0, 0, 0))
    give(game, 1, (0, 1, 0, 0, 0))
    active = controller_for(game)
    active.offer_card(0)
    assert active.choose_trade("invite") is A.INVITE_TRADE
    game.step(A.INVITE_TRADE)
    responder = controller_for(game)
    panel = responder.trade_panel()
    assert panel is not None and panel.title == "P0 asks for offers" and not panel.rows
    assert [b.key for b in panel.buttons] == ["reject", "counter"]
    assert responder.choose_trade("counter") is None
    panel = responder.trade_panel()
    assert panel is not None and panel.kind == "counter"
    assert [b.key for b in panel.buttons] == ["back", "propose"]
    assert responder.choose_trade("back") is None
    panel = responder.trade_panel()
    assert panel is not None and panel.kind == "respond"
    counter = drive(
        responder,
        [("offer_card", 1), ("choose_trade", ("want", 0)), ("choose_trade", "propose")],
    )
    assert counter == A.ONE_FOR_ONE[1][0]
    step(game, counter)
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    active = controller_for(game)
    panel = active.trade_panel()
    assert panel is not None
    p1 = panel.rows[0]  # no offer row after an invitation
    assert (p1.label, p1.status) == ("P1", "counters")
    assert (p1.give, p1.get) == ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))  # P0's side
    step(game, active.choose_index(0))  # the number keys: the first Trade button
    assert game.state.hands[0] == [1, 1, 0, 0, 0]
    assert game.state.hands[1] == [1, 0, 0, 0, 0]


def test_a_counteroffer_the_proposer_cannot_pay_cannot_be_picked():
    game = post_setup_game(config=TRADING)
    give(game, 0, (1, 0, 0, 0, 0))
    give(game, 1, (0, 1, 0, 0, 0))
    step(game, A.ONE_FOR_ONE[0][1])
    step(game, A.ONE_FOR_ONE[1][4])  # P1 wants ore, which P0 does not have
    game.step(A.REJECT_TRADE)
    game.step(A.REJECT_TRADE)
    panel = controller_for(game).trade_panel()
    assert panel is not None and panel.hint == "No deal to make"
    button = panel.rows[1].button
    assert button is not None and not button.enabled
    assert [b.key for b in panel.buttons] == ["cancel_trade"]
    read_only = negotiation_panel(game.state, viewer=None)
    assert read_only is not None and read_only.columns == ("P0 gives", "P0 gets")
    assert not read_only.buttons and all(r.button is None for r in read_only.rows)
    assert read_only.hint == "P0 is choosing a deal…"
    assert negotiation_panel(post_setup_game().state) is None


def test_trading_needs_the_main_phase():
    game = post_setup_game(rolled=False, config=TRADING)
    give(game, 0, (1, 0, 0, 0, 0))
    controller = controller_for(game)
    assert controller.offerable_cards() == frozenset()
    controller.offer_card(0)
    controller.toggle_trade()
    assert controller.trade_panel() is None
    idle = Controller(seat=1)
    idle.reset(game.state)
    assert idle.offerable_cards() == frozenset() and idle.trade_panel() is None
    assert idle.choose_trade("bank") is None and idle.take_queued() == []


def test_clicking_a_development_card_plays_it_like_the_picker():
    game = post_setup_game()
    for card in DevCard:
        give_dev(game, 0, card)
    put_settlement(game, 0, 20)
    controller = controller_for(game)
    assert controller.playable_cards() == {0, 1, 2, 3}  # victory points are never played
    assert controller.play_card(DevCard.KNIGHT) is A.PLAY_KNIGHT
    assert controller.play_card(DevCard.YEAR_OF_PLENTY) is None
    picker = controller.picker
    assert picker is not None and picker.kind == "plenty"
    assert controller.choose(0) is None and controller.choose(4) is A.YEAR_OF_PLENTY[0][4]
    assert controller.play_card(DevCard.MONOPOLY) is None
    assert controller.picker is not None and controller.picker.kind == "monopoly"
    assert controller.play_card(DevCard.VICTORY_POINT) is None
    assert controller.picker is not None and controller.picker.kind == "monopoly"  # unchanged


def test_cards_bought_this_turn_or_a_mandatory_picker_block_card_clicks():
    game = post_setup_game()
    give_dev(game, 0, DevCard.KNIGHT)
    game.state.dev_bought_this_turn[DevCard.KNIGHT] = 1
    game.set_state(game.state)
    controller = controller_for(game)
    assert controller.playable_cards() == frozenset()
    assert controller.play_card(DevCard.KNIGHT) is None and controller.picker is None
    idle = Controller(seat=0)
    idle.reset(None)
    assert idle.playable_cards() == frozenset() and idle.play_card(DevCard.KNIGHT) is None
    roller = post_setup_game(rolled=False)
    give(roller, 0, (3, 0, 2, 2, 2))
    give_dev(roller, 0, DevCard.KNIGHT)
    roller.step(A.ROLL, outcome=(3, 4))  # a 7 with nine cards: discard first
    discarding = controller_for(roller)
    assert discarding.picker is not None and discarding.picker.kind == "discard"
    assert discarding.playable_cards() == frozenset()
    assert discarding.play_card(DevCard.KNIGHT) is None
