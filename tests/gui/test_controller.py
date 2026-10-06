from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.constants import GENERIC_PORT, DevCard
from carla.engine.game import Game
from carla.engine.rng import Rng
from carla.engine.state import Phase
from carla.engine.topology import EDGE_CORNERS, PORT_CORNERS, TILE_CORNERS
from carla.gui.controller import Controller
from tests.engine.scenario import BOARD, give, give_dev, post_setup_game, put_settlement
from tests.gui.clicks import drive, script_for


def controller_for(game: Game, domestic_trade: bool = False) -> Controller:
    controller = Controller(game.decision_player, domestic_trade)
    controller.reset(game.state, game.legal_actions())
    return controller


def check_every_action(game: Game) -> set[ActionType]:
    """Every legal action of the position is produced, exactly, by its click script."""
    legal = game.legal_actions()
    for action in legal:
        controller = controller_for(game)
        assert drive(controller, script_for(action, legal)) == action
    return {a.type for a in legal}


def test_click_scripts_reach_every_legal_action_in_random_games():
    seen: set[ActionType] = set()
    for seed in range(3):
        game, rng = Game(seed=seed), Rng(seed)
        while not game.is_over and game.num_steps < 1500:
            seen |= check_every_action(game)
            legal = game.legal_actions()
            kinds = sorted({a.type for a in legal})  # random by type, like RandomAgent
            kind = kinds[rng.randbelow(len(kinds))]
            game.step(rng.choice([a for a in legal if a.type == kind]))
    domestic = {"PROPOSE_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"}
    assert {t.name for t in seen} == {t.name for t in ActionType} - domestic


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


def test_maritime_trade_offers_only_legal_ratios():
    slot = next(i for i, kind in enumerate(BOARD.port_types) if kind != GENERIC_PORT)
    resource = BOARD.port_types[slot]
    game = post_setup_game()
    put_settlement(game, 0, PORT_CORNERS[slot][0])
    give(game, 0, [2 if r == resource else 3 for r in range(5)])
    controller = controller_for(game)
    controller.press("bank_trade")
    give_picker = controller.picker
    assert give_picker is not None
    labels = {o.key: (o.label, o.enabled) for o in give_picker.options}
    assert labels[resource][0].startswith("2 ") and labels[resource][1]  # 2:1 port
    other = (resource + 1) % 5
    assert labels[other][0].startswith("4 ") and not labels[other][1]  # 3 cards < 4:1
    assert controller.choose(other) is None and controller.picker is give_picker  # disabled
    controller.choose(resource)
    get_picker = controller.picker
    assert get_picker is not None and resource not in {o.key for o in get_picker.options}
    assert controller.choose(other) is A.MARITIME[resource][other]


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
        "bank_trade": False,
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
    controller.press("bank_trade")
    assert controller.click_corner(20) is None and controller.corner_targets() == {}
    controller.press("play_dev")  # another button replaces the open picker
    assert controller.picker is not None and controller.picker.kind == "dev"
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


# ------------------------------------------------------------------ domestic trade (UI only)


def trade_position(phase: Phase, current: int = 1) -> Game:
    game = post_setup_game()
    s = game.state
    s.phase, s.current_player = phase, current
    s.pending_trade = TradeOffer((1, 0, 0, 0, 0), (0, 0, 0, 0, 2))
    game.set_state(s)
    return game


def test_trade_response_picker():
    game = trade_position(Phase.TRADE_RESPONSE)
    controller = Controller(seat=0, domestic_trade=True)
    controller.reset(game.state, [A.ACCEPT_TRADE, A.REJECT_TRADE])
    picker = controller.picker
    assert picker is not None and picker.kind == "response" and not picker.cancellable
    assert picker.title == "P1 offers 1 lumber" and picker.detail.endswith("2 ore in return")
    assert controller.choose("accept") is A.ACCEPT_TRADE
    controller.reset(game.state, [A.REJECT_TRADE])  # cannot pay: accepting is disabled
    assert controller.choose("accept") is None and controller.choose("reject") is A.REJECT_TRADE


def test_trade_confirmation_picker():
    game = trade_position(Phase.TRADE_CONFIRM, current=0)
    controller = Controller(seat=0, domestic_trade=True)
    legal = [A.CONFIRM_TRADE_WITH[2], A.CONFIRM_TRADE_WITH[3], A.CANCEL_TRADE]
    controller.reset(game.state, legal)
    picker = controller.picker
    assert picker is not None and [o.key for o in picker.options] == [2, 3, "cancel"]
    assert not controller.cancel()
    assert controller.choose(3) is A.CONFIRM_TRADE_WITH[3]
    controller.reset(game.state, legal)
    assert controller.choose("cancel") is A.CANCEL_TRADE


def test_trade_offer_composer():
    game = post_setup_game()
    give(game, 0, (1, 1, 0, 0, 0))
    hidden = controller_for(game)
    assert "trade" not in {b.id for b in hidden.buttons()}  # domestic trade disabled
    controller = controller_for(game, domestic_trade=True)
    assert controller.press("trade") is None
    assert controller.choose("propose") is None  # empty offer
    assert controller.choose(("give", 0, 1)) is None
    assert controller.choose(("give", 0, 1)) is None  # only one lumber in hand: ignored
    controller.choose(("get", 4, 1))
    controller.choose(("get", 4, 1))
    controller.choose(("get", 4, -1))
    assert controller.offer == ((1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    action = controller.choose("propose")
    assert action == Action(
        ActionType.PROPOSE_TRADE, trade=TradeOffer((1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
    )
    assert controller.picker is None


def test_trade_button_needs_the_main_phase():
    game = post_setup_game(rolled=False)
    give(game, 0, (1, 0, 0, 0, 0))
    controller = Controller(0, domestic_trade=True)
    controller.reset(game.state, game.legal_actions())
    assert not next(b for b in controller.buttons() if b.id == "trade").enabled
