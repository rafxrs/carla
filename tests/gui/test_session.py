import pytest

from carla.agents import make_agent
from carla.engine import actions as A
from carla.engine.actions import Action, ActionType, TradeOffer
from carla.engine.config import EngineConfig
from carla.engine.events import EventKind
from carla.engine.game import Game, IllegalActionError
from carla.engine.record import GameRecord, ReplayError
from carla.engine.rng import Rng
from carla.engine.simulate import play_random_game
from carla.engine.text import describe_event
from carla.eval.runner import play_game
from carla.gui.controller import Controller
from carla.gui.session import GameSession, HumanAgent, Pacer
from tests.engine.scenario import give, post_setup_game
from tests.gui.clicks import drive, script_for


def test_human_can_submit_a_multi_card_trade():
    game = post_setup_game(config=EngineConfig(allow_domestic_trade=True))
    give(game, 0, (2, 0, 0, 0, 0))
    offer = Action(
        ActionType.PROPOSE_TRADE,
        trade=TradeOffer((2, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
    )
    assert offer not in game.legal_actions() and game.is_legal(offer)
    human = HumanAgent()
    human.submit(offer)
    assert human.act(game, 0, game.legal_actions()) == offer
    game.step(offer)
    assert game.state.pending_trade == offer.trade


@pytest.fixture(scope="module")
def record():
    return GameRecord.from_game(play_random_game(5), {"agents": ["random"] * 4})


def test_seek_matches_record_replay(record):
    session = GameSession(record)
    n = len(record)
    assert session.position == 0 and not session.live and session.seat_names() == ["random"] * 4
    for k in (1, 31, 32, 33, 100, n // 2, n, 7, 0, n - 1, 64, 65, 3):
        session.seek(k)
        assert session.position == k
        assert session.game.state_hash() == record.replay(k).state_hash()
    session.seek(-5)
    assert session.position == 0
    session.seek(10**6)
    assert session.at_end and session.position == n
    assert not session.step_forward()  # a replay never grows
    assert session.step_back() and session.position == n - 1
    assert session.step_forward() and session.at_end


def test_replay_log_follows_the_position(record):
    session = GameSession(record)
    lines = [session.entry_text(e) for e in session.log if e.event is not None]
    expected = [
        describe_event(ev, record.board) for _, events in record.iter_steps() for ev in events
    ]
    assert lines == expected
    assert session.entry_text(session.log[0]) == "Setup"
    headers = [e.header for e in session.log if e.event is None]
    assert headers[1] == "Turn 1 · P0" and headers[2] == "Turn 2 · P1"
    session.seek(20)
    visible = session.visible_log()
    assert visible and all(e.step <= 20 for e in visible)
    assert len(visible) == sum(e.step <= 20 for e in session.log)
    session.seek(0)
    assert [e.header for e in session.visible_log()] == ["Setup"]


def test_corrupt_records_are_rejected(record):
    steps = list(record.steps)
    steps[30], steps[31] = steps[31], steps[30]
    with pytest.raises(ReplayError):
        GameSession(GameRecord(record.seed, record.config, record.board, steps, record.hashes))


def test_live_bot_game_matches_the_eval_runner(tmp_path):
    names = ["heuristic", "random", "heuristic", "random"]
    session = GameSession.new_game(EngineConfig(), 3, [make_agent(n) for n in names])
    assert session.live and session.human_seat is None and session.seat_names() == names
    assert not session.finished
    while session.step_forward():
        pass
    reference = play_game([make_agent(n) for n in names], seed=3)
    assert session.game.is_over and session.finished and not session.can_extend
    assert (session.game.winner, session.length) == (reference.winner, reference.steps)
    saved = session.final_record().save(tmp_path / "game.json")
    loaded = GameRecord.load(saved)
    assert loaded.metadata["agents"] == names
    assert loaded.replay().state_hash() == session.game.state_hash()
    session.seek(40)  # live sessions can be reviewed too
    assert session.game.state_hash() == loaded.replay(40).state_hash() and session.finished
    assert GameSession(loaded).finished and not GameSession(record_of_unfinished()).finished
    assert session.final_record().hashes == loaded.hashes


def record_of_unfinished() -> GameRecord:
    game = Game(seed=2)
    game.play(game.legal_actions()[:1])
    return GameRecord.from_game(game)


def test_play_mode_full_game_through_the_controller():
    """The human seat picks random legal actions, entered as clicks through the controller."""
    bots = [make_agent("heuristic") for _ in range(3)]
    session = GameSession.new_game(EngineConfig(), 11, [HumanAgent(), *bots], viewer=0)
    controller, rng = Controller(seat=0), Rng(7)
    human_moves = 0
    while not session.game.is_over:
        if not session.waiting_for_human:
            assert session.advance()
            continue
        game = session.game
        legal = game.legal_actions()
        controller.reset(game.state, legal)
        kinds = sorted({a.type for a in legal})
        kind = kinds[rng.randbelow(len(kinds))]
        target = rng.choice([a for a in legal if a.type == kind])
        action = drive(controller, script_for(target, legal))
        assert action == target
        assert action is not None
        session.submit(action)
        human_moves += 1
    assert human_moves > 30 and session.game.winner is not None
    assert human_moves == sum(step.player == 0 for step in session.record.steps)
    assert session.seat_names() == ["You", "heuristic", "heuristic", "heuristic"]
    assert session.final_record().replay().state_hash() == session.game.state_hash()


def test_human_agent_and_submit_checks():
    session = GameSession.new_game(
        EngineConfig(), 1, [HumanAgent(), *(make_agent("random") for _ in range(3))], viewer=0
    )
    assert session.waiting_for_human and session.human_seat == 0
    assert not session.advance()  # the human moves only through submit
    with pytest.raises(IllegalActionError):
        session.submit(A.ROLL)  # setup: a settlement first
    human = HumanAgent()
    with pytest.raises(RuntimeError):
        human.act(session.game, 0, session.game.legal_actions())
    human.submit(A.SETTLEMENT_AT[0])
    assert human.ready
    assert human.act(session.game, 0, session.game.legal_actions()) is A.SETTLEMENT_AT[0]
    session.submit(A.SETTLEMENT_AT[0])
    session.submit(session.game.legal_actions()[0])
    assert not session.waiting_for_human
    with pytest.raises(RuntimeError):
        session.submit(A.ROLL)
    assert session.advance() and session.position == 3


def test_log_hides_other_seats_cards_from_the_player():
    session = GameSession.new_game(EngineConfig(), 2, [make_agent("heuristic") for _ in range(4)])
    while session.step_forward():
        pass
    kinds = (EventKind.DEV_CARD_BOUGHT, EventKind.RESOURCE_STOLEN)
    private = [e for e in session.log if e.event is not None and e.event.kind in kinds]
    buyers = sorted({e.player for e in private if e.event and e.event.kind == kinds[0]})
    assert len(buyers) >= 2
    session.viewer = buyers[0]  # as if that seat were the human player
    for entry in private:
        assert entry.event is not None
        involved = buyers[0] in (entry.event.player, entry.event.target)
        text = session.entry_text(entry)
        assert ("(hidden)" in text or "steals a card" in text) == (not involved)
    session.viewer = None  # a spectator (or the game-over review) sees every card
    assert not any("(hidden)" in session.entry_text(e) for e in private)


def test_last_dice_follows_the_position():
    game = play_random_game(8)
    session = GameSession(GameRecord.from_game(game))
    rolls = [i for i, s in enumerate(game.history, start=1) if s.action.type.name == "ROLL_DICE"]
    session.seek(rolls[0] - 1)
    assert session.last_dice() is None
    session.seek(rolls[1])
    assert session.last_dice() == game.history[rolls[1] - 1].outcome


def test_pacer():
    pacer = Pacer(delay=1.0)
    assert pacer.due(0.5) == 0 and pacer.due(0.6) == 1
    assert pacer.due(2.0) == 2  # 0.1 carried over: 2.1 seconds
    assert pacer.due(0.5, factor=0.2) == 3  # 0.6 seconds at 0.2 per move
    pacer.restart()
    assert pacer.due(0.99) == 0
    pacer.faster()
    assert pacer.speed == 2 and pacer.due(0.02) == 2  # 1.01 seconds waited, 0.5 per move
    for _ in range(20):
        pacer.faster()
    assert pacer.speed == Pacer.SPEEDS[-1]
    assert pacer.due(1000.0) == 50 and pacer.due(0.0) == 0  # a backlog is dropped
    pacer.playing = False
    assert pacer.due(10.0) == 0
    for _ in range(20):
        pacer.slower()
    assert pacer.speed == Pacer.SPEEDS[0]
    instant = Pacer(delay=0)
    assert instant.due(0.001) == 1


def test_new_game_uses_the_config_board():
    config = EngineConfig(board_seed=99)
    session = GameSession.new_game(config, 5, [make_agent("random") for _ in range(4)])
    assert session.game.board == Game(config, 1).board
    assert session.record.board == session.game.board
