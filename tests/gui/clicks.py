"""Click scripts: the GUI input that should produce a given legal action through the controller."""

from collections.abc import Hashable, Sequence

from carla.engine.actions import Action, ActionType
from carla.engine.constants import NUM_RESOURCES, DevCard
from carla.engine.rules import trade_ratios
from carla.engine.state import GameState, Phase
from carla.gui.controller import Controller

_T = ActionType
Script = list[tuple[str, Hashable]]


def offer_script(give: Sequence[int], get: Sequence[int]) -> Script:
    """Click the cards to give in the hand, then the cards to get among the five resources."""
    return [
        *(("offer_card", r) for r in range(NUM_RESOURCES) for _ in range(give[r])),
        *(("choose_trade", ("want", r)) for r in range(NUM_RESOURCES) for _ in range(get[r])),
    ]


def script_for(action: Action, legal: Sequence[Action], state: GameState) -> Script:
    """Controller calls (method, argument) a player makes to choose `action`."""
    t = action.type
    simple = {_T.ROLL_DICE: "roll", _T.END_TURN: "end_turn", _T.BUY_DEV_CARD: "buy_dev"}
    if t in simple:
        return [("press", simple[t])]
    if t in (_T.BUILD_SETTLEMENT, _T.BUILD_CITY):
        return [("click_corner", action.corner)]
    if t == _T.BUILD_ROAD:
        return [("click_edge", action.edge)]
    if t == _T.MOVE_ROBBER:
        same_tile = [a for a in legal if a.type == t and a.tile == action.tile]
        victim: Script = [("choose", action.player)] if len(same_tile) > 1 else []
        return [("click_tile", action.tile), *victim]
    if t == _T.DISCARD:
        return [("choose", action.resource)]
    if t == _T.MARITIME_TRADE:
        give = [0] * NUM_RESOURCES
        give[action.resource] = trade_ratios(state, state.decision_player)[action.resource]
        get = [int(r == action.resource2) for r in range(NUM_RESOURCES)]
        return [*offer_script(give, get), ("choose_trade", "bank")]
    if t == _T.PLAY_DEV_CARD:
        script: Script = [("press", "play_dev"), ("choose", action.dev)]
        if action.dev == DevCard.YEAR_OF_PLENTY:
            script += [("choose", action.resource), ("choose", action.resource2)]
        elif action.dev == DevCard.MONOPOLY:
            script.append(("choose", action.resource))
        return script
    if t == _T.PROPOSE_TRADE:
        assert action.trade is not None
        counter: Script = [("choose_trade", "counter")]
        start = counter if state.phase == Phase.TRADE_RESPONSE else []
        return [*start, *offer_script(*action.trade), ("choose_trade", "propose")]
    if t == _T.INVITE_TRADE:
        card = next(r for r, n in enumerate(state.hands[state.decision_player]) if n)
        return [("offer_card", card), ("choose_trade", "invite")]
    answers = {
        _T.ACCEPT_TRADE: "accept",
        _T.REJECT_TRADE: "reject",
        _T.CANCEL_TRADE: "cancel_trade",
    }
    if t in answers:
        return [("choose_trade", answers[t])]
    if t == _T.CONFIRM_TRADE:
        return [("choose_trade", ("confirm", action.player))]
    raise ValueError(f"no click script for {action}")


def card_script_for(action: Action) -> Script:
    """The same development card play started by clicking the card in the hand bar."""
    assert action.type == _T.PLAY_DEV_CARD
    picks = [("choose", r) for r in (action.resource, action.resource2) if r >= 0]
    return [("play_card", action.dev), *picks]  # Year of Plenty and Monopoly ask resources


def drive(controller: Controller, script: Script) -> Action | None:
    """Run `script`; every call but the last must leave the action incomplete."""
    result = None
    for i, (method, arg) in enumerate(script):
        result = getattr(controller, method)(arg)
        if i < len(script) - 1:
            assert result is None, f"{method}({arg!r}) completed the action too early"
    return result
