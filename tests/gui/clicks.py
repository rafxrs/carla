"""Click scripts: the GUI input that should produce a given legal action through the controller."""

from collections.abc import Hashable, Sequence

from carla.engine.actions import Action, ActionType
from carla.engine.constants import DevCard
from carla.gui.controller import Controller

_T = ActionType
Script = list[tuple[str, Hashable]]


def script_for(action: Action, legal: Sequence[Action]) -> Script:
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
        return [("press", "bank_trade"), ("choose", action.resource), ("choose", action.resource2)]
    if t == _T.PLAY_DEV_CARD:
        script: Script = [("press", "play_dev"), ("choose", action.dev)]
        if action.dev == DevCard.YEAR_OF_PLENTY:
            script += [("choose", action.resource), ("choose", action.resource2)]
        elif action.dev == DevCard.MONOPOLY:
            script.append(("choose", action.resource))
        return script
    answers = {_T.ACCEPT_TRADE: "accept", _T.REJECT_TRADE: "reject", _T.CANCEL_TRADE: "cancel"}
    if t in answers:
        return [("choose", answers[t])]
    if t == _T.CONFIRM_TRADE:
        return [("choose", action.player)]
    raise ValueError(f"no click script for {action}")


def drive(controller: Controller, script: Script) -> Action | None:
    """Run `script`; every call but the last must leave the action incomplete."""
    result = None
    for i, (method, arg) in enumerate(script):
        result = getattr(controller, method)(arg)
        if i < len(script) - 1:
            assert result is None, f"{method}({arg!r}) completed the action too early"
    return result
