"""Human-readable descriptions of actions, events and positions."""

from carla.engine.actions import Action, ActionType
from carla.engine.board import Board, describe_port
from carla.engine.constants import (
    DEV_CARD_NAMES,
    NUM_PLAYERS,
    NUM_RESOURCES,
    RESOURCE_NAMES,
    Building,
    DevCard,
)
from carla.engine.events import Event, EventKind
from carla.engine.game import Game
from carla.engine.rules import victory_points
from carla.engine.topology import NUM_CORNERS, NUM_EDGES, PORT_EDGES

_SHORT = ("L", "B", "W", "G", "O")
_DEV_SHORT = ("K", "RB", "YP", "M", "VP")


def seat(p: int) -> str:
    return f"P{p}"


def resources_text(counts: tuple[int, ...] | list[int]) -> str:
    parts = [f"{n} {RESOURCE_NAMES[r]}" for r, n in enumerate(counts) if n]
    return ", ".join(parts) if parts else "nothing"


def describe_action(action: Action, board: Board | None = None) -> str:
    t = action.type
    if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
        where = board.describe_corner(action.corner) if board else f"corner {action.corner}"
        return f"{t.name} {where}"
    if t == ActionType.BUILD_ROAD:
        return f"BUILD_ROAD {board.describe_edge(action.edge) if board else action.edge}"
    if t == ActionType.MOVE_ROBBER:
        where = board.describe_tile(action.tile) if board else ""
        victim = f", steal from {seat(action.player)}" if action.player >= 0 else ""
        return f"MOVE_ROBBER tile {action.tile} ({where}){victim}"
    if t == ActionType.PLAY_DEV_CARD:
        extra = ""
        if action.dev == DevCard.YEAR_OF_PLENTY:
            extra = f" ({RESOURCE_NAMES[action.resource]}, {RESOURCE_NAMES[action.resource2]})"
        elif action.dev == DevCard.MONOPOLY:
            extra = f" ({RESOURCE_NAMES[action.resource]})"
        return f"PLAY {DEV_CARD_NAMES[action.dev]}{extra}"
    if t == ActionType.DISCARD:
        return f"DISCARD {RESOURCE_NAMES[action.resource]}"
    if t == ActionType.MARITIME_TRADE:
        return f"MARITIME_TRADE {RESOURCE_NAMES[action.resource]} -> {RESOURCE_NAMES[action.resource2]}"
    if t == ActionType.CONFIRM_TRADE:
        return f"CONFIRM_TRADE with {seat(action.player)}"
    if t == ActionType.PROPOSE_TRADE and action.trade is not None:
        give, receive = resources_text(action.trade.give), resources_text(action.trade.receive)
        return f"PROPOSE_TRADE give {give} for {receive}"
    return t.name


def describe_event(event: Event, board: Board) -> str:
    """One line per event. Hidden values (-1, see `Event.visible_to`) are shown as such."""
    k, p, v = event.kind, seat(event.player), event.value
    if k == EventKind.DICE_ROLLED:
        d1, d2 = event.counts
        return f"{p} rolls {d1}+{d2} = {v}"
    if k == EventKind.RESOURCES_PRODUCED:
        return f"{p} receives {resources_text(event.counts)}"
    if k == EventKind.STARTING_RESOURCES:
        return f"{p} takes starting resources: {resources_text(event.counts)}"
    if k == EventKind.SETTLEMENT_BUILT:
        return f"{p} builds a settlement at {board.describe_corner(event.location)}"
    if k == EventKind.CITY_BUILT:
        return f"{p} builds a city at {board.describe_corner(event.location)}"
    if k == EventKind.ROAD_BUILT:
        return f"{p} builds a road on {board.describe_edge(event.location)}"
    if k == EventKind.DEV_CARD_BOUGHT:
        card = DEV_CARD_NAMES[v] if v >= 0 else "hidden"
        return f"{p} buys a development card ({card})"
    if k == EventKind.DEV_CARD_PLAYED:
        return f"{p} plays {DEV_CARD_NAMES[v]}"
    if k == EventKind.ROBBER_MOVED:
        return (
            f"{p} moves the robber to tile {event.location} ({board.describe_tile(event.location)})"
        )
    if k == EventKind.RESOURCE_STOLEN:
        what = RESOURCE_NAMES[v] if v >= 0 else "a card"
        return f"{p} steals {what} from {seat(event.target)}"
    if k == EventKind.DISCARDED:
        return f"{p} discards 1 {RESOURCE_NAMES[v]}"
    if k == EventKind.MARITIME_TRADED:
        give = next(r for r, n in enumerate(event.counts) if n < 0)
        get = next(r for r, n in enumerate(event.counts) if n > 0)
        return (
            f"{p} trades {-event.counts[give]} {RESOURCE_NAMES[give]} "
            f"for 1 {RESOURCE_NAMES[get]} with the bank"
        )
    if k == EventKind.YEAR_OF_PLENTY_TAKEN:
        return f"{p} takes {resources_text(event.counts)} from the bank"
    if k == EventKind.MONOPOLY_TAKEN:
        detail = ", ".join(f"{seat(q)}: {n}" for q, n in enumerate(event.counts) if n)
        return f"{p} claims {sum(event.counts)} {RESOURCE_NAMES[v]}" + (
            f" ({detail})" if detail else ""
        )
    if k in (EventKind.LONGEST_ROAD_CHANGED, EventKind.LARGEST_ARMY_CHANGED):
        title = "Longest Road" if k == EventKind.LONGEST_ROAD_CHANGED else "Largest Army"
        holder = f"goes to {p}" if event.player >= 0 else "is set aside"
        previous = f" (was {seat(event.target)})" if event.target >= 0 else ""
        return f"{title} {holder}{previous}"
    if k == EventKind.TURN_ENDED:
        return f"{p} ends the turn"
    if k == EventKind.GAME_OVER:
        return f"{p} wins!" if event.player >= 0 else "Game stopped at the turn limit (no winner)"
    return f"{k.name} {event}"


def describe_board(board: Board) -> str:
    tiles = " | ".join(f"{t} {board.describe_tile(t)}" for t in range(len(board.tile_resource)))
    ports = " | ".join(
        f"{board.describe_edge(e)} {describe_port(kind)}"
        for e, kind in zip(PORT_EDGES, board.port_types, strict=True)
    )
    return f"tiles: {tiles}\nports: {ports}"


def describe_state(game: Game) -> str:
    """Multi-line summary of a position, with every hidden card shown."""
    s = game.state
    actor = seat(s.decision_player) if s.decision_player >= 0 else "nobody"
    lines = [
        f"step {game.num_steps} | turn {s.turn} | {s.phase.name} | {actor} to act "
        f"(turn of {seat(s.current_player)}) | last roll {s.last_roll or '-'}",
        f"robber on tile {s.robber_tile} ({s.board.describe_tile(s.robber_tile)}) | "
        f"bank {' '.join(f'{_SHORT[r]}{s.bank[r]}' for r in range(NUM_RESOURCES))} | "
        f"dev deck {sum(s.dev_deck)}",
        "seat  VP(public)  hand                 dev held / played      left S/C/R  road",
    ]
    for p in range(NUM_PLAYERS):
        hand = " ".join(f"{_SHORT[r]}{n}" for r, n in enumerate(s.hands[p]))
        held = " ".join(f"{_DEV_SHORT[d]}{n}" for d, n in enumerate(s.dev_hands[p]) if n) or "-"
        played = " ".join(f"{_DEV_SHORT[d]}{n}" for d, n in enumerate(s.dev_played[p]) if n) or "-"
        titles = ("LR " if s.longest_road_owner == p else "") + (
            "LA" if s.largest_army_owner == p else ""
        )
        lines.append(
            f"{seat(p):<5} {victory_points(s, p):>2} ({victory_points(s, p, False):>2})     "
            f"{hand:<20} {held + ' / ' + played:<22} "
            f"{s.settlements_left[p]}/{s.cities_left[p]}/{s.roads_left[p]:<7} "
            f"{s.road_lengths[p]:>2} {titles}"
        )
    for p in range(NUM_PLAYERS):
        settlements = [
            c
            for c in range(NUM_CORNERS)
            if s.corner_owner[c] == p and s.corner_building[c] == Building.SETTLEMENT
        ]
        cities = [
            c
            for c in range(NUM_CORNERS)
            if s.corner_owner[c] == p and s.corner_building[c] == Building.CITY
        ]
        roads = [e for e in range(NUM_EDGES) if s.edge_owner[e] == p]
        lines.append(f"{seat(p)}: settlements {settlements} cities {cities} roads {roads}")
    return "\n".join(lines)
