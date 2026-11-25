"""Retrospective gameplay diagnostics. Never called by policies, rewards or live stepping.

Replay uses recorded chance outcomes and the engine's own legal sets/events. Hidden hands
are read only here, after evaluation, for offer feasibility and resource affordability.
"""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from carla.engine.actions import Action
from carla.engine.actions import ActionType as AT
from carla.engine.constants import CITY_COST, DEV_CARD_COST, ROAD_COST, SETTLEMENT_COST, DevCard
from carla.engine.events import EventKind as EK
from carla.engine.game import Game
from carla.engine.rules import can_afford
from carla.engine.state import SETUP_ORDER, Phase
from carla.envs.rewards import expected_production

PURCHASES = {
    "road": ROAD_COST,
    "settlement": SETTLEMENT_COST,
    "city": CITY_COST,
    "dev": DEV_CARD_COST,
}
CONSTRUCTION = {
    AT.BUILD_ROAD: "road",
    AT.BUILD_SETTLEMENT: "settlement",
    AT.BUILD_CITY: "city",
    AT.BUY_DEV_CARD: "dev",
}
BUILD_ACTIONS = {AT.BUILD_ROAD, AT.BUILD_SETTLEMENT, AT.BUILD_CITY}
DOMESTIC = {
    AT.PROPOSE_TRADE,
    AT.INVITE_TRADE,
    AT.ACCEPT_TRADE,
    AT.REJECT_TRADE,
    AT.CONFIRM_TRADE,
    AT.CANCEL_TRADE,
}
TRADING = DOMESTIC | {AT.MARITIME_TRADE}
COUNTS = (
    "decisions",
    "main_decisions",
    "domestic_trade_decisions",
    "maritime_trade_decisions",
    "voluntary_trade_initiations",
    "required_trade_responses",
    "trade_conclusions",
    "opening_proposals",
    "counteroffers",
    "invitations",
    "acceptances",
    "rejections",
    "confirmations",
    "cancellations",
    "negotiations",
    "completed_negotiations",
    "failed_negotiations",
    "negotiations_with_direct_acceptance",
    "completed_via_counteroffer",
    "all_opponents_rejected",
    "completed_trades_initiated",
    "completed_trades_as_partner",
    "offers",
    "cards_offered",
    "cards_requested",
    "multi_card_offers",
    "requests_at_least_twice_offer",
    "unaffordable_opening_offers",
    "repeated_failed_offers_same_turn",
    "repeated_identical_offers_across_turns",
    "same_turn_reverse_exchanges",
    "domestic_cards_given",
    "domestic_cards_received",
    "maritime_cards_given",
    "maritime_cards_received",
    "trades_with_ahead_public_leader",
    "setup_settlements",
    "setup_roads",
    "settlements",
    "cities",
    "roads",
    "free_roads",
    "purchased_roads",
    "dev_cards_bought",
    "robber_moves",
    "completed_turns",
    "turns_at_two_vp_or_less",
    "construction_possible_decisions",
    "construction_selected_when_possible",
    "trading_when_construction_possible",
    "end_turn_when_construction_possible",
    "settlement_when_settlement_legal",
    "city_when_city_legal",
    "road_when_settlement_legal",
    "road_when_city_legal",
    "dev_when_settlement_legal",
    "dev_when_city_legal",
    "trading_or_end_when_settlement_legal",
    "trading_or_end_when_city_legal",
    "settlement_or_city_legal_decisions",
    "trading_or_end_when_settlement_or_city_legal",
    "dice_resources",
    "starting_resources",
    "stolen_resources",
    "development_resources",
    *(f"{name}_legal_decisions" for name in PURCHASES),
    *(f"trades_unlocking_{name}" for name in PURCHASES),
    *(f"trades_removing_{name}" for name in PURCHASES),
    *(f"action_{kind.name.lower()}" for kind in AT),
    *(f"dev_played_{kind.name.lower()}" for kind in DevCard),
)


def ratio(numerator, denominator):
    """None is exported as an empty CSV cell, never a fictitious zero denominator."""
    return numerator / denominator if denominator else None


def rates(row: dict) -> dict:
    turns = row["completed_turns"]
    result = {
        "negotiation_completion_rate": ratio(row["completed_negotiations"], row["negotiations"]),
        "opening_direct_acceptance_rate": ratio(
            row["negotiations_with_direct_acceptance"], row["opening_proposals"]
        ),
        "mean_cards_offered": ratio(row["cards_offered"], row["offers"]),
        "mean_cards_requested": ratio(row["cards_requested"], row["offers"]),
        "multi_card_offer_rate": ratio(row["multi_card_offers"], row["offers"]),
        "construction_selection_rate": ratio(
            row["construction_selected_when_possible"], row["construction_possible_decisions"]
        ),
        "settlement_selection_rate": ratio(
            row["settlement_when_settlement_legal"], row["settlement_legal_decisions"]
        ),
        "city_selection_rate": ratio(row["city_when_city_legal"], row["city_legal_decisions"]),
        "trading_or_end_when_settlement_or_city_legal_rate": ratio(
            row["trading_or_end_when_settlement_or_city_legal"],
            row["settlement_or_city_legal_decisions"],
        ),
    }
    for name in (
        "settlements",
        "cities",
        "roads",
        "dev_cards_bought",
        "domestic_trade_decisions",
        "maritime_trade_decisions",
        "opening_proposals",
        "counteroffers",
        "invitations",
        "offers",
        "completed_trades_initiated",
        "completed_trades_as_partner",
        "dice_resources",
    ):
        result[f"{name}_per_player_turn"] = ratio(row[name], turns)
    return result


@dataclass
class GameplayDiagnostics:
    players: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    offers: list[dict] = field(default_factory=list)
    trajectories: list[dict] = field(default_factory=list)
    summary: list[dict] = field(default_factory=list)


class DiagnosticRecorder:
    """Observe engine transitions on a replay (also usable with hand-made test positions)."""

    def __init__(self, *, trajectories: bool = False):
        self.counts = [Counter(dict.fromkeys(COUNTS, 0)) for _ in range(4)]
        self.dice_by_resource = [[0] * 5 for _ in range(4)]
        self.trades: list[dict] = []
        self.offers: list[dict] = []
        self.trajectory: list[dict] = []
        self.keep_trajectories = trajectories
        self.setup_production: list[dict] | None = None
        self.production_sums = [[0.0, 0.0] for _ in range(4)]
        self.resource_production_sums = [[[0.0] * 5 for _ in range(2)] for _ in range(4)]
        self.production_samples = 0
        self.milestones = [
            dict.fromkeys(("first_settlement_turn", "first_city_turn"), None) for _ in range(4)
        ]
        self.negotiation: dict | None = None
        self.last_offer_turn: dict = {}
        self.failed_offers: set = set()
        self.exchanges: dict = defaultdict(set)

    def _sample(self, game: Game, point: str) -> list[dict]:
        result = []
        for seat in range(4):
            blocked = expected_production(game.state, seat)
            capacity = expected_production(game.state, seat, robber=False)
            row = {
                "point": point,
                "turn": game.turn,
                "player_seat": seat,
                "victory_points": game.victory_points(seat),
                "public_victory_points": game.victory_points(seat, include_hidden=False),
                "production_by_resource": blocked,
                "capacity_by_resource": capacity,
                "production": sum(blocked),
                "production_capacity": sum(capacity),
            }
            result.append(row)
            if point == "turn_end":
                self.production_sums[seat][0] += row["production"]
                self.production_sums[seat][1] += row["production_capacity"]
                for index, values in enumerate((blocked, capacity)):
                    self.resource_production_sums[seat][index] = [
                        a + b
                        for a, b in zip(
                            self.resource_production_sums[seat][index], values, strict=True
                        )
                    ]
            if self.keep_trajectories:
                self.trajectory.append(row)
        if point == "turn_end":
            self.production_samples += 1
        return result

    def _offer(self, game: Game, player: int, action: Action, opening: bool):
        offer = action.trade
        assert offer is not None
        counts = self.counts[player]
        given, requested = sum(offer.give), sum(offer.receive)
        counts.update(
            offers=1,
            cards_offered=given,
            cards_requested=requested,
            multi_card_offers=int(given > 1 or requested > 1),
            requests_at_least_twice_offer=int(requested >= 2 * given),
        )
        key = (player, offer)
        repeated = key in self.last_offer_turn and self.last_offer_turn[key] != game.turn
        counts["repeated_identical_offers_across_turns"] += int(repeated)
        self.last_offer_turn[key] = game.turn
        unaffordable = opening and not any(
            can_afford(game.state.hands[p], offer.receive) for p in range(4) if p != player
        )
        failed_repeat = (player, game.turn, offer) in self.failed_offers
        counts["unaffordable_opening_offers"] += int(unaffordable)
        counts["repeated_failed_offers_same_turn"] += int(failed_repeat)
        self.offers.append(
            {
                "turn": game.turn,
                "decision": game.num_steps + 1,
                "negotiation_decision": (
                    None if self.negotiation is None else self.negotiation["decision"]
                ),
                "player_seat": player,
                "offer_kind": "opening" if opening else "counteroffer",
                "target": None if opening else game.current_player,
                "give": offer.give,
                "receive": offer.receive,
                "no_opponent_can_afford": bool(unaffordable) if opening else None,
                "repeated_failed_same_turn": bool(failed_repeat),
                "repeated_across_turns": bool(repeated),
            }
        )

    def _close_negotiation(
        self, completed: bool, *, counter: bool = False, partner: int | None = None
    ):
        negotiation = self.negotiation
        if negotiation is None:
            return
        counts = self.counts[negotiation["owner"]]
        counts["completed_negotiations" if completed else "failed_negotiations"] += 1
        counts["completed_via_counteroffer"] += int(completed and counter)
        counts["all_opponents_rejected"] += int(
            negotiation["offer"] is not None and len(negotiation["rejects"]) == 3
        )
        if (not completed or counter) and negotiation["offer"] is not None:
            self.failed_offers.add(
                (negotiation["owner"], negotiation["turn"], negotiation["offer"])
            )
        for responder, offer in negotiation["counters"].items():
            if not (completed and counter and partner == responder):
                self.failed_offers.add((responder, negotiation["turn"], offer))
        self.negotiation = None

    def step(self, game: Game, action: Action, outcome=None):
        """Apply one recorded decision through the real engine and observe its actual events."""
        state = game.state
        player, phase, turn = game.decision_player, game.phase, game.turn
        counts = self.counts[player]
        counts["decisions"] += 1
        counts[f"action_{action.type.name.lower()}"] += 1
        counts["domestic_trade_decisions"] += int(action.type in DOMESTIC)
        counts["maritime_trade_decisions"] += int(action.type == AT.MARITIME_TRADE)
        counts["required_trade_responses"] += int(phase == Phase.TRADE_RESPONSE)
        counts["trade_conclusions"] += int(phase == Phase.TRADE_CONFIRM)
        if phase == Phase.MAIN:
            counts["main_decisions"] += 1
            legal = {a.type for a in game.legal_actions()}
            available = legal & BUILD_ACTIONS
            high_value = bool(legal & {AT.BUILD_SETTLEMENT, AT.BUILD_CITY})
            counts["settlement_or_city_legal_decisions"] += int(high_value)
            counts["trading_or_end_when_settlement_or_city_legal"] += int(
                high_value and (action.type in TRADING or action.type == AT.END_TURN)
            )
            for kind, name in CONSTRUCTION.items():
                counts[f"{name}_legal_decisions"] += int(kind in legal)
            counts["construction_possible_decisions"] += int(bool(available))
            counts["construction_selected_when_possible"] += int(
                bool(available) and action.type in BUILD_ACTIONS
            )
            counts["trading_when_construction_possible"] += int(
                bool(available) and action.type in TRADING
            )
            counts["end_turn_when_construction_possible"] += int(
                bool(available) and action.type == AT.END_TURN
            )
            for kind, name in ((AT.BUILD_SETTLEMENT, "settlement"), (AT.BUILD_CITY, "city")):
                if kind in legal:
                    counts[f"{name}_when_{name}_legal"] += int(action.type == kind)
                    counts[f"road_when_{name}_legal"] += int(action.type == AT.BUILD_ROAD)
                    counts[f"dev_when_{name}_legal"] += int(action.type == AT.BUY_DEV_CARD)
                    counts[f"trading_or_end_when_{name}_legal"] += int(
                        action.type in TRADING or action.type == AT.END_TURN
                    )
        opening = phase == Phase.MAIN and action.type in (AT.PROPOSE_TRADE, AT.INVITE_TRADE)
        if opening:
            counts["negotiations"] += 1
            counts["voluntary_trade_initiations"] += 1
            self.negotiation = {
                "owner": player,
                "decision": game.num_steps + 1,
                "offer": action.trade,
                "turn": turn,
                "accepts": set(),
                "rejects": set(),
                "counters": {},
            }
        if action.type == AT.PROPOSE_TRADE:
            counts["opening_proposals" if opening else "counteroffers"] += 1
            self._offer(game, player, action, opening)
            if not opening and self.negotiation is not None:
                self.negotiation["counters"][player] = action.trade
        for kind, name in (
            (AT.INVITE_TRADE, "invitations"),
            (AT.ACCEPT_TRADE, "acceptances"),
            (AT.REJECT_TRADE, "rejections"),
            (AT.CONFIRM_TRADE, "confirmations"),
            (AT.CANCEL_TRADE, "cancellations"),
        ):
            counts[name] += int(action.type == kind)
        if self.negotiation is not None:
            if action.type == AT.ACCEPT_TRADE:
                if not self.negotiation["accepts"]:
                    self.counts[self.negotiation["owner"]][
                        "negotiations_with_direct_acceptance"
                    ] += 1
                self.negotiation["accepts"].add(player)
            elif action.type == AT.REJECT_TRADE:
                self.negotiation["rejects"].add(player)
        hands = (
            [hand[:] for hand in state.hands]
            if action.type in (AT.CONFIRM_TRADE, AT.MARITIME_TRADE)
            else None
        )
        public_vp = [game.victory_points(p, include_hidden=False) for p in range(4)]
        through_counter = (
            action.type == AT.CONFIRM_TRADE and state.trade_counteroffers[action.player] is not None
        )
        vp_at_turn_end = game.victory_points(player) if action.type == AT.END_TURN else None
        events = game.step(action, outcome)
        for event in events:
            row = self.counts[event.player] if event.player >= 0 else None
            if event.kind == EK.SETTLEMENT_BUILT:
                setup = phase == Phase.SETUP_SETTLEMENT
                row["setup_settlements" if setup else "settlements"] += 1
                if not setup and self.milestones[event.player]["first_settlement_turn"] is None:
                    self.milestones[event.player]["first_settlement_turn"] = turn
            elif event.kind == EK.CITY_BUILT:
                row["cities"] += 1
                if self.milestones[event.player]["first_city_turn"] is None:
                    self.milestones[event.player]["first_city_turn"] = turn
            elif event.kind == EK.ROAD_BUILT:
                setup = phase == Phase.SETUP_ROAD
                row["setup_roads" if setup else "roads"] += 1
                if not setup:
                    row["free_roads" if phase == Phase.ROAD_BUILDING else "purchased_roads"] += 1
            elif event.kind == EK.DEV_CARD_BOUGHT:
                row["dev_cards_bought"] += 1
            elif event.kind == EK.DEV_CARD_PLAYED:
                row[f"dev_played_{DevCard(event.value).name.lower()}"] += 1
            elif event.kind == EK.ROBBER_MOVED:
                row["robber_moves"] += 1
            elif event.kind in (
                EK.RESOURCES_PRODUCED,
                EK.STARTING_RESOURCES,
                EK.YEAR_OF_PLENTY_TAKEN,
            ):
                key = {
                    EK.RESOURCES_PRODUCED: "dice_resources",
                    EK.STARTING_RESOURCES: "starting_resources",
                    EK.YEAR_OF_PLENTY_TAKEN: "development_resources",
                }[event.kind]
                row[key] += sum(event.counts)
                if event.kind == EK.RESOURCES_PRODUCED:
                    self.dice_by_resource[event.player] = [
                        a + b
                        for a, b in zip(
                            self.dice_by_resource[event.player], event.counts, strict=True
                        )
                    ]
            elif event.kind == EK.RESOURCE_STOLEN:
                row["stolen_resources"] += 1
            elif event.kind == EK.MONOPOLY_TAKEN:
                row["development_resources"] += sum(event.counts)
            elif event.kind in (EK.TRADE_EXECUTED, EK.MARITIME_TRADED):
                assert hands is not None
                domestic = event.kind == EK.TRADE_EXECUTED
                participants = (event.player, event.target) if domestic else (event.player,)
                if domestic:
                    self.counts[event.player]["completed_trades_initiated"] += 1
                    self.counts[event.target]["completed_trades_as_partner"] += 1
                for participant in participants:
                    delta = [
                        a - b
                        for a, b in zip(
                            game.state.hands[participant], hands[participant], strict=True
                        )
                    ]
                    give = tuple(max(-value, 0) for value in delta)
                    receive = tuple(max(value, 0) for value in delta)
                    stats = self.counts[participant]
                    prefix = "domestic" if domestic else "maritime"
                    stats[f"{prefix}_cards_given"] += sum(give)
                    stats[f"{prefix}_cards_received"] += sum(receive)
                    directions = (
                        tuple(r for r, value in enumerate(give) if value),
                        tuple(r for r, value in enumerate(receive) if value),
                    )
                    reverse = (directions[1], directions[0]) in self.exchanges[(participant, turn)]
                    stats["same_turn_reverse_exchanges"] += int(reverse)
                    self.exchanges[(participant, turn)].add(directions)
                    partner = event.target if participant == event.player else event.player
                    opponent_vp = [public_vp[p] for p in range(4) if p != participant]
                    ahead = (
                        domestic
                        and public_vp[partner] == max(opponent_vp)
                        and public_vp[partner] > public_vp[participant]
                    )
                    stats["trades_with_ahead_public_leader"] += int(ahead)
                    unlocked, removed = [], []
                    for name, cost in PURCHASES.items():
                        before, after = can_afford(hands[participant], cost), can_afford(
                            game.state.hands[participant], cost
                        )
                        stats[f"trades_unlocking_{name}"] += int(after and not before)
                        stats[f"trades_removing_{name}"] += int(before and not after)
                        if after and not before:
                            unlocked.append(name)
                        if before and not after:
                            removed.append(name)
                    self.trades.append(
                        {
                            "turn": turn,
                            "decision": game.num_steps,
                            "negotiation_decision": (
                                self.negotiation["decision"]
                                if domestic and self.negotiation is not None
                                else None
                            ),
                            "player_seat": participant,
                            "partner_seat": partner if domestic else None,
                            "trade_kind": prefix,
                            "give": give,
                            "receive": receive,
                            "via_counteroffer": bool(through_counter) if domestic else False,
                            "ahead_public_leader": bool(ahead),
                            "unlocked_affordability": unlocked,
                            "removed_affordability": removed,
                            "same_turn_construction_after": [],
                            "same_turn_dev_purchases_after": 0,
                        }
                    )
                if domestic:
                    self._close_negotiation(True, counter=through_counter, partner=event.target)
            elif event.kind == EK.TRADE_CANCELLED:
                self._close_negotiation(False)
            elif event.kind == EK.TURN_ENDED:
                row["completed_turns"] += 1
                row["turns_at_two_vp_or_less"] += int(vp_at_turn_end <= 2)
        if action.type in CONSTRUCTION and phase not in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            for exchange in reversed(self.trades):
                if exchange["turn"] != turn:
                    break
                if exchange["player_seat"] == player:
                    if action.type in BUILD_ACTIONS:
                        exchange["same_turn_construction_after"].append(CONSTRUCTION[action.type])
                    else:
                        exchange["same_turn_dev_purchases_after"] += 1
        if (
            self.setup_production is None
            and phase == Phase.SETUP_ROAD
            and game.state.setup_step == len(SETUP_ORDER)
        ):
            self.setup_production = self._sample(game, "setup_end")
        for seat in range(4):
            for vp in range(3, game.victory_points(seat) + 1):
                self.milestones[seat].setdefault(f"vp_{vp}_turn", turn)
        if any(event.kind == EK.TURN_ENDED for event in events):
            self._sample(game, "turn_end")
        return events

    def finish(self, game: Game) -> GameplayDiagnostics:
        if self.negotiation is not None:
            self._close_negotiation(False)
        final = self._sample(game, "terminal")
        players = []
        for seat in range(4):
            row = dict(self.counts[seat])
            row.update(self.milestones[seat])
            for vp in range(
                3, max(10, game.config.vp_to_win, *(game.victory_points(p) for p in range(4))) + 1
            ):
                row.setdefault(f"vp_{vp}_turn", None)
            initial = self.setup_production[seat] if self.setup_production is not None else None
            row.update(
                player_seat=seat,
                final_victory_points=game.victory_points(seat),
                setup_production=None if initial is None else initial["production"],
                setup_production_capacity=(
                    None if initial is None else initial["production_capacity"]
                ),
                setup_production_by_resource=(
                    None if initial is None else initial["production_by_resource"]
                ),
                setup_capacity_by_resource=(
                    None if initial is None else initial["capacity_by_resource"]
                ),
                final_production=final[seat]["production"],
                final_production_capacity=final[seat]["production_capacity"],
                final_production_by_resource=final[seat]["production_by_resource"],
                final_capacity_by_resource=final[seat]["capacity_by_resource"],
                mean_production=ratio(self.production_sums[seat][0], self.production_samples),
                mean_production_capacity=ratio(
                    self.production_sums[seat][1], self.production_samples
                ),
                mean_production_by_resource=[
                    ratio(value, self.production_samples)
                    for value in self.resource_production_sums[seat][0]
                ],
                mean_capacity_by_resource=[
                    ratio(value, self.production_samples)
                    for value in self.resource_production_sums[seat][1]
                ],
                production_samples=self.production_samples,
                dice_resources_by_resource=self.dice_by_resource[seat],
            )
            row.update(rates(row))
            players.append(row)
        return GameplayDiagnostics(players, self.trades, self.offers, self.trajectory)


def diagnose_game(game: Game, *, trajectories: bool = False) -> GameplayDiagnostics:
    """Replay and verify state AND RNG, without consulting or changing the live game RNG."""
    replay = Game(game.config, game.seed, game.board)
    recorder = DiagnosticRecorder(trajectories=trajectories)
    for step in game.history:
        if replay.decision_player != step.player:
            raise ValueError("diagnostic replay actor mismatch")
        recorder.step(replay, step.action, step.outcome)
    if replay.to_dict() != game.to_dict():
        raise ValueError("diagnostic replay did not reproduce the recorded final state/RNG")
    return recorder.finish(replay)


def summarize_diagnostics(players: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in players:
        groups[(row["role"], row["agent_type"])].append(row)
    result = []
    for (role, agent_type), rows in sorted(groups.items()):
        row = {"role": role, "agent_type": agent_type, "player_games": len(rows)}
        row.update(
            {
                key: rows[0][key]
                for key in ("checkpoint", "eval_seed", "opponent", "opponent_composition", "mode")
                if key in rows[0]
            }
        )
        row.update({name: sum(player[name] for player in rows) for name in COUNTS})
        row.update(rates(row))
        for name in (
            "final_victory_points",
            "setup_production",
            "setup_production_capacity",
            "final_production",
            "final_production_capacity",
            "mean_production",
            "mean_production_capacity",
        ):
            values = [player[name] for player in rows if player[name] is not None]
            row[f"mean_{name}"] = ratio(sum(values), len(values))
        for name in (
            key
            for key in dict.fromkeys(key for player in rows for key in player)
            if key.endswith("_turn") and (key.startswith("vp_") or key.startswith("first_"))
        ):
            values = [player[name] for player in rows if player.get(name) is not None]
            row[f"{name}_reached"] = len(values)
            row[f"mean_{name}"] = ratio(sum(values), len(values))
        result.append(row)
    return result


def export_diagnostics(diagnostics: GameplayDiagnostics, path: Path) -> list[Path]:
    """Create separate per-player/summary/trade/offer CSVs; never overwrite previous exports."""
    tables = {
        "": diagnostics.players,
        "_summary": diagnostics.summary,
        "_trades": diagnostics.trades,
        "_offers": diagnostics.offers,
    }
    if diagnostics.trajectories:
        tables["_trajectories"] = diagnostics.trajectories
    paths = [path.with_name(path.stem + suffix + path.suffix) for suffix in tables]
    if any(target.exists() for target in paths):
        raise ValueError("diagnostic export already exists; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, rows in zip(paths, tables.values(), strict=True):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if not fields:
            fields = [
                "checkpoint",
                "eval_seed",
                "board",
                "game_seed",
                "learner_seat",
                "player_seat",
                "role",
                "agent_type",
                "opponent_composition",
                "mode",
                "outcome",
                "winner",
                "game_turns",
                "game_steps",
                "turn",
                "give",
                "receive",
            ]
        with target.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, separators=(",", ":"))
                            if isinstance(value, dict | list | tuple)
                            else value
                        )
                        for key, value in row.items()
                    }
                )
    return paths
