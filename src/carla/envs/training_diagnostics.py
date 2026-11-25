"""Fixed-size, learner-only training counters from already-applied engine events.

This module deliberately does not import evaluation diagnostics, observations, policies, or
legality/production helpers. It is imported lazily when collection is enabled.
"""

from dataclasses import dataclass, field

from carla.engine.actions import ActionType as AT
from carla.engine.constants import RESOURCE_NAMES, DevCard
from carla.engine.events import EventKind as EK
from carla.engine.state import Phase

RESOURCE_SOURCES = (
    "gained_dice",
    "gained_setup",
    "gained_domestic_trade",
    "gained_maritime_trade",
    "gained_robber_theft",
    "gained_development",
    "lost_discard",
    "lost_robber_theft",
    "lost_domestic_trade",
    "lost_maritime_trade",
    "lost_development",
)
SCALAR_FIELDS = (
    "diagnostic_learner_decisions",
    "main_phase_decisions",
    "completed_learner_turns",
    "learner_turns_at_two_vp_or_less",
    "domestic_trade_decisions",
    "opening_trade_proposals",
    "counteroffers",
    "trade_invitations",
    "trade_acceptances",
    "trade_rejections",
    "trade_confirmations",
    "trade_cancellations",
    "negotiations_opened",
    "negotiations_completed",
    "domestic_trades_as_partner",
    "maritime_trades",
    "setup_settlements",
    "setup_roads",
    "settlements_built",
    "cities_built",
    "roads_built",
    "free_roads_built",
    "purchased_roads",
    "dev_cards_purchased",
    "robber_moves",
)
TRAINING_COUNTER_FIELDS = (
    "diagnostics_complete",
    *SCALAR_FIELDS,
    *(f"action_{kind.name.lower()}" for kind in AT),
    *(f"dev_played_{kind.name.lower()}" for kind in DevCard),
    *(
        name
        for source in RESOURCE_SOURCES
        for name in (
            f"cards_{source}",
            *(f"cards_{source}_{resource}" for resource in RESOURCE_NAMES),
        )
    ),
    "final_victory_points",
    "first_settlement_learner_turn",
    "first_city_learner_turn",
    *(f"vp_{vp}_learner_turn" for vp in range(3, 11)),
)
_DOMESTIC = frozenset(
    (
        AT.PROPOSE_TRADE,
        AT.INVITE_TRADE,
        AT.ACCEPT_TRADE,
        AT.REJECT_TRADE,
        AT.CONFIRM_TRADE,
        AT.CANCEL_TRADE,
    )
)
_AWARDS = frozenset((EK.LONGEST_ROAD_CHANGED, EK.LARGEST_ARMY_CHANGED))
_CROSS_PLAYER = frozenset((EK.TRADE_EXECUTED, EK.RESOURCE_STOLEN, EK.MONOPOLY_TAKEN, *_AWARDS))
_UNCOUNTED_EVENTS = frozenset(
    (
        EK.DICE_ROLLED,
        EK.GAME_OVER,
        EK.TRADE_PROPOSED,
        EK.TRADE_RESPONDED,
        EK.TRADE_CANCELLED,
        EK.TRADE_INVITED,
        EK.TRADE_COUNTERED,
    )
)


@dataclass(slots=True)
class TrainingDiagnostics:
    learner: int
    vp: int = 0
    complete: bool = True
    phase: Phase = Phase.SETUP_SETTLEMENT
    diagnostic_learner_decisions: int = 0
    main_phase_decisions: int = 0
    completed_learner_turns: int = 0
    learner_turns_at_two_vp_or_less: int = 0
    domestic_trade_decisions: int = 0
    opening_trade_proposals: int = 0
    counteroffers: int = 0
    trade_invitations: int = 0
    trade_acceptances: int = 0
    trade_rejections: int = 0
    trade_confirmations: int = 0
    trade_cancellations: int = 0
    negotiations_opened: int = 0
    negotiations_completed: int = 0
    domestic_trades_as_partner: int = 0
    maritime_trades: int = 0
    setup_settlements: int = 0
    setup_roads: int = 0
    settlements_built: int = 0
    cities_built: int = 0
    roads_built: int = 0
    free_roads_built: int = 0
    purchased_roads: int = 0
    dev_cards_purchased: int = 0
    robber_moves: int = 0
    actions: list[int] = field(default_factory=lambda: [0] * len(AT))
    dev_played: list[int] = field(default_factory=lambda: [0] * len(DevCard))
    resources: list[list[int]] = field(default_factory=lambda: [[0] * 5 for _ in RESOURCE_SOURCES])
    first_settlement_learner_turn: int | None = None
    first_city_learner_turn: int | None = None
    milestones: list[int | None] = field(default_factory=lambda: [None] * 8)

    def action_applied(self, actor: int, phase: Phase, action) -> None:
        if actor != self.learner:
            return
        kind, self.phase = action.type, phase
        self.actions[kind] += 1
        self.diagnostic_learner_decisions += 1
        self.main_phase_decisions += self.phase == Phase.MAIN
        self.domestic_trade_decisions += kind in _DOMESTIC
        if kind == AT.PROPOSE_TRADE:
            if self.phase == Phase.MAIN:
                self.opening_trade_proposals += 1
                self.negotiations_opened += 1
            else:
                self.counteroffers += 1
        elif kind == AT.INVITE_TRADE:
            self.trade_invitations += 1
            self.negotiations_opened += 1
        elif kind == AT.ACCEPT_TRADE:
            self.trade_acceptances += 1
        elif kind == AT.REJECT_TRADE:
            self.trade_rejections += 1
        elif kind == AT.CONFIRM_TRADE:
            self.trade_confirmations += 1
        elif kind == AT.CANCEL_TRADE:
            self.trade_cancellations += 1

    def _add(self, source: int, counts) -> None:
        # Avoid zip/generator/list allocations in this five-resource hot path.
        values = self.resources[source]
        values[0] += counts[0]
        values[1] += counts[1]
        values[2] += counts[2]
        values[3] += counts[3]
        values[4] += counts[4]

    def after_step(self, game, events) -> None:
        update_vp = False
        learner = self.learner
        for event in events:
            kind = event.kind
            if kind in _UNCOUNTED_EVENTS:
                continue  # Negotiation decisions were already counted with their action.
            if event.player != learner and kind not in _CROSS_PLAYER:
                continue
            if kind == EK.TRADE_EXECUTED:
                if event.player == learner:
                    self.negotiations_completed += 1
                    self._add(2, event.receive)
                    self._add(8, event.give)
                elif event.target == learner:
                    self.domestic_trades_as_partner += 1
                    self._add(2, event.give)
                    self._add(8, event.receive)
            elif kind == EK.RESOURCE_STOLEN:
                if event.player == learner:
                    self.resources[4][event.value] += 1
                elif event.target == learner:
                    self.resources[7][event.value] += 1
            elif kind == EK.MONOPOLY_TAKEN:
                if event.player == learner:
                    self.resources[5][event.value] += sum(event.counts)
                else:
                    self.resources[10][event.value] += event.counts[learner]
            elif kind in _AWARDS:
                update_vp |= event.player == learner or event.target == learner
            elif event.player == learner:
                if kind == EK.RESOURCES_PRODUCED:
                    self._add(0, event.counts)
                elif kind == EK.STARTING_RESOURCES:
                    self._add(1, event.counts)
                elif kind == EK.YEAR_OF_PLENTY_TAKEN:
                    self._add(5, event.counts)
                elif kind == EK.MARITIME_TRADED:
                    self.maritime_trades += 1
                    given, received = self.resources[9], self.resources[3]
                    for resource, count in enumerate(event.counts):
                        if count > 0:
                            received[resource] += count
                        else:
                            given[resource] -= count
                elif kind == EK.DISCARDED:
                    self.resources[6][event.value] += 1
                elif kind == EK.SETTLEMENT_BUILT:
                    update_vp = True
                    if self.phase == Phase.SETUP_SETTLEMENT:
                        self.setup_settlements += 1
                    else:
                        self.settlements_built += 1
                        if self.first_settlement_learner_turn is None:
                            self.first_settlement_learner_turn = self.completed_learner_turns
                elif kind == EK.CITY_BUILT:
                    update_vp = True
                    self.cities_built += 1
                    if self.first_city_learner_turn is None:
                        self.first_city_learner_turn = self.completed_learner_turns
                elif kind == EK.ROAD_BUILT:
                    if self.phase == Phase.SETUP_ROAD:
                        self.setup_roads += 1
                    else:
                        self.roads_built += 1
                        if self.phase == Phase.ROAD_BUILDING:
                            self.free_roads_built += 1
                        else:
                            self.purchased_roads += 1
                elif kind == EK.DEV_CARD_BOUGHT:
                    self.dev_cards_purchased += 1
                    update_vp |= event.value == DevCard.VICTORY_POINT
                elif kind == EK.DEV_CARD_PLAYED:
                    self.dev_played[event.value] += 1
                elif kind == EK.ROBBER_MOVED:
                    self.robber_moves += 1
                elif kind == EK.TURN_ENDED:
                    self.learner_turns_at_two_vp_or_less += self.vp <= 2
                    self.completed_learner_turns += 1
        if update_vp:
            self.vp = game.victory_points(learner)
            if self.complete:
                for vp in range(3, min(self.vp, 10) + 1):
                    if self.milestones[vp - 3] is None:
                        self.milestones[vp - 3] = self.completed_learner_turns

    def record(self) -> dict:
        result = {name: getattr(self, name) for name in SCALAR_FIELDS}
        result["diagnostics_complete"] = self.complete
        result.update((f"action_{kind.name.lower()}", self.actions[kind]) for kind in AT)
        result.update(
            (f"dev_played_{kind.name.lower()}", self.dev_played[kind]) for kind in DevCard
        )
        for source, values in zip(RESOURCE_SOURCES, self.resources, strict=True):
            result[f"cards_{source}"] = sum(values)
            result.update(
                (f"cards_{source}_{resource}", value)
                for resource, value in zip(RESOURCE_NAMES, values, strict=True)
            )
        result.update(
            final_victory_points=self.vp,
            first_settlement_learner_turn=(
                self.first_settlement_learner_turn if self.complete else None
            ),
            first_city_learner_turn=self.first_city_learner_turn if self.complete else None,
        )
        result.update(
            (f"vp_{vp}_learner_turn", self.milestones[vp - 3] if self.complete else None)
            for vp in range(3, 11)
        )
        return result
