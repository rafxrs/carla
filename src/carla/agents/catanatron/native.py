"""CARLA implementations of Catanatron's published value formula and expectiminimax.

Formula derived from GPL-3.0-or-later Catanatron ecf9311 (Bryan Collazo).
Search uses CARLA rules. `chance="exact"` (default) takes exact expectations, a better
defined variant. `chance="upstream"` reproduces the pinned originals' chance handling:

- F scores each root move on ONE outcome, drawn as upstream draws it: dice as two
  `randint(1, 6)`, a steal as `choice` over the victim's cards (wood, brick, sheep, wheat,
  ore), a development card from the top of the deck. Hosted on Catanatron's engine, the
  draws come from Catanatron's own `random` (`chance_rng`), so they match exactly.
- AB (depth >= 2) uses upstream's `execute_spectrum`, quirks included. Its 11 roll branches
  execute without an action record, so each rolls RANDOM dice (two `randint` draws) and is
  weighted by its intended sum's probability; steals are 1/5 per resource type, and a type
  the victim lacks leaves the position unchanged (the swallowed exception); every dev-card
  branch draws the deck's top card. Every move's spectrum at a node is expanded before any
  is searched, which fixes the order of RNG draws. Search is upstream's alpha-beta, which
  passes the same bounds into chance children, and stops at any player's 10 actual VP.

Without a known deck order (`chance_deck`), dev-card draws are sampled (F) or averaged (AB)
over the deck composition, and inner-node move order is CARLA's, so upstream parity is then
distributional rather than exact.
"""

import math

from carla.agents.api import AgentView
from carla.agents.base import Agent
from carla.engine.actions import CANCEL_TRADE, REJECT_TRADE, ActionType
from carla.engine.constants import PIPS
from carla.engine.rng import Rng, derive_seed
from carla.engine.rules import distance_rule_ok
from carla.engine.state import Phase
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS, CORNER_TILES, EDGE_CORNERS
from carla.planning import sample_world
from carla.planning.chance import successors

WEIGHTS = {
    "public_vps": 3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    "reachable_production_0": 0,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10,
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
}


def features(game, seat):
    s = game.state
    owned = {c for c, p in enumerate(s.corner_owner) if p == seat}
    zero = set(owned)
    for e, p in enumerate(s.edge_owner):
        if p == seat:
            zero.update(EDGE_CORNERS[e])
    available = {c for c in range(54) if distance_rule_ok(s, c)}
    buildable = len(zero & available)
    layer = set(zero)
    for c in zero:
        if s.corner_owner[c] in (-1, seat):
            for e, n in zip(CORNER_EDGES[c], CORNER_NEIGHBORS[c], strict=True):
                if s.edge_owner[e] in (-1, seat):
                    layer.add(n)
    reachable = (owned | available) & layer

    def production(p):
        rates = [
            sum(
                s.corner_building[c] * PIPS[s.board.tile_number[t]] / 36
                for c, owner in enumerate(s.corner_owner)
                if owner == p
                for t in CORNER_TILES[c]
                if s.board.tile_resource[t] == r and t != s.robber_tile
            )
            for r in range(5)
        ]
        return sum(rates) + (4 * 0.02778 * sum(n > 0 for n in rates) if p == seat else 0)

    h = s.hands[seat]
    city = (max(2 - h[3], 0) + max(3 - h[4], 0)) / 5
    settlement = sum(max(1 - h[r], 0) for r in range(4)) / 4
    return {
        "public_vps": game.victory_points(seat, include_hidden=False),
        "production": production(seat),
        "enemy_production": production((seat + 1) % 4),
        "num_tiles": len({t for c in owned for t in CORNER_TILES[c]}),
        "reachable_production_0": 0,
        "reachable_production_1": sum(
            PIPS[s.board.tile_number[t]] / 36
            for c in reachable
            for t in CORNER_TILES[c]
            if s.board.tile_resource[t] >= 0
        ),
        "buildable_nodes": buildable,
        "longest_road": s.road_lengths[seat],
        "hand_synergy": (2 - city - settlement) / 2,
        "hand_resources": sum(h),
        "discard_penalty": int(sum(h) > 7),
        "hand_devs": sum(s.dev_hands[seat]),
        "army_size": s.dev_played[seat][0],
        "_reachable": reachable,
    }


def value(game, seat):
    f = features(game, seat)
    weights = dict(WEIGHTS)
    if f["buildable_nodes"]:
        weights["longest_road"] = 0.1
    return sum(f[k] * weights[k] for k in WEIGHTS)


def _upstream_dice():
    """map.build_dice_probas: 1/36 added once per dice pair (so P(7) != 6/36 exactly)."""
    probas = [0.0] * 13
    for i in range(1, 7):
        for j in range(1, 7):
            probas[i + j] += 1 / 36
    return tuple(probas)


UPSTREAM_DICE = _upstream_dice()
PRODUCTION_ORDER = (3, 4, 2, 0, 1)
"""value_production's order: wheat, ore, sheep, wood, brick (CARLA resource ids)."""
PROBA_POINT = 2.778 / 100
TRANSLATE_VARIETY = 4


def _catanatron_order(corners):
    """Corners in Catanatron node-id order, approximating its int-set iteration."""
    try:
        from carla.agents.catanatron.coordinates import mappings

        nodes = mappings()[1]
    except ImportError:
        return sorted(corners)
    return sorted(corners, key=nodes.__getitem__)


def _node_production(s, corner, resource, robber):
    return sum(
        UPSTREAM_DICE[s.board.tile_number[t]]
        for t in CORNER_TILES[corner]
        if s.board.tile_resource[t] == resource and (not robber or t != s.robber_tile)
    )


def _upstream_production(s, player, variety):
    """build_production_features(True) + value_production, in upstream's float order."""
    produced = [0.0] * 5
    owned = [c for c, owner in enumerate(s.corner_owner) if owner == player]
    for r in range(5):
        production = 0
        for c in owned:
            if s.corner_building[c] == 1:
                production += _node_production(s, c, r, True)
        for c in owned:
            if s.corner_building[c] == 2:
                production += 2 * _node_production(s, c, r, True)
        produced[r] = production
    total = sum([produced[r] for r in PRODUCTION_ORDER])
    if not variety:
        return total
    return (
        total + sum([produced[r] != 0 for r in PRODUCTION_ORDER]) * TRANSLATE_VARIETY * PROBA_POINT
    )


def value_upstream(game, seat):
    """base_fn with upstream's constants, grouping and summation order. The features are
    those of `features`; only the float arithmetic is arranged to match the original."""
    s = game.state
    f = features(game, seat)
    reachable = f["_reachable"]
    reach1 = sum(
        [
            sum(
                UPSTREAM_DICE[s.board.tile_number[t]]
                for t in CORNER_TILES[c]
                if s.board.tile_resource[t] == r
            )
            for r in range(5)
            for c in _catanatron_order(reachable)
        ]
    )
    factor = WEIGHTS["longest_road"] if f["buildable_nodes"] == 0 else 0.1
    return float(
        f["public_vps"] * WEIGHTS["public_vps"]
        + _upstream_production(s, seat, True) * WEIGHTS["production"]
        + _upstream_production(s, (seat + 1) % 4, False) * WEIGHTS["enemy_production"]
        + 0 * WEIGHTS["reachable_production_0"]
        + reach1 * WEIGHTS["reachable_production_1"]
        + f["hand_synergy"] * WEIGHTS["hand_synergy"]
        + f["buildable_nodes"] * WEIGHTS["buildable_nodes"]
        + f["num_tiles"] * WEIGHTS["num_tiles"]
        + f["hand_resources"] * WEIGHTS["hand_resources"]
        + (WEIGHTS["discard_penalty"] if f["discard_penalty"] else 0)
        + f["longest_road"] * factor
        + f["hand_devs"] * WEIGHTS["hand_devs"]
        + f["army_size"] * WEIGHTS["army_size"]
    )


class NativeCatanatronAgent(Agent):
    def __init__(self, depth=1, node_budget=10000, information="sampled", chance="exact"):
        """`information`: "sampled" plans in an inventory-constrained world (the CARLA referee
        boundary); "open" plans on the given game itself, for comparisons with the originals on
        Catanatron's engine, whose bots see every hand (`host.carla_game(..., "open")`).
        `chance`: "exact" or "upstream" (see the module docstring)."""
        if depth < 1 or node_budget < 1:
            raise ValueError("depth and node budget must be positive")
        if information not in ("sampled", "open"):
            raise ValueError("information must be 'sampled' or 'open'")
        if chance not in ("exact", "upstream"):
            raise ValueError("chance must be 'exact' or 'upstream'")
        self.depth, self.node_budget, self.information = depth, node_budget, information
        self.chance = chance
        self.chance_rng = None
        """Upstream mode: the generator to draw from (hosted: Catanatron's `random`)."""
        self.chance_deck = None
        """Upstream mode: development cards in draw order, top first, when known."""
        base = "native-f" if depth == 1 else f"native-ab{depth}"
        self.name = base if chance == "exact" else f"{base}-upstream"

    def reset(self, game, seat, seed):
        self.seed = seed
        self.nodes = self.budget_exhaustions = 0

    @staticmethod
    def candidates(game):
        return [
            a
            for a in game.legal_actions()
            if a.type not in (ActionType.PROPOSE_TRADE, ActionType.INVITE_TRADE)
        ]

    def _search(self, game, seat, depth):
        if depth == 0 or game.is_over or self.nodes >= self.node_budget:
            return value(game, seat)
        values = []
        for action in self.candidates(game):
            if self.nodes >= self.node_budget:
                break
            values.append(self._score(game, action, seat, depth))
        return (
            (max(values) if game.decision_player == seat else min(values))
            if values
            else value(game, seat)
        )

    def _score(self, game, action, seat, depth):
        self.nodes += 1
        return sum(
            probability * self._search(child, seat, depth - 1)
            for child, probability in successors(game, action)
        )

    # ----------------------------------------------------------- upstream chance handling

    @staticmethod
    def _draw_dev(game, rng, deck):
        if deck:
            return deck[0], deck[1:]
        cards = [c for c, n in enumerate(game.state.dev_deck) for _ in range(n)]
        return (rng.choice(cards) if cards else None), None

    def _sampled(self, world, action, seat, rng, deck):
        """ValueFunctionPlayer: `game.copy().execute(action)` on one drawn outcome."""
        outcome = None
        if action.type == ActionType.ROLL_DICE:
            outcome = (rng.randint(1, 6), rng.randint(1, 6))
        elif action.type == ActionType.MOVE_ROBBER and action.player >= 0:
            hand = world.state.hands[action.player]
            cards = [r for r in range(5) for _ in range(hand[r])]
            outcome = rng.choice(cards) if cards else None
        elif action.type == ActionType.BUY_DEV_CARD:
            outcome, _ = self._draw_dev(world, rng, deck)
        child = world.clone()
        child.step(action, outcome)
        return value_upstream(child, seat)

    def _spectrum(self, game, action, deck, rng):
        """execute_spectrum: (child, remaining deck, probability) per upstream outcome."""

        def child(outcome=None):
            new = game.clone()
            new.step(action, outcome)
            return new

        if action.type == ActionType.ROLL_DICE:
            # The requested dice are ignored upstream: each branch rolls afresh.
            return [
                (child((rng.randint(1, 6), rng.randint(1, 6))), deck, UPSTREAM_DICE[n])
                for n in range(2, 13)
            ]
        if action.type == ActionType.BUY_DEV_CARD:
            if deck:  # Every upstream branch pops the same top card.
                return [(child(deck[0]), deck[1:], 1.0)]
            return [(c, None, p) for c, p in successors(game, action)]
        if action.type == ActionType.MOVE_ROBBER and action.player >= 0:
            hand = game.state.hands[action.player]
            if sum(hand):
                # A type the victim lacks raises before the robber moves: unchanged position.
                return [(child(r) if hand[r] else game.clone(), deck, 1 / 5.0) for r in range(5)]
        return [(child(), deck, 1.0)]

    @staticmethod
    def _upstream_order(actions):
        """generate_playable_actions' order where it does not depend on history or hashing:
        dev-card plays, roll or end turn, roads, settlements, cities, purchase, maritime
        trades; settlements and roads by Catanatron node id."""
        try:
            from carla.agents.catanatron.coordinates import mappings

            _, nodes, _, edges = mappings()
        except ImportError:  # Catanatron absent: CARLA's order within each type.
            nodes = edges = None
        dev_rank = {2: 0, 3: 1, 0: 2, 1: 3}  # Year of Plenty, Monopoly, Knight, Road Building

        def key(a):
            t = a.type
            if t == ActionType.PLAY_DEV_CARD:
                return (dev_rank.get(int(a.dev), 4), a.resource, a.resource2)
            rank = {
                ActionType.ROLL_DICE: 5,
                ActionType.END_TURN: 5,
                ActionType.BUILD_ROAD: 6,
                ActionType.BUILD_SETTLEMENT: 7,
                ActionType.BUILD_CITY: 8,
                ActionType.BUY_DEV_CARD: 9,
                ActionType.MARITIME_TRADE: 10,
            }.get(t, 11)
            if t == ActionType.BUILD_ROAD and edges is not None:
                return (rank, edges[a.edge])
            if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY) and nodes is not None:
                return (rank, nodes[a.corner])
            return (rank, a.resource, a.resource2, a.tile, a.player)

        return sorted(actions, key=key)

    @staticmethod
    def _upstream_over(game):
        need = game.state.config.vp_to_win
        return any(game.victory_points(p) >= need for p in range(4))

    def _alphabeta(self, game, deck, seat, depth, alpha, beta, rng):
        """AlphaBetaPlayer.alphabeta, bounds passed into chance children as upstream does."""
        if depth == 0 or self._upstream_over(game):
            return value_upstream(game, seat)
        maximizing = game.decision_player == seat
        best = -math.inf if maximizing else math.inf
        # Move order changes these bounded values, so follow upstream's where possible.
        actions = self._upstream_order(self.candidates(game))
        spectra = [self._spectrum(game, a, deck, rng) for a in actions]  # expand_spectrum
        for outcomes in spectra:
            expected = 0.0
            for child, rest, p in outcomes:
                expected += p * self._alphabeta(child, rest, seat, depth - 1, alpha, beta, rng)
            if maximizing:
                best = max(best, expected)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            else:
                best = min(best, expected)
                beta = min(beta, best)
                if beta <= alpha:
                    break
        return best

    def act(self, game, seat, legal):
        if game.state.phase == Phase.TRADE_RESPONSE:
            return REJECT_TRADE
        if game.state.phase == Phase.TRADE_CONFIRM:
            return CANCEL_TRADE
        # Root candidates are the referee's legal moves; deeper levels use CARLA's rules.
        actions = [
            a for a in legal if a.type not in (ActionType.PROPOSE_TRADE, ActionType.INVITE_TRADE)
        ]
        if len(actions) == 1:
            return actions[0]
        world = (
            game.clone()
            if self.information == "open"
            else sample_world(
                AgentView.from_game(game, seat),
                derive_seed(self.seed, "decision", game.num_steps),
            )
        )
        self.nodes = 0
        scores = []
        if self.chance == "upstream":
            rng = self.chance_rng or Rng(derive_seed(self.seed, "chance", game.num_steps))
            deck = self.chance_deck
            if self.depth == 1:
                scores = [self._sampled(world, a, seat, rng, deck) for a in actions]
            else:
                alpha = -math.inf
                spectra = [self._spectrum(world, a, deck, rng) for a in actions]
                for outcomes in spectra:
                    expected = 0.0
                    for child, rest, p in outcomes:
                        expected += p * self._alphabeta(
                            child, rest, seat, self.depth - 1, alpha, math.inf, rng
                        )
                    scores.append(expected)
                    alpha = max(alpha, max(scores))
        else:
            for action in actions:
                # Every root gets evaluated, even after the deeper-search budget expires.
                scores.append(self._score(world, action, seat, self.depth))
        self.last_scores = [(a.to_json(), v) for a, v in zip(actions, scores, strict=True)]
        self.budget_exhaustions += self.nodes >= self.node_budget
        return actions[max(range(len(actions)), key=lambda i: scores[i])]
