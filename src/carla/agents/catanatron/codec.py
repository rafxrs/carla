"""CARLA private world to Catanatron, using attributed OSP reconstruction."""

from carla.agents.catanatron.coordinates import mappings
from carla.engine.actions import ActionType
from carla.engine.constants import RESOURCE_NAMES
from carla.engine.rules import trade_ratios
from carla.engine.state import Phase
from carla.engine.topology import EDGE_CORNERS

CARDS = ("knight", "road_building", "year_of_plenty", "monopoly", "victory_point")
CODES = ("FO", "H", "P", "F", "M")
HARBORS = ("HRL", "HRB", "HRW", "HRG", "HRO", "HR")
DOMESTIC_OPENINGS = (ActionType.PROPOSE_TRADE, ActionType.INVITE_TRADE)


def representable(actions):
    """The actions Catanatron can choose: it has no domestic trade, so it never opens one."""
    return [a for a in actions if a.type not in DOMESTIC_OPENINGS]


def moves(world, actions):
    keys, _, tiles, _ = mappings()
    result = []
    for a in actions:
        t = a.type
        simple = {
            ActionType.ROLL_DICE: "roll",
            ActionType.END_TURN: "end_turn",
            ActionType.BUY_DEV_CARD: "buy_development",
        }
        if t in simple:
            move = [simple[t]]
        elif t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            move = [
                "build_settlement" if t == ActionType.BUILD_SETTLEMENT else "build_city",
                keys[a.corner],
            ]
        elif t == ActionType.BUILD_ROAD:
            x, y = EDGE_CORNERS[a.edge]
            move = ["build_road", sorted((keys[x], keys[y]))]
        elif t == ActionType.MOVE_ROBBER:
            move = [
                "move_robber",
                {"hex": tiles[a.tile], "victim": None if a.player < 0 else a.player},
            ]
        elif t == ActionType.DISCARD:
            move = ["discard", {RESOURCE_NAMES[a.resource]: 1}]
        elif t == ActionType.MARITIME_TRADE:
            move = [
                "bank_trade",
                {
                    "give": {
                        RESOURCE_NAMES[a.resource]: trade_ratios(
                            world.state, world.decision_player
                        )[a.resource]
                    },
                    "get": {RESOURCE_NAMES[a.resource2]: 1},
                },
            ]
        elif t == ActionType.PLAY_DEV_CARD:
            names = ("play_knight", "play_road_building", "play_year_of_plenty", "play_monopoly")
            move = [names[a.dev]]
            if a.dev == 2:
                move.append(sorted((RESOURCE_NAMES[a.resource], RESOURCE_NAMES[a.resource2])))
            elif a.dev == 3:
                move.append(RESOURCE_NAMES[a.resource])
        else:
            raise ValueError(f"no Catanatron conversion for {a}")
        result.append(move)
    return result


def reconstruct(world, seed):
    import random

    from catanatron.models.enums import Action
    from catanatron.models.enums import ActionType as CT
    from catanatron.models.player import Color

    from carla.agents.catanatron.vendor.rebuild import rebuild

    s = world.state
    keys, _, tiles, _ = mappings()
    board = {
        "tiles": [
            {
                "tile": tiles[t],
                "tCode": "D" if r < 0 else CODES[r],
                "rNumber": s.board.tile_number[t] or None,
            }
            for t, r in enumerate(s.board.tile_resource)
        ],
        "harbors": [
            {"path": sorted((keys[a], keys[b])), "hCode": HARBORS[s.board.port_types[i]]}
            for i, (a, b) in enumerate(s.board.port_corners)
        ],
    }
    prompt = {
        Phase.SETUP_SETTLEMENT: "setup_settlement",
        Phase.SETUP_ROAD: "setup_road",
        Phase.PRE_ROLL: "roll",
        Phase.DISCARD: "discard",
        Phase.MOVE_ROBBER: "move_robber",
        Phase.MAIN: "main",
        Phase.ROAD_BUILDING: "road_building",
    }[s.phase]
    players = []
    for p in range(4):
        players.append(
            {
                "seat": p,
                "public_vp": world.victory_points(p, include_hidden=False),
                "vp": world.victory_points(p),
                "hand": dict(zip(RESOURCE_NAMES, s.hands[p], strict=True)),
                "development": dict(zip(CARDS, s.dev_hands[p], strict=True)),
                "played": dict(zip(CARDS, s.dev_played[p], strict=True)),
                "longest_road": s.road_lengths[p],
                "has_longest_road": s.longest_road_owner == p,
                "has_largest_army": s.largest_army_owner == p,
                "pieces_left": {
                    "roads": s.roads_left[p],
                    "settlements": s.settlements_left[p],
                    "cities": s.cities_left[p],
                },
            }
        )
    state = {
        "players": players,
        "prompt": prompt,
        "current": s.decision_player,
        "on_turn": s.current_player,
        "turn": s.turn,
        "free_roads": s.free_roads,
        "robber": tiles[s.robber_tile],
        "bank": {**dict(zip(RESOURCE_NAMES, s.bank, strict=True)), "development": sum(s.dev_deck)},
        "buildings": {
            "settlements": [
                [p, keys[c]]
                for c, p in enumerate(s.corner_owner)
                if p >= 0 and s.corner_building[c] == 1
            ],
            "cities": [
                [p, keys[c]]
                for c, p in enumerate(s.corner_owner)
                if p >= 0 and s.corner_building[c] == 2
            ],
            "roads": [
                [p, sorted((keys[a], keys[b]))]
                for e, (a, b) in enumerate(EDGE_CORNERS)
                if (p := s.edge_owner[e]) >= 0
            ],
        },
    }
    legal = representable(world.legal_actions())
    game = rebuild(
        board,
        state,
        moves(world, legal),
        {"victory_points": s.config.vp_to_win, "discard_limit": s.config.discard_limit},
    )
    game.random = game.state.random = random.Random(seed)
    # OSP reconstruction shuffles its deck before assigning our deterministic RNG.
    cards = ("KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY", "VICTORY_POINT")
    game.state.development_listdeck = [cards[c] for c, n in enumerate(s.dev_deck) for _ in range(n)]
    game.state.random.shuffle(game.state.development_listdeck)
    # Both engines discard one resource at a time. Keep the full pending count.
    if s.phase == Phase.DISCARD:
        game.state.discard_counts = list(s.discards_pending)
    if s.phase == Phase.SETUP_ROAD:
        # Catanatron normally reads its previous action record to find this node.
        from catanatron.models.enums import ActionRecord

        node = mappings()[1][s.setup_corner]
        game.state.action_records = [
            ActionRecord(Action(tuple(Color)[s.decision_player], CT.BUILD_SETTLEMENT, node), None)
        ]
    return game
