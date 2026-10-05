from carla.engine import actions as A
from carla.engine.constants import DEV_CARD_COST, DevCard, Resource
from carla.engine.state import Phase
from carla.engine.topology import CORNER_EDGES, CORNER_NEIGHBORS
from tests.engine.scenario import check_invariants, give, give_dev, post_setup_game, put_settlement


def dev_plays(game) -> list:
    return [a for a in game.legal_actions() if a.type == A.ActionType.PLAY_DEV_CARD]


def advance_to(game, seat: int) -> None:
    """End turns (rolling a harmless 5) until `seat` is about to roll."""
    while True:
        if game.phase == Phase.PRE_ROLL:
            if game.current_player == seat:
                return
            game.step(A.ROLL, outcome=(2, 3))
        game.step(A.END_TURN)


def test_bought_card_cannot_be_played_the_same_turn():
    game = post_setup_game()
    give(game, 0, DEV_CARD_COST)
    events = game.step(A.BUY_DEV, outcome=DevCard.MONOPOLY)
    assert events[0].value == DevCard.MONOPOLY and events[0].visible_to(1).value == -1
    s = game.state
    assert s.dev_hands[0][DevCard.MONOPOLY] == 1 and s.dev_deck[DevCard.MONOPOLY] == 1
    assert s.hands[0] == [0] * 5 and not dev_plays(game)
    advance_to(game, 0)
    assert set(dev_plays(game)) == set(A.MONOPOLY_ON)
    check_invariants(game)


def test_one_development_card_per_turn():
    game = post_setup_game()
    give_dev(game, 0, DevCard.KNIGHT, 2)
    game.step(A.PLAY_KNIGHT)
    game.step(game.legal_actions()[0])
    assert game.phase == Phase.MAIN and not dev_plays(game)


def test_victory_point_cards_count_but_are_never_played():
    game = post_setup_game()
    give_dev(game, 0, DevCard.VICTORY_POINT)
    assert not dev_plays(game)
    assert game.victory_points(0) == 1 and game.victory_points(0, include_hidden=False) == 0


def test_year_of_plenty():
    game = post_setup_game()
    give_dev(game, 0, DevCard.YEAR_OF_PLENTY)
    assert len(dev_plays(game)) == 15
    give(game, 1, (0, 0, 0, 0, 18))  # one ore left in the bank
    options = {(a.resource, a.resource2) for a in dev_plays(game)}
    assert (Resource.ORE, Resource.ORE) not in options and (Resource.GRAIN, Resource.ORE) in options
    game.step(A.YEAR_OF_PLENTY[Resource.ORE][Resource.GRAIN])
    assert game.state.hands[0] == [0, 0, 0, 1, 1]
    check_invariants(game)


def test_monopoly_takes_every_card_of_a_type():
    game = post_setup_game()
    give_dev(game, 0, DevCard.MONOPOLY)
    give(game, 1, (0, 0, 0, 3, 1))
    give(game, 2, (0, 0, 0, 2, 0))
    events = game.step(A.MONOPOLY_ON[Resource.GRAIN])
    assert events[-1].counts == (0, 3, 2, 0)
    assert game.state.hands[0][Resource.GRAIN] == 5
    assert game.state.hands[1] == [0, 0, 0, 0, 1] and game.state.hands[2] == [0] * 5
    check_invariants(game)


def test_road_building_places_two_free_roads():
    game = post_setup_game(rolled=False)
    corner = next(c for c in range(54) if len(CORNER_NEIGHBORS[c]) == 3)
    put_settlement(game, 0, corner)
    give_dev(game, 0, DevCard.ROAD_BUILDING)
    game.step(A.PLAY_ROAD_BUILDING)
    assert game.phase == Phase.ROAD_BUILDING
    assert {a.edge for a in game.legal_actions()} == set(CORNER_EDGES[corner])
    game.step(game.legal_actions()[0])
    assert game.phase == Phase.ROAD_BUILDING
    game.step(game.legal_actions()[0])
    assert game.phase == Phase.PRE_ROLL  # played before rolling
    assert game.state.roads_left[0] == 13 and game.state.hands[0] == [0] * 5
    check_invariants(game)


def test_road_building_stops_when_out_of_roads():
    game = post_setup_game()
    corner = next(c for c in range(54) if len(CORNER_NEIGHBORS[c]) == 3)
    put_settlement(game, 0, corner)
    give_dev(game, 0, DevCard.ROAD_BUILDING)
    s = game.state
    s.roads_left[0] = 1
    game.set_state(s)
    game.step(A.PLAY_ROAD_BUILDING)
    game.step(game.legal_actions()[0])
    assert game.phase == Phase.MAIN and game.state.roads_left[0] == 0


def test_largest_army_needs_three_and_a_strict_majority():
    game = post_setup_game(rolled=False)
    give_dev(game, 0, DevCard.KNIGHT, 3)
    give_dev(game, 1, DevCard.KNIGHT, 4)

    def play_knight(seat: int) -> None:
        advance_to(game, seat)
        game.step(A.PLAY_KNIGHT)
        game.step(game.legal_actions()[0])
        game.step(A.ROLL, outcome=(2, 3))
        game.step(A.END_TURN)

    for n in (1, 2, 3):
        play_knight(0)
        assert game.state.largest_army_owner == (0 if n == 3 else -1)
        play_knight(1)
    assert game.state.largest_army_owner == 0  # 3 vs 3: the holder keeps it
    play_knight(1)
    assert game.state.largest_army_owner == 1
    assert game.victory_points(1) == 2 and game.victory_points(0) == 0
    check_invariants(game)
