from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from carla.config import EnvConfig, load_config
from carla.engine.constants import DESERT, PIPS, Building, DevCard
from carla.engine.rules import road_edges, settlement_corners
from carla.engine.state import Phase
from carla.engine.topology import CORNER_NEIGHBORS, CORNER_TILES, TILE_CORNERS
from carla.envs import rewards
from tests.engine.scenario import (
    BOARD,
    edge_between,
    give,
    give_dev,
    post_setup_game,
    put_road,
    put_settlement,
)

STRATEGIC = EnvConfig(shaping="strategic", shaping_coef=0.5)
CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "experiments"
COAST = [
    c
    for c, tiles in enumerate(CORNER_TILES)
    if len(tiles) == 1 and BOARD.tile_resource[tiles[0]] != DESERT
]


@pytest.fixture
def build_game():
    game = post_setup_game()
    start = 0
    middle = CORNER_NEIGHBORS[start][0]
    end = next(c for c in CORNER_NEIGHBORS[middle] if c != start)
    put_settlement(game, 0, start)
    put_road(game, 0, edge_between(start, middle))
    put_road(game, 0, edge_between(middle, end))
    assert end in settlement_corners(game.state, 0)
    give(game, 0, (1, 1, 1, 2, 3))
    return game


@pytest.mark.parametrize("winner", [-1, 0])
def test_terminal_potential_is_zero_and_terminal_rewards_are_unchanged(winner):
    game = post_setup_game()
    put_settlement(game, 0, COAST[0])
    assert rewards.potential(game, 0, STRATEGIC) != 0
    game.state.phase = Phase.GAME_OVER
    game.state.winner = winner
    assert rewards.potential(game, 0, STRATEGIC) == 0
    assert rewards.strategic_potential(game, 0, STRATEGIC) == 0
    expected = [1, -1, -1, -1] if winner == 0 else [0, 0, 0, 0]
    np.testing.assert_array_equal(rewards.step_rewards(None, game, STRATEGIC), expected)


def test_none_still_has_zero_potential(build_game):
    assert rewards.potential(build_game, 0, EnvConfig()) == 0
    assert rewards.potential(build_game, 0, STRATEGIC) > 0


def test_production_is_additive_and_city_is_twice_settlement():
    game = post_setup_game()
    corner = COAST[0]
    tile = CORNER_TILES[corner][0]
    expected = PIPS[BOARD.tile_number[tile]] / 72.0
    assert rewards.production_strength(game.state, 0) == 0
    put_settlement(game, 0, corner)
    assert rewards.production_strength(game.state, 0) == pytest.approx(expected)
    city = game.clone()
    city.state.corner_building[corner] = Building.CITY
    city.state.settlements_left[0] += 1
    city.state.cities_left[0] -= 1
    assert rewards.production_strength(city.state, 0) == pytest.approx(2 * expected)
    extra = next(c for c in COAST if c != corner and c not in CORNER_NEIGHBORS[corner])
    put_settlement(game, 0, extra)
    extra_tile = CORNER_TILES[extra][0]
    assert rewards.production_strength(game.state, 0) == pytest.approx(
        expected + PIPS[BOARD.tile_number[extra_tile]] / 72.0
    )


def test_robber_blocked_tile_produces_nothing():
    game = post_setup_game()
    corner = COAST[0]
    put_settlement(game, 0, corner)
    game.state.robber_tile = CORNER_TILES[corner][0]
    assert rewards.production_strength(game.state, 0) == 0


def test_desert_is_ignored_even_when_robber_moves_away():
    game = post_setup_game()
    desert = BOARD.desert_tile
    corner = TILE_CORNERS[desert][0]
    put_settlement(game, 0, corner)
    game.state.robber_tile = next(
        t for t in range(len(TILE_CORNERS)) if t not in CORNER_TILES[corner]
    )
    expected = sum(PIPS[BOARD.tile_number[t]] / 72.0 for t in CORNER_TILES[corner] if t != desert)
    assert rewards.production_strength(game.state, 0) == pytest.approx(expected)


def test_build_readiness_uses_partial_cost_deficits(build_game):
    assert rewards.build_readiness(build_game.state, 0) == pytest.approx(1)
    build_game.state.hands[0][1] = 0  # Missing brick: road=1/2, settlement=3/4.
    assert rewards.build_readiness(build_game.state, 0) == pytest.approx(
        0.15 * 0.5 + 0.35 * 0.75 + 0.35 + 0.15
    )
    give(build_game, 0, (0, 1, 0, 0, 0))
    assert rewards.build_readiness(build_game.state, 0) == pytest.approx(1)


@pytest.mark.parametrize(
    ("unavailable", "expected"),
    [
        ("road_piece", 0.85),
        ("road_location", 0.85),
        ("settlement_piece", 0.65),
        ("settlement_location", 0.65),
        ("city_piece", 0.65),
        ("city_location", 0.65),
        ("dev_deck", 0.85),
    ],
)
def test_structurally_unavailable_purchases_contribute_zero(build_game, unavailable, expected):
    state = build_game.state
    if unavailable == "road_piece":
        state.roads_left[0] = 0
    elif unavailable == "road_location":
        for edge in road_edges(state, 0):
            put_road(build_game, 1, edge)
    elif unavailable == "settlement_piece":
        state.settlements_left[0] = 0
    elif unavailable == "settlement_location":
        for corner in settlement_corners(state, 0):
            put_settlement(build_game, 1, corner)
    elif unavailable == "city_piece":
        state.cities_left[0] = 0
    elif unavailable == "city_location":
        state.corner_building[0] = Building.CITY
        state.settlements_left[0] += 1
        state.cities_left[0] -= 1
    else:
        state.dev_deck = [0] * len(state.dev_deck)
    assert rewards.build_readiness(state, 0) == pytest.approx(expected)


def test_setup_corners_are_not_normal_play_build_locations():
    game = post_setup_game()
    give(game, 0, (1, 1, 1, 2, 3))
    # No network or upgradeable buildings: only the development card is available.
    assert rewards.build_readiness(game.state, 0) == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("owner", "own_length", "owner_length", "expected"),
    [(-1, 2, 0, 2 / 5), (1, 3, 6, 3 / 7), (0, 6, 6, 1), (-1, 7, 0, 1)],
)
def test_longest_road_progress(owner, own_length, owner_length, expected):
    state = post_setup_game().state
    state.longest_road_owner = owner
    state.road_lengths[0] = own_length
    if owner >= 0:
        state.road_lengths[owner] = owner_length
    assert rewards.road_progress(state, 0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("owner", "own_played", "owner_played", "expected"),
    [(-1, 1, 0, 1 / 3), (1, 2, 4, 2 / 5), (0, 3, 3, 1), (-1, 4, 0, 1)],
)
def test_largest_army_progress(owner, own_played, owner_played, expected):
    state = post_setup_game().state
    state.largest_army_owner = owner
    state.dev_played[0][DevCard.KNIGHT] = own_played
    if owner >= 0:
        state.dev_played[owner][DevCard.KNIGHT] = owner_played
    state.dev_hands[0][DevCard.KNIGHT] = 5  # Unplayed knights must not count.
    assert rewards.army_progress(state, 0) == pytest.approx(expected)


def test_opponent_weights_use_raw_public_vp_and_stable_softmax(monkeypatch):
    state = post_setup_game().state
    state.settlements_left[1:] = [3, 1, 0]  # Public VP: 2, 4, 5.
    weights = rewards.opponent_weights(state, 0)
    expected = np.exp(np.asarray([2, 4, 5]) / 2 - 2.5)
    np.testing.assert_allclose(weights, expected / expected.sum())
    assert np.all(weights > 0) and weights.sum() == pytest.approx(1)
    assert weights[0] < weights[1] < weights[2]

    def high_public_vp(state, seat, include_hidden=True):
        assert not include_hidden
        return [0, 10002, 10004, 10005][seat]

    monkeypatch.setattr(rewards, "victory_points", high_public_vp)
    np.testing.assert_allclose(rewards.opponent_weights(state, 0), weights)


def test_strategic_potential_matches_the_specified_weighted_formula():
    game = post_setup_game()
    own_corner = COAST[0]
    opponent_corner = next(
        c for c in COAST if c != own_corner and c not in CORNER_NEIGHBORS[own_corner]
    )
    put_settlement(game, 0, own_corner)
    put_settlement(game, 1, opponent_corner, city=True)
    give(game, 0, (1, 1, 1, 2, 3))
    game.state.road_lengths[1] = 3
    game.state.dev_played[1][DevCard.KNIGHT] = 1
    own_production = PIPS[BOARD.tile_number[CORNER_TILES[own_corner][0]]] / 72.0
    opponent_production = 2 * PIPS[BOARD.tile_number[CORNER_TILES[opponent_corner][0]]] / 72.0
    opponent_awards = (3 / 5 + 1 / 3) / 2
    # No normal-play settlement site without roads, so own readiness is 0.65.
    utility = 0.50 * 0.1 + 0.25 * own_production + 0.10 * 0.65
    strength = 0.56 * 0.2 + 0.28 * opponent_production + 0.16 * opponent_awards
    threat = np.exp(1) / (np.exp(1) + 2) * strength
    assert rewards.award_progress(game.state, 1) == pytest.approx(opponent_awards)
    assert rewards.player_utility(game.state, 0) == pytest.approx(utility)
    assert rewards.public_opponent_strength(game.state, 1) == pytest.approx(strength)
    assert rewards.opponent_threat(game.state, 0) == pytest.approx(threat)
    phi = rewards.potential(game, 0, STRATEGIC)
    assert phi == pytest.approx(0.5 * (utility - 0.35 * threat))
    assert -0.175 <= phi <= 0.5
    assert rewards.potential(game, 0, replace(STRATEGIC, shaping_coef=1)) == pytest.approx(2 * phi)


@pytest.mark.parametrize("seat", range(4))
def test_hidden_opponent_resource_composition_does_not_affect_potential(seat):
    game = post_setup_game()
    opponent = (seat + 1) % 4
    give(game, opponent, (2, 0, 0, 0, 0))
    before = rewards.potential(game, seat, STRATEGIC)
    # Same hand size and all public fields, including bank, unchanged.
    game.state.hands[opponent] = [0, 2, 0, 0, 0]
    assert rewards.potential(game, seat, STRATEGIC) == before


@pytest.mark.parametrize("seat", range(4))
def test_hidden_opponent_vp_cards_do_not_affect_potential(seat):
    game = post_setup_game()
    opponent = (seat + 1) % 4
    give_dev(game, opponent, DevCard.KNIGHT)
    before = rewards.potential(game, seat, STRATEGIC)
    # Counterfactual hidden card identity; all public fields stay fixed.
    game.state.dev_hands[opponent][DevCard.KNIGHT] -= 1
    game.state.dev_hands[opponent][DevCard.VICTORY_POINT] += 1
    assert rewards.potential(game, seat, STRATEGIC) == before


def test_learner_hidden_vp_card_increases_utility():
    game = post_setup_game()
    before = rewards.potential(game, 0, STRATEGIC)
    give_dev(game, 0, DevCard.VICTORY_POINT)
    assert rewards.potential(game, 0, STRATEGIC) - before == pytest.approx(0.5 * 0.50 / 10)


def test_strategic_potential_never_reads_hidden_opponent_hands(build_game):
    class Unreadable(list):
        def __getitem__(self, key):
            raise AssertionError("read hidden opponent cards")

        def __iter__(self):
            raise AssertionError("iterate hidden opponent cards")

        def __len__(self):
            raise AssertionError("inspect hidden opponent cards")

    before = rewards.potential(build_game, 0, STRATEGIC)
    for seat in range(1, 4):
        build_game.state.hands[seat] = Unreadable()
        build_game.state.dev_hands[seat] = Unreadable()
    assert rewards.potential(build_game, 0, STRATEGIC) == before


def test_strategic_config_resolves_full_model_without_changing_other_settings():
    parent = load_config(CONFIGS / "full_trade.yaml")
    config = load_config(CONFIGS / "full_trade_strategic.yaml")
    assert config == replace(
        parent,
        name="full_trade_strategic",
        model=replace(parent.model, hidden_dim=256),
        env=replace(
            parent.env,
            win_reward=1.0,
            loss_reward=-1.0,
            truncation_reward=0.0,
            vp_reward=0.0,
            shaping="strategic",
            shaping_coef=0.5,
            shaping_gamma=0.999,
        ),
        ppo=replace(parent.ppo, gamma=0.999),
    )
    assert config.model.encoder == "gnn" and config.model.hidden_dim == 256
    assert config.model.set_attention and config.model.belief == "gru"
    assert config.model.trade_decoder == "gru" and config.engine.allow_domestic_trade
    assert config.env.shaping == "strategic" and config.env.shaping_coef == 0.5
    assert config.env.vp_reward == 0 and config.ppo.gamma == config.env.shaping_gamma == 0.999
