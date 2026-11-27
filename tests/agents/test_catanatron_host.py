"""CARLA agents seated in Catanatron's own engine (`carla.agents.catanatron.host`)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("catanatron")

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer

from carla.agents.catanatron.host import CarlaPlayer, carla_action, carla_game
from carla.agents.catanatron.native import NativeCatanatronAgent
from carla.engine.actions import ActionType

TRADES = (ActionType.PROPOSE_TRADE, ActionType.INVITE_TRADE)


@pytest.mark.parametrize("seed", [0, 1])
def test_converted_states_list_exactly_catanatrons_playable_actions(seed):
    game = Game([ValueFunctionPlayer(c) for c in Color], seed=seed)
    checked = 0
    while game.winning_color() is None and game.state.num_turns < 1000:
        colors, seat = list(game.state.colors), game.state.current_player_index
        theirs = {carla_action(a, colors) for a in game.playable_actions}
        for information in ("view", "open"):
            world = carla_game(game, seat, information)
            mine = {a for a in world.legal_actions() if a.type not in TRADES}
            assert mine == theirs, (game.state.num_turns, information, mine ^ theirs)
        checked += 1
        game.play_tick()
    assert checked > 100


def test_view_hides_opponent_cards_and_open_reveals_them():
    game = Game([ValueFunctionPlayer(c) for c in Color], seed=3)
    while game.state.num_turns < 12:
        game.play_tick()
    ps, seat = game.state.player_state, game.state.current_player_index
    view, open_ = carla_game(game, seat, "view"), carla_game(game, seat, "open")
    for p in range(4):
        truth = [ps[f"P{p}_{r}_IN_HAND"] for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")]
        assert open_.state.hands[p] == truth
        assert view.state.hands[p] == (truth if p == seat else [sum(truth), 0, 0, 0, 0])
    assert sum(view.state.dev_deck) == sum(open_.state.dev_deck) == view.state.dev_deck[0]


@pytest.mark.parametrize(("depth", "information"), [(1, "view"), (1, "open"), (2, "view")])
def test_native_ports_complete_games_on_catanatrons_engine(depth, information):
    colors = list(Color)
    agent = NativeCatanatronAgent(
        depth=depth, node_budget=500, information="open" if information == "open" else "sampled"
    )
    players = [CarlaPlayer(colors[0], agent, information=information)]
    players += [ValueFunctionPlayer(c) for c in colors[1:]]
    game = Game(players, seed=5)
    assert game.play() is not None


def test_policy_comparison_script_reports_agreement_by_phase():
    path = Path(__file__).resolve().parents[2] / "scripts" / "check_catanatron_policy.py"
    spec = importlib.util.spec_from_file_location("check_catanatron_policy", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    row = script.compare("f", 12, 0, "open", 40)
    assert row["failure"] is None and row["decisions"] == 40
    assert row["agreements"] <= row["decisions"]
    assert sum(v["compared"] for v in row["by_phase"].values()) == row["decisions"]


def _trajectory(players, seed):
    game = Game(players, seed=seed)
    game.play()
    return [
        (r.action.color, r.action.action_type, r.action.value, r.result)
        for r in game.state.action_records
    ]


@pytest.mark.parametrize(("depth", "seed"), [(1, 3), pytest.param(2, 7, marks=pytest.mark.slow)])
def test_upstream_chance_ports_reproduce_original_games_exactly(depth, seed, monkeypatch):
    """Seated with Catanatron's RNG, the upstream-mode port makes the original's every choice
    and RNG draw, so a whole game matches the all-original game action for action."""
    from catanatron.players import minimax
    from catanatron.players.minimax import AlphaBetaPlayer

    monkeypatch.setattr(minimax, "MAX_SEARCH_TIME_SECS", 1e9)  # deadline-free, deterministic

    def original(color):
        if depth == 1:
            return ValueFunctionPlayer(color)
        return AlphaBetaPlayer(color, AlphaBetaPlayer.Params(depth=2))

    colors = list(Color)
    reference = _trajectory([original(c) for c in colors], seed)
    port = NativeCatanatronAgent(depth=depth, information="open", chance="upstream")
    players = [CarlaPlayer(colors[0], port, information="open"), *map(original, colors[1:])]
    assert _trajectory(players, seed) == reference


@pytest.mark.parametrize("name", ["native-f-upstream", "native-ab2-upstream"])
def test_upstream_chance_ports_complete_games_under_carlas_referee(name):
    from carla.agents.heuristic import HeuristicAgent
    from carla.agents.registry import make_agent
    from carla.engine.config import EngineConfig
    from carla.eval.runner import play_game

    result = play_game(
        [make_agent(name), *[HeuristicAgent() for _ in range(3)]], 12, EngineConfig(max_turns=30)
    )
    assert result.steps > 16


@pytest.mark.parametrize("seed", range(6))
def test_mirror_replays_catanatron_transitions_under_its_rules_profile(seed):
    """With `catanatron_rules`, every executed Catanatron record replays on a CARLA game with
    the same hands, bank, pieces, robber, cards and turn state (road lengths are adopted by
    the bridge where Catanatron's component bookkeeping differs)."""
    from carla.agents.catanatron.host import _CHECKED, record_outcome
    from carla.engine.game import Game as CarlaGame

    played = Game([ValueFunctionPlayer(c) for c in Color], seed=seed)
    played.play()
    replay = Game([ValueFunctionPlayer(c) for c in Color], seed=seed)
    world = carla_game(replay, 0, "open")
    mirror = CarlaGame(world.config, seed, world.state.board)
    colors = list(replay.state.colors)
    strict = [f for f in _CHECKED if f not in ("road_lengths", "longest_road_owner")]
    for record in played.state.action_records:
        replay.execute(record.action, validate_action=False, action_record=record)
        mirror.step(carla_action(record.action, colors), record_outcome(record), validate=False)
        if replay.winning_color() is not None:
            break
        truth = carla_game(replay, 0, "open").state
        mine = mirror.state
        mine.road_lengths[:], mine.longest_road_owner = truth.road_lengths, truth.longest_road_owner
        assert [f for f in strict if getattr(truth, f) != getattr(mine, f)] == []


@pytest.fixture(scope="module")
def recurrent_checkpoint(tmp_path_factory):
    import torch

    from carla.config import to_dict
    from carla.config.schema import ExperimentConfig, ModelConfig
    from carla.engine.config import EngineConfig
    from carla.models import build_policy

    torch.manual_seed(0)
    model = ModelConfig(
        encoder="gnn", hidden_dim=32, gnn_layers=2, set_attention=True, belief="gru", belief_dim=16
    )
    config = ExperimentConfig(engine=EngineConfig(max_turns=500), model=model)
    path = tmp_path_factory.mktemp("policy") / "latest.pt"
    torch.save({"config": to_dict(config), "policy": build_policy(model).state_dict()}, path)
    return path


@pytest.mark.parametrize("name", ["heuristic", "jsettlers-fast-native", "checkpoint"])
def test_mirror_hosts_history_and_event_based_agents(name, request):
    from carla.agents.registry import make_agent

    if name == "checkpoint":
        pytest.importorskip("torch")
        name = f"ppo:{request.getfixturevalue('recurrent_checkpoint')}"
    colors = list(Color)
    agent = make_agent(name)
    seen = []
    original = agent.observe
    agent.observe = lambda events, step: (seen.append(step), original(events, step))
    player = CarlaPlayer(colors[1], agent)
    game = Game([ValueFunctionPlayer(colors[0]), player, *map(ValueFunctionPlayer, colors[2:])], 9)
    assert game.play() is not None
    assert seen and player._mirror.num_steps == len(game.state.action_records)
    assert getattr(agent, "fallbacks", 0) == 0
