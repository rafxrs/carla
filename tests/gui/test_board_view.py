from carla.engine.constants import DESERT, GENERIC_PORT
from carla.engine.game import Game
from carla.engine.simulate import play_random_game
from carla.gui import theme
from carla.gui.board_view import BoardMarks, BoardView
from carla.gui.geometry import Rect, compute_layout

LAYOUT = compute_layout(1280, 800)


def view() -> BoardView:
    board_view = BoardView()
    board_view.set_viewport(LAYOUT.viewport)
    return board_view


def test_static_layer_is_cached_until_the_board_or_size_changes():
    board_view, board = view(), Game(seed=1).board
    first = board_view.static_layer(board, LAYOUT.board)
    assert board_view.static_layer(board, LAYOUT.board) is first and first.key is not None
    assert board_view.static_layer(Game(seed=2).board, LAYOUT.board) is not first
    bigger = compute_layout(1600, 1000)
    board_view.set_viewport(bigger.viewport)
    assert board_view.static_layer(board, bigger.board).key != first.key


def test_static_layer_labels_tokens_and_ports():
    board = Game(seed=3).board
    layer = view().static_layer(board, LAYOUT.board)
    texts = [t.text for t in layer.texts]
    numbers = sorted(str(n) for n in board.tile_number if n)
    assert sorted(t for t in texts if t.isdigit()) == numbers
    hot = [t for t in layer.texts if t.text in ("6", "8")]
    assert len(hot) == 4 and all(t.color == theme.HOT and t.bold for t in hot)
    generic = sum(kind == GENERIC_PORT for kind in board.port_types)
    assert texts.count("3:1") == generic and texts.count("2:1") == 9 - generic
    initials = {theme.RESOURCE_INITIALS[k] for k in board.port_types if k != GENERIC_PORT}
    assert initials <= set(texts)


def test_pieces_layer_draws_pieces_targets_and_previews():
    game = play_random_game(6)
    board_view = view()
    empty = board_view.pieces_layer(Game(seed=6).state, BoardMarks(), 0.0)
    full = board_view.pieces_layer(game.state, BoardMarks(), 0.0)
    assert full.vertex_count > empty.vertex_count > 0 and not full.texts  # robber at least
    marks = BoardMarks(corners={3: False, 10: True}, edges=frozenset({5}), tiles=frozenset({4}))
    targets = board_view.pieces_layer(game.state, marks, 0.0)
    assert targets.vertex_count > full.vertex_count
    for hover in (("corner", 3), ("corner", 10), ("edge", 5), ("tile", 4)):
        marks.hover, marks.player = hover, 2
        assert board_view.pieces_layer(game.state, marks, 0.0).vertex_count > targets.vertex_count
    marks.hover = ("corner", 30)  # not a target: no preview
    assert board_view.pieces_layer(game.state, marks, 0.0).vertex_count == targets.vertex_count
    flashes = BoardMarks(flash=("edge", 7), flash_strength=0.5, rolled=8)
    assert board_view.pieces_layer(game.state, flashes, 0.0).vertex_count > full.vertex_count


def test_pulsing_targets_change_with_time():
    board_view, state = view(), Game(seed=2).state
    marks = BoardMarks(corners=dict.fromkeys(range(54), False))
    a = board_view.pieces_layer(state, marks, 0.0).data
    b = board_view.pieces_layer(state, marks, 0.3).data
    assert a != b


def test_robber_stands_beside_the_token_or_on_the_desert():
    board_view, board = view(), Game(seed=4).board
    desert = board.tile_resource.index(DESERT)
    assert board_view.robber_point(board, desert) == board_view.transform.tiles[desert]
    other = (desert + 1) % 19
    x, y = board_view.robber_point(board, other)
    tx, ty = board_view.transform.tiles[other]
    assert y == ty and x < tx - 0.4 * board_view.transform.scale


def test_viewport_change_moves_the_transform():
    board_view = view()
    transform = board_view.transform
    board_view.set_viewport(LAYOUT.viewport)
    assert board_view.transform is transform
    board_view.set_viewport(Rect(0, 0, 300, 300))
    assert board_view.transform is not transform and board_view.transform.scale < transform.scale
