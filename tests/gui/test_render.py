from math import hypot

import pytest

from carla.gui.render import Layer, circle_points, rounded_rect_points, scaled, segments_for

RED = (200, 10, 10, 255)


def colours(layer: Layer) -> set[tuple[float, ...]]:
    return {tuple(layer.data[i + 2 : i + 6]) for i in range(0, len(layer.data), 6)}


def points(layer: Layer) -> list[tuple[float, float]]:
    return [(layer.data[i], layer.data[i + 1]) for i in range(0, len(layer.data), 6)]


@pytest.mark.parametrize(
    ("draw", "vertices"),
    [
        (lambda layer: layer.triangle((0, 0), (1, 0), (0, 1), RED), 3),
        (lambda layer: layer.rect(0, 0, 10, 5, RED), 6),
        (lambda layer: layer.polygon([(0, 0), (2, 0), (3, 1), (2, 2), (0, 2)], RED), 9),
        (lambda layer: layer.circle(0, 0, 10, RED, segments=12), 30),
        (lambda layer: layer.ring(0, 0, 10, 2, RED, segments=12), 72),
        (lambda layer: layer.line(0, 0, 10, 0, 4, RED), 6),
        (lambda layer: layer.rounded_rect(0, 0, 40, 20, 6, RED), 54),
        (lambda layer: layer.rounded_frame(0, 0, 40, 20, 6, 2, RED), 120),
    ],
)
def test_shapes_become_coloured_triangles(draw, vertices):
    layer = Layer("test")
    draw(layer)
    assert layer.vertex_count == vertices and len(layer.data) == 6 * vertices
    assert colours(layer) == {RED}


def test_line_width_and_round_caps():
    layer = Layer("test")
    layer.line(0, 0, 10, 0, 4, RED)
    assert {round(y, 9) for _, y in points(layer)} == {-2, 2}
    capped = Layer("test")
    capped.line(0, 0, 10, 0, 4, RED, round_caps=True)
    assert capped.vertex_count > layer.vertex_count
    assert -2 - 1e-9 <= min(x for x, _ in points(capped)) < -1.8  # the cap's polygon
    dot = Layer("test")
    dot.line(5, 5, 5, 5, 4, RED, round_caps=True)  # zero length: only the caps
    assert dot.vertex_count > 0


def test_outline_has_the_requested_width():
    hexagon = circle_points(0, 0, 50, 6)
    layer = Layer("test")
    layer.outline(hexagon, (0, 0), 5, RED)
    distances = {round(hypot(x, y), 6) for x, y in points(layer)}
    apothem = 50 * 3**0.5 / 2
    assert distances == {50, round(50 * (1 - 5 / apothem), 6)}


def test_point_helpers():
    assert all(hypot(x - 3, y - 4) == pytest.approx(7) for x, y in circle_points(3, 4, 7))
    assert segments_for(1) == 8 and segments_for(1000) == 64
    assert segments_for(10) <= segments_for(30)
    box = rounded_rect_points(10, 20, 40, 30, 8)
    assert all(10 - 1e-9 <= x <= 50 + 1e-9 and 20 - 1e-9 <= y <= 50 + 1e-9 for x, y in box)
    assert len(box) == 20
    assert scaled([(2, 0), (0, 2)], (0, 0), 1.5) == [(3, 0), (0, 3)]


def test_sector_and_text():
    layer = Layer("test", key="static")
    layer.sector(0, 0, 10, 0, 180, RED, ry=4)
    assert all(y >= -1e-9 and y <= 4 + 1e-9 for _, y in points(layer))
    layer.text("Hi", 10.6, 20.4, 12, RED, True, "center", "top")
    (item,) = layer.texts
    assert (item.text, item.x, item.y, item.size, item.bold) == ("Hi", 11, 20, 12, True)
    assert layer.key == "static" and layer.name == "test"
