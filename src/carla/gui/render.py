"""Batched drawing. Views describe a frame as `Layer`s of coloured triangles plus text (pure
Python, testable without a display); the `Painter` draws each layer with one vertex buffer and
one batch of cached labels. Arcade's immediate-mode helpers cost one draw call per shape, which
is too slow for a board of a few hundred shapes at 60 frames per second.
"""

from array import array
from collections.abc import Hashable, Iterable, Sequence
from math import cos, hypot, radians, sin, tau
from typing import TYPE_CHECKING, NamedTuple

from carla.gui import theme
from carla.gui.theme import Color

if TYPE_CHECKING:
    import arcade
    from arcade.gl import Buffer, Geometry

Point = tuple[float, float]


class TextItem(NamedTuple):
    text: str
    x: float
    y: float
    size: float
    color: Color
    bold: bool
    anchor_x: str
    anchor_y: str


def circle_points(x: float, y: float, r: float, segments: int = 0) -> list[Point]:
    n = segments or segments_for(r)
    return [(x + r * cos(tau * k / n), y + r * sin(tau * k / n)) for k in range(n)]


def segments_for(radius: float) -> int:
    """Enough segments for a smooth circle of `radius` pixels, and no more."""
    return max(8, min(64, int(radius * 0.8) + 8))


def rounded_rect_points(
    x: float, y: float, w: float, h: float, radius: float, segments: int = 4
) -> list[Point]:
    """Outline of a rounded rectangle, counter-clockwise from the bottom right corner."""
    r = max(0.0, min(radius, w / 2, h / 2))
    corners = ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90))
    corners += ((x + r, y + r, 180),)
    points = []
    for cx, cy, start in corners:
        for k in range(segments + 1):
            a = radians(start + 90 * k / segments)
            points.append((cx + r * cos(a), cy + r * sin(a)))
    return points


def scaled(points: Sequence[Point], center: Point, factor: float) -> list[Point]:
    cx, cy = center
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points]


class Layer:
    """Coloured triangles (x, y, r, g, b, a per vertex) with text drawn over them.

    A layer with a `key` is static: the painter keeps its vertex buffer until the key changes.
    """

    def __init__(self, name: str, key: Hashable = None) -> None:
        self.name = name
        self.key = key
        self.data: list[float] = []
        self.texts: list[TextItem] = []

    @property
    def vertex_count(self) -> int:
        return len(self.data) // 6

    # ------------------------------------------------------------------ shapes

    def triangle(self, a: Point, b: Point, c: Point, color: Color) -> None:
        r, g, bl, al = color
        self.data.extend(
            (a[0], a[1], r, g, bl, al, b[0], b[1], r, g, bl, al, c[0], c[1], r, g, bl, al)
        )

    def polygon(self, points: Sequence[Point], color: Color) -> None:
        """Convex polygon (a fan from the first point)."""
        first = points[0]
        for i in range(1, len(points) - 1):
            self.triangle(first, points[i], points[i + 1], color)

    def band(self, outer: Sequence[Point], inner: Sequence[Point], color: Color) -> None:
        """The area between two closed outlines with matching points (a polygon's border)."""
        n = len(outer)
        for i in range(n):
            j = (i + 1) % n
            self.triangle(outer[i], outer[j], inner[j], color)
            self.triangle(outer[i], inner[j], inner[i], color)

    def outline(self, points: Sequence[Point], center: Point, width: float, color: Color) -> None:
        """Border of `width` pixels inside a convex polygon that is regular around `center`."""
        apothem = min(
            hypot(((a[0] + b[0]) / 2) - center[0], ((a[1] + b[1]) / 2) - center[1])
            for a, b in zip(points, (*points[1:], points[0]), strict=True)
        )
        if apothem > 0:
            self.band(points, scaled(points, center, max(0.0, 1 - width / apothem)), color)

    def rect(self, x: float, y: float, w: float, h: float, color: Color) -> None:
        self.polygon(((x, y), (x + w, y), (x + w, y + h), (x, y + h)), color)

    def rounded_rect(
        self, x: float, y: float, w: float, h: float, radius: float, color: Color
    ) -> None:
        self.polygon(rounded_rect_points(x, y, w, h, radius), color)

    def rounded_frame(
        self, x: float, y: float, w: float, h: float, radius: float, width: float, color: Color
    ) -> None:
        """Border of a rounded rectangle, `width` pixels thick, inside its bounds."""
        outer = rounded_rect_points(x, y, w, h, radius)
        d = min(width, w / 2, h / 2)
        inner = rounded_rect_points(x + d, y + d, w - 2 * d, h - 2 * d, max(0.0, radius - d))
        self.band(outer, inner, color)

    def circle(self, x: float, y: float, r: float, color: Color, segments: int = 0) -> None:
        self.polygon(circle_points(x, y, r, segments), color)

    def ring(
        self, x: float, y: float, r: float, width: float, color: Color, segments: int = 0
    ) -> None:
        """Circle outline: `width` pixels inside radius `r`."""
        n = segments or segments_for(r)
        self.band(circle_points(x, y, r, n), circle_points(x, y, max(0.0, r - width), n), color)

    def sector(
        self, x: float, y: float, r: float, start: float, end: float, color: Color, ry: float = 0
    ) -> None:
        """Filled pie slice from `start` to `end` degrees (counter-clockwise from +x); `ry`
        squashes it into an ellipse."""
        ry = ry or r
        n = max(3, int(segments_for(r) * (end - start) / 360) + 1)
        angles = [radians(start + (end - start) * k / n) for k in range(n + 1)]
        self.polygon([(x, y), *((x + r * cos(a), y + ry * sin(a)) for a in angles)], color)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float,
        color: Color,
        round_caps: bool = False,
    ) -> None:
        length = hypot(x2 - x1, y2 - y1)
        if length > 0:
            px, py = -(y2 - y1) / length * width / 2, (x2 - x1) / length * width / 2
            a, b = (x1 + px, y1 + py), (x2 + px, y2 + py)
            c, d = (x2 - px, y2 - py), (x1 - px, y1 - py)
            self.triangle(a, b, c, color)
            self.triangle(a, c, d, color)
        if round_caps:
            self.circle(x1, y1, width / 2, color)
            self.circle(x2, y2, width / 2, color)

    # ------------------------------------------------------------------ text

    def text(
        self,
        text: str,
        x: float,
        y: float,
        size: float = theme.FONT,
        color: Color = theme.TEXT,
        bold: bool = False,
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> None:
        """Text in Arcade's default font; positions are rounded to whole pixels to stay crisp."""
        self.texts.append(TextItem(text, round(x), round(y), size, color, bold, anchor_x, anchor_y))


class _Mesh:
    def __init__(self, buffer: "Buffer", geometry: "Geometry") -> None:
        self.buffer = buffer
        self.geometry = geometry
        self.key: Hashable = None
        self.count = 0


class _Label:
    __slots__ = ("label", "used", "visible", "x", "y")

    def __init__(self, label: "arcade.Text", frame: int, x: float, y: float) -> None:
        self.label = label
        self.used = frame
        self.visible = True
        self.x, self.y = x, y


class _TextBatch:
    """Labels of one layer, kept between frames and drawn as a single pyglet batch.

    Labels are keyed by content and style (not position), so moving text only updates it; labels
    not drawn in a frame are hidden, and deleted after a while.

    On high-DPI screens glyphs are rasterised at `scale` (the window's pixel ratio) and placed on
    whole physical pixels, then drawn under a 1/scale view: text stays as crisp as the screen.
    """

    EXPIRY = 600

    def __init__(self, scale: float = 1.0) -> None:
        import pyglet

        self.batch = pyglet.graphics.Batch()
        self.labels: dict[tuple, _Label] = {}
        self.scale = scale

    def draw(self, items: Iterable[TextItem], frame: int) -> None:
        import arcade

        k = self.scale
        count: dict[tuple, int] = {}
        for item in items:
            style = (item.text, item.size, item.color, item.bold, item.anchor_x, item.anchor_y)
            n = count[style] = count.get(style, -1) + 1
            entry = self.labels.get((style, n))
            x, y = round(item.x * k), round(item.y * k)
            if entry is None:
                label = arcade.Text(
                    item.text,
                    x,
                    y,
                    item.color,
                    item.size * k,
                    bold=item.bold,
                    anchor_x=item.anchor_x,
                    anchor_y=item.anchor_y,
                    batch=self.batch,
                )
                self.labels[(style, n)] = _Label(label, frame, x, y)
                continue
            if (entry.x, entry.y) != (x, y):
                entry.x, entry.y = x, y
                entry.label.position = x, y
            if not entry.visible:
                entry.label.visible = entry.visible = True
            entry.used = frame
        for key, entry in list(self.labels.items()):
            if entry.used != frame:
                if entry.visible:
                    entry.label.visible = entry.visible = False
                if frame - entry.used > self.EXPIRY:
                    entry.label.label.delete()
                    del self.labels[key]
        if k == 1:
            self.batch.draw()
            return
        from pyglet.math import Mat4, Vec3

        window = arcade.get_window()
        view = window.view
        window.view = view @ Mat4.from_scale(Vec3(1 / k, 1 / k, 1))
        try:
            self.batch.draw()
        finally:
            window.view = view


class Painter:
    """Draws `Layer`s with Arcade (needs an open window): one vertex buffer per layer name,
    re-uploaded every frame unless the layer is static, and one label batch per layer name."""

    def __init__(self) -> None:
        import arcade

        self.ctx = arcade.get_window().ctx
        self._program = self.ctx.line_generic_with_colors_program
        self._meshes: dict[str, _Mesh] = {}
        self._texts: dict[str, _TextBatch] = {}
        self.frame = 0

    def pixel_ratio(self) -> float:
        import arcade

        return arcade.get_window().get_pixel_ratio() or 1.0

    def draw(self, layers: Sequence[Layer]) -> None:
        self.frame += 1
        ratio = self.pixel_ratio()
        if any(batch.scale != ratio for batch in self._texts.values()):
            self._texts = {}  # moved to a screen with another scale: rasterise again
        names = set()
        for layer in layers:
            names.add(layer.name)
            self._draw_mesh(layer)
            batch = self._texts.get(layer.name)
            if batch is None:
                batch = self._texts[layer.name] = _TextBatch(ratio)
            batch.draw(layer.texts, self.frame)
        for name, batch in self._texts.items():
            if name not in names:  # hide the text of layers missing from this frame
                batch.draw((), self.frame)

    def _draw_mesh(self, layer: Layer) -> None:
        from arcade.gl import BufferDescription

        ctx = self.ctx
        mesh = self._meshes.get(layer.name)
        if mesh is None:
            buffer = ctx.buffer(reserve=24 * 4096, usage="dynamic")
            description = BufferDescription(buffer, "2f 4f", ("in_vert", "in_color"))
            mesh = self._meshes[layer.name] = _Mesh(buffer, ctx.geometry([description]))
        if layer.key is None or layer.key != mesh.key:
            data = array("f", layer.data)
            size = len(data) * 4
            mesh.buffer.orphan(
                size=max(size, mesh.buffer.size * 2) if size > mesh.buffer.size else -1
            )
            if size:
                mesh.buffer.write(data)
            mesh.key, mesh.count = layer.key, layer.vertex_count
        if mesh.count:
            ctx.enable(ctx.BLEND)
            # Usual alpha blending for colours; alpha itself accumulates, so the window stays
            # opaque (screenshots included).
            ctx.blend_func = (
                ctx.SRC_ALPHA,
                ctx.ONE_MINUS_SRC_ALPHA,
                ctx.ONE,
                ctx.ONE_MINUS_SRC_ALPHA,
            )
            mesh.geometry.render(self._program, mode=ctx.TRIANGLES, vertices=mesh.count)
