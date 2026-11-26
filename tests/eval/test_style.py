import numpy as np
import pytest

from carla.eval.style import moving_average, run_color


def test_report_colour_and_trailing_average_are_stable():
    first = {name: run_color(name) for name in ("R8", "push", "KL")}
    second = {name: run_color(name) for name in ("KL", "R8", "push")}
    assert first == second
    np.testing.assert_allclose(moving_average([1, 3, 9, 11], 2), [1, 2, 6, 10])
    with pytest.raises(ValueError, match="positive"):
        moving_average([1], 0)
