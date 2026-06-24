"""Tests for the inverse base cube (Rec.709 → DWG/DI), used by the 'between' placement.
ADR-0009/0017/0018."""

from __future__ import annotations

import numpy as np

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import INVERSE_SIZE, load_base, load_base_inverse
from lutgen.engine.grid import identity_grid


def test_inverse_asset_shape_range():
    inv = load_base_inverse()
    assert inv.shape == (INVERSE_SIZE ** 3, 3)        # higher-resolution (65-point)
    assert inv.min() >= 0.0 and inv.max() <= 1.0


def test_inverse_is_left_inverse_of_base():
    # base maps DWG/DI→Rec.709; the inverse should bring it back near the original grid.
    grid = identity_grid()
    rec709 = apply_cube(grid, load_base())
    back = apply_cube(rec709, load_base_inverse(), INVERSE_SIZE)
    err = np.abs(back - grid).mean()
    assert err < 0.045                                 # inverse is reasonable (highlights ill-conditioned)
