"""Deterministic tests for engine/apply.py (ADR-0007 b5.1)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import load_base
from lutgen.engine.grid import identity_grid


def test_reproduces_lut_at_nodes():
    base = load_base()
    # feeding the grid coordinates through the cube returns the cube samples (linear interp
    # is exact at the nodes).
    out = apply_cube(identity_grid().reshape(-1, 1, 3), base).reshape(-1, 3)
    np.testing.assert_allclose(out, base, atol=1e-9)


def test_shape_preserved():
    base = load_base()
    img = np.random.default_rng(0).random((12, 20, 3))
    out = apply_cube(img, base)
    assert out.shape == (12, 20, 3)
    assert np.isfinite(out).all()


def test_gray_ramp_monotone():
    base = load_base()
    ramp = np.linspace(0, 1, 64)
    img = np.stack([ramp, ramp, ramp], axis=-1)
    out = apply_cube(img, base)
    luma = out @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luma) >= -1e-9)


def test_bad_shape_raises():
    import pytest

    with pytest.raises(ValueError):
        apply_cube(np.zeros((4, 2)), load_base())
