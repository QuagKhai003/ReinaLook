"""Deterministic tests for engine/grid.py (ADR-0001 batch 0.1)."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.grid import (
    DEFAULT_SIZE,
    flatten_lattice,
    identity_grid,
    reshape_to_lattice,
)


def test_shape_and_count():
    g = identity_grid()
    assert DEFAULT_SIZE == 33
    assert g.shape == (33 ** 3, 3) == (35937, 3)
    assert g.dtype == np.float64


def test_range_endpoints():
    g = identity_grid()
    assert g.min() == 0.0 and g.max() == 1.0
    # first node is black, last is white
    np.testing.assert_array_equal(g[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(g[-1], [1.0, 1.0, 1.0])


def test_red_fastest_ordering():
    # Resolve convention: along the first rows only RED changes (green, blue held at 0).
    size = 4
    g = identity_grid(size)
    step = 1.0 / (size - 1)
    # rows 0..size-1: green=0, blue=0, red ramps 0..1
    for r in range(size):
        np.testing.assert_allclose(g[r], [r * step, 0.0, 0.0])
    # row `size`: red wrapped to 0, green ticked one step
    np.testing.assert_allclose(g[size], [0.0, step, 0.0])
    # row size*size: green wrapped, blue ticked one step (blue slowest)
    np.testing.assert_allclose(g[size * size], [0.0, 0.0, step])


def test_reshape_flatten_round_trip():
    g = identity_grid()
    lat = reshape_to_lattice(g)
    assert lat.shape == (33, 33, 33, 3)
    np.testing.assert_array_equal(flatten_lattice(lat), g)


def test_lattice_indexing_matches_coords():
    # lattice is indexed [blue, green, red]; value at [bi,gi,ri] == (ri,gi,bi)/(size-1).
    size = 5
    lat = reshape_to_lattice(identity_grid(size), size)
    step = 1.0 / (size - 1)
    for bi in (0, 2, 4):
        for gi in (0, 1, 4):
            for ri in (0, 3, 4):
                np.testing.assert_allclose(lat[bi, gi, ri], [ri * step, gi * step, bi * step])


def test_size_validation():
    with pytest.raises(ValueError):
        identity_grid(1)
    with pytest.raises(ValueError):
        reshape_to_lattice(identity_grid(33), 17)
