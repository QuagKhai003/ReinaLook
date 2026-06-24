"""Deterministic tests for engine/base.py — the protected base layer (ADR-0003)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import load_base
from lutgen.engine.cube_io import read_cube
from lutgen.engine.grid import reshape_to_lattice


def test_shape_and_range():
    base = load_base()
    assert base.shape == (33 ** 3, 3)
    assert base.dtype == np.float64
    assert base.min() >= 0.0 and base.max() <= 1.0  # Resolve output, clamped


def test_black_maps_to_black():
    np.testing.assert_allclose(load_base()[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_neutral_axis_stays_neutral():
    # Greys must stay neutral (no cast) in the verified base.
    lat = reshape_to_lattice(load_base())
    step = 33 - 1
    for k in (0, 8, 16, 24, 32):
        rgb = lat[k, k, k]  # lattice[blue,green,red]: the grey node has b=g=r=k
        np.testing.assert_allclose(rgb, [rgb[0]] * 3, atol=2e-3)


def test_matches_bundled_asset():
    from importlib.resources import as_file, files

    src = files("lutgen.engine").joinpath("data", "base_dwg_di_to_rec709_g24.cube")
    with as_file(src) as p:
        cube = read_cube(p)
    np.testing.assert_array_equal(load_base(), cube.samples)


def test_returns_fresh_writable_copy():
    a = load_base()
    a[0] = [9.0, 9.0, 9.0]  # must not corrupt the cached asset
    b = load_base()
    np.testing.assert_allclose(b[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_highlight_rolloff_present():
    # Sanity: the base rolls off (does NOT clip ~code 0.5 like the pure synth). Grey at code
    # ~0.5 should be well below 1.0 — evidence the DaVinci tone map is baked in.
    lat = reshape_to_lattice(load_base())
    mid = lat[16, 16, 16][0]  # ~code 0.5 grey
    assert 0.6 < mid < 0.95
