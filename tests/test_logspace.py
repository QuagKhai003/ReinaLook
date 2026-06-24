"""Tests for the log-space (between Node 1 & 2) look pipeline (ADR-0009)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import load_base, load_base_inverse
from lutgen.engine.grid import identity_grid
from lutgen.orchestration.pipeline import render_look_cube


def _warm_png(path, seed):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.random((40, 40, 3)) * 0.6 + np.array([0.15, 0.0, -0.1]), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8), "RGB").save(path)


def _refs(tmp_path, n=3):
    out = []
    for i in range(n):
        p = tmp_path / f"r{i}.png"
        _warm_png(p, i)
        out.append(p)
    return out


def test_inverse_asset_shape_range():
    inv = load_base_inverse()
    assert inv.shape == (33 ** 3, 3)
    assert inv.min() >= 0.0 and inv.max() <= 1.0


def test_inverse_is_left_inverse_of_base():
    # base maps DWG/DI->Rec709; inverse should bring it back near the original grid.
    grid = identity_grid()
    rec709 = apply_cube(grid, load_base())
    back = apply_cube(rec709, load_base_inverse())
    # interpolated double-LUT round trip: loose tolerance, ignore gamut-edge nodes
    err = np.abs(back - grid).mean()
    assert err < 0.05


def test_strength_zero_is_passthrough(tmp_path):
    cube = render_look_cube(_refs(tmp_path), 0.0)
    np.testing.assert_allclose(cube.samples, identity_grid(), atol=1e-9)  # identity → Node2 unchanged


def test_look_applied_at_full_strength(tmp_path):
    grid = identity_grid()
    cube = render_look_cube(_refs(tmp_path), 1.0, tone_strength=0.0)
    assert cube.samples.shape == (35937, 3)
    assert not np.allclose(cube.samples, grid)        # the look changed something
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0


def test_tone_zero_mostly_preserves_grid_luma(tmp_path):
    # In DWG/DI, tone_strength=0 keeps luma (color/sat only); clamping at gamut edges shifts a
    # few saturated nodes, so check the bulk stays put.
    grid = identity_grid()
    w = np.array([0.2126, 0.7152, 0.0722])
    cube = render_look_cube(_refs(tmp_path), 1.0, tone_strength=0.0)
    luma_diff = np.abs(cube.samples @ w - grid @ w)
    assert np.median(luma_diff) < 1e-6        # most nodes: luma untouched
    assert luma_diff.mean() < 0.02            # only gamut-edge clamping moves a few
