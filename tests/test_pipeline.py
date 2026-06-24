"""Deterministic tests for orchestration/pipeline.py (ADR-0006 b4.1)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lutgen.engine.base import load_base
from lutgen.orchestration.pipeline import render_cube


def _warm_png(path, seed):
    rng = np.random.default_rng(seed)
    content = rng.random((40, 40, 3)) * 0.6 + 0.2
    looked = np.clip(content + np.array([0.18, 0.0, -0.12]), 0, 1)
    Image.fromarray((looked * 255).astype(np.uint8), "RGB").save(path)


def _refs(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"ref{i}.png"
        _warm_png(p, i)
        paths.append(p)
    return paths


def test_strength_zero_is_base(tmp_path):
    cube = render_cube(_refs(tmp_path), 0.0)
    np.testing.assert_array_equal(cube.samples, load_base())  # Golden Rule end-to-end


def test_render_valid_and_warm(tmp_path):
    base = load_base()
    cube = render_cube(_refs(tmp_path), 1.0, title="warm")
    assert cube.size == 65 and cube.samples.shape == (274625, 3)
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    assert cube.title == "warm"
    warmth = lambda x: x[:, 0].mean() - x[:, 2].mean()
    assert warmth(cube.samples) > warmth(base)


def _neutral_png(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((40, 40, 3)) * 255).astype(np.uint8), "RGB").save(path)


def test_dual_pool_unpaired(tmp_path):
    from lutgen.fitter.rich import RichFitter
    from lutgen.orchestration.pipeline import render_cube_dual

    # 3 neutral + 2 graded (unequal counts, unpaired)
    src = []
    for i in range(3):
        p = tmp_path / f"n{i}.png"; _neutral_png(p, 100 + i); src.append(p)
    tgt = _refs(tmp_path, 2)  # warm graded
    base = load_base()
    cube = render_cube_dual(src, tgt, 1.0, fitter=RichFitter(space="rgb"))
    assert cube.samples.shape == (274625, 3)
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    warmth = lambda x: x[:, 0].mean() - x[:, 2].mean()
    assert warmth(cube.samples) > warmth(base)                       # target look imposed
    np.testing.assert_array_equal(
        render_cube_dual(src, tgt, 0.0, fitter=RichFitter(space="rgb")).samples, base)  # s0==base


def test_placement_between(tmp_path):
    from lutgen.engine.apply import apply_cube
    from lutgen.engine.grid import identity_grid
    from lutgen.fitter.rich import RichFitter

    refs = _refs(tmp_path)
    base = load_base()
    grid = identity_grid()
    # between + strength 0 → identity grid (pass-through; Node 2 unchanged)
    c0 = render_cube(refs, 0.0, placement="between")
    np.testing.assert_allclose(c0.samples, grid, atol=1e-9)
    # between-cube then Node 2 ≈ the replace-Node-2 cube (same final look)
    cb = render_cube(refs, 1.0, placement="between", fitter=RichFitter(space="rgb"))
    cn = render_cube(refs, 1.0, placement="node2", fitter=RichFitter(space="rgb"))
    chained = apply_cube(cb.samples, base)            # apply our between cube, then Node 2
    assert np.abs(chained - cn.samples).mean() < 0.05
    assert cb.samples.min() >= 0.0 and cb.samples.max() <= 1.0


def test_strength_scales_effect(tmp_path):
    refs = _refs(tmp_path)
    base = load_base()
    d = lambda s: np.abs(render_cube(refs, s).samples - base).mean()
    assert d(0.25) < d(0.5) < d(1.0)  # more strength = further from base
