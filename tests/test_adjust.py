"""Tests for engine/adjust.py creative adjustments (ADR-0020)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.adjust import Adjustments, apply_adjustments

_W = np.array([0.2126, 0.7152, 0.0722])


def _img(seed=0):
    return np.random.default_rng(seed).random((400, 3))


def test_identity_is_bit_exact():
    x = _img()
    np.testing.assert_array_equal(apply_adjustments(x, Adjustments()), x)
    assert Adjustments().is_identity()


def test_saturation_direction():
    x = _img()
    chroma = lambda a: float((a.max(1) - a.min(1)).mean())
    assert chroma(apply_adjustments(x, Adjustments(saturation=0.6))) > chroma(x)
    assert chroma(apply_adjustments(x, Adjustments(saturation=-1.0))) < chroma(x)  # toward grey


def test_temperature_warm_and_cool():
    x = _img()
    warm = apply_adjustments(x, Adjustments(temperature=0.8))
    cool = apply_adjustments(x, Adjustments(temperature=-0.8))
    rb = lambda a: float(a[:, 0].mean() - a[:, 2].mean())
    assert rb(warm) > rb(x) > rb(cool)


def test_tint_green_magenta():
    x = _img()
    mag = apply_adjustments(x, Adjustments(tint=0.8))     # magenta: G down
    grn = apply_adjustments(x, Adjustments(tint=-0.8))    # green: G up
    assert grn[:, 1].mean() > x[:, 1].mean() > mag[:, 1].mean()


def test_contrast_increases_spread():
    x = _img()
    out = apply_adjustments(x, Adjustments(contrast=0.8))
    assert out.std() > x.std()


def test_highlight_rolloff_compresses_brights():
    x = np.linspace(0, 1, 200)[:, None].repeat(3, 1)
    out = apply_adjustments(x, Adjustments(highlight_rolloff=1.0))
    assert out[-1].max() < 1.0                      # whites pulled below 1
    np.testing.assert_allclose(out[x[:, 0] < 0.7], x[x[:, 0] < 0.7], atol=1e-9)  # shadows untouched


def test_shadows_and_highlights():
    x = np.full((50, 3), 0.1)   # dark
    assert apply_adjustments(x, Adjustments(shadows=1.0)).mean() > x.mean()
    y = np.full((50, 3), 0.9)   # bright
    assert apply_adjustments(y, Adjustments(highlights=-1.0)).mean() < y.mean()


def test_manual_only_render(tmp_path):
    from lutgen.engine.base import load_base
    from lutgen.orchestration.pipeline import render_cube

    base = load_base()
    cube = render_cube([], 1.0, adjust=Adjustments(contrast=0.5, saturation=0.4))
    assert cube.samples.shape == base.shape
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    assert cube.samples.std() > base.std()                       # contrast applied
    # neutral adjust + no refs → base, bit-exact
    np.testing.assert_array_equal(render_cube([], 1.0, adjust=Adjustments()).samples, base)
    np.testing.assert_array_equal(render_cube([], 0.0, adjust=Adjustments(contrast=1.0)).samples, base)  # s0
