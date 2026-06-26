"""Tests for engine/film.py film-stock transfer (ADR-0021)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.film import FilmStock, apply_film

_W = np.array([0.2126, 0.7152, 0.0722])


def test_identity_is_bit_exact():
    x = np.random.default_rng(0).random((500, 3))
    np.testing.assert_array_equal(apply_film(x, FilmStock()), x)
    assert FilmStock().is_identity()


def test_contrast_scurve_increases_spread():
    x = np.linspace(0, 1, 256)[:, None].repeat(3, 1)
    out = apply_film(x, FilmStock(contrast=0.8))
    assert out.std() > x.std()                       # S-curve adds mid contrast
    # may transiently exceed [0,1] (regularize clamps the final cube); endpoints stay anchored
    np.testing.assert_allclose(out[[0, -1]], x[[0, -1]], atol=0.1)


def test_toe_lifts_shadows_only():
    x = np.linspace(0, 1, 256)[:, None].repeat(3, 1)
    out = apply_film(x, FilmStock(toe=1.0))
    assert out[5].mean() > x[5].mean()               # deep shadow lifted
    np.testing.assert_allclose(out[230:], x[230:], atol=1e-6)  # highlights untouched


def test_shoulder_rolls_highlights():
    x = np.linspace(0, 1, 256)[:, None].repeat(3, 1)
    out = apply_film(x, FilmStock(shoulder=1.0))
    assert out[-1].max() < 1.0                        # white pulled below 1
    np.testing.assert_allclose(out[:120], x[:120], atol=1e-6)   # below the knee untouched


def test_highlight_bleach_desaturates_brights():
    hi = np.array([[0.9, 0.5, 0.4]])                  # saturated bright
    lo = np.array([[0.2, 0.1, 0.08]])                 # saturated dark
    chroma = lambda a: float((a.max(1) - a.min(1))[0])
    assert chroma(apply_film(hi, FilmStock(highlight_bleach=1.0))) < chroma(hi)   # brights wash out
    np.testing.assert_allclose(apply_film(lo, FilmStock(highlight_bleach=1.0)), lo, atol=1e-3)


def test_split_tone_warms_highs_cools_shadows():
    hi = np.full((20, 3), 0.8); lo = np.full((20, 3), 0.2)
    rb = lambda a: float(a[:, 0].mean() - a[:, 2].mean())
    assert rb(apply_film(hi, FilmStock(split_warm=1.0))) > 0   # highlights warm
    assert rb(apply_film(lo, FilmStock(split_warm=1.0))) < 0   # shadows cool


def test_pipeline_film_only_no_refs(tmp_path):
    from lutgen.engine.base import load_base
    from lutgen.orchestration.pipeline import render_cube

    base = load_base()
    cube = render_cube([], 1.0, film=FilmStock(contrast=0.5, shoulder=0.4, split_warm=0.3))
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    assert not np.array_equal(cube.samples, base)                       # film reshaped the base
    np.testing.assert_array_equal(render_cube([], 1.0, film=FilmStock()).samples, base)  # identity
    np.testing.assert_array_equal(
        render_cube([], 0.0, film=FilmStock(contrast=1.0)).samples, base)   # s=0 == base
