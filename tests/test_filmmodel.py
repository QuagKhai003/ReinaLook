"""Deterministic tests for the v2 film model — Blocks A (crosstalk) + B (S-curves), ADR-0001 b1.1.

The identity@0 tests are the Golden-Rule guarantee: neutral params must return the input
bit-for-bit so strength=0 stays the sacred base. Monotonicity + C1 smoothness keep baked LUTs
free of tone reversals and banding.
"""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    SCurveParams,
    apply_crosstalk,
    apply_scurve,
    crosstalk_matrix,
)

_GRID = np.linspace(0.0, 1.0, 257, dtype=np.float64)


def _rgb_grid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(512, 3))


# ── Block A: crosstalk ────────────────────────────────────────────────

def test_crosstalk_identity_is_bit_for_bit():
    rgb = _rgb_grid()
    np.testing.assert_array_equal(apply_crosstalk(rgb, CrosstalkParams()), rgb)


def test_crosstalk_identity_matrix_exact():
    np.testing.assert_array_equal(crosstalk_matrix(CrosstalkParams()), np.eye(3))


def test_crosstalk_rows_sum_to_one():
    m = crosstalk_matrix(CrosstalkParams(rg=0.04, rb=-0.02, gr=0.03, gb=0.01, br=-0.05, bg=0.02))
    np.testing.assert_allclose(m.sum(axis=1), np.ones(3), atol=1e-15)


def test_crosstalk_preserves_neutral_axis():
    # r=g=b in -> r=g=b out, because every row sums to 1 (energy preserving)
    p = CrosstalkParams(rg=0.06, rb=0.03, gr=0.02, gb=0.05, br=0.04, bg=0.01)
    gray = np.tile(np.linspace(0, 1, 20)[:, None], (1, 3))
    out = apply_crosstalk(gray, p)
    np.testing.assert_allclose(out, gray, atol=1e-14)


def test_crosstalk_actually_mixes():
    p = CrosstalkParams(gr=0.1)  # green leaks into red
    out = apply_crosstalk(np.array([[0.2, 0.8, 0.2]]), p)
    assert out[0, 0] > 0.2  # red lifted by green's contribution


def test_crosstalk_bad_shape_raises():
    with pytest.raises(ValueError):
        apply_crosstalk(np.zeros((4, 2)), CrosstalkParams(rg=0.1))


# ── Block B: per-channel S-curves ─────────────────────────────────────

def test_scurve_identity_is_bit_for_bit():
    rgb = _rgb_grid(1)
    ident = (SCurveParams(), SCurveParams(), SCurveParams())
    np.testing.assert_array_equal(apply_scurve(rgb, ident), rgb)


def test_scurve_endpoints_fixed():
    # f(0)=0 and f(1)=1 for any params — black and white points preserved
    p = SCurveParams(toe=0.4, shoulder=0.6, slope=1.8)
    curves = (p, p, p)
    ends = apply_scurve(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), curves)
    np.testing.assert_allclose(ends[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(ends[1], 1.0, atol=1e-12)


def test_scurve_passes_through_pivot():
    p = SCurveParams(toe=0.3, shoulder=0.5, slope=1.5, pivot=0.42)
    out = apply_scurve(np.full((1, 3), 0.42), (p, p, p))
    np.testing.assert_allclose(out, 0.42, atol=1e-12)


def _channel_curve(p: SCurveParams) -> np.ndarray:
    col = np.repeat(_GRID[:, None], 3, axis=1)
    return apply_scurve(col, (p, p, p))[:, 0]


@pytest.mark.parametrize(
    "p",
    [
        SCurveParams(toe=0.5, shoulder=0.5, slope=1.9),
        SCurveParams(toe=0.0, shoulder=0.8, slope=0.6),
        SCurveParams(toe=0.9, shoulder=0.1, slope=1.2, pivot=0.35),
        SCurveParams(slope=2.0),
    ],
)
def test_scurve_strictly_monotonic(p):
    y = _channel_curve(p)
    assert np.all(np.diff(y) > -1e-12)  # non-decreasing, no reversals


@pytest.mark.parametrize(
    "p",
    [
        SCurveParams(toe=0.5, shoulder=0.5, slope=1.9),
        SCurveParams(toe=0.9, shoulder=0.1, slope=1.2, pivot=0.35),
    ],
)
def test_scurve_c1_smooth_no_kinks(p):
    # second difference stays bounded — a kink (C0-only join) would spike it
    y = _channel_curve(p)
    d2 = np.diff(y, 2)
    assert np.max(np.abs(d2)) < 5e-3


def test_scurve_channels_independent():
    # a curve on R only must leave G, B untouched — the mechanism behind film hue drift
    rgb = _rgb_grid(2)
    curves = (SCurveParams(slope=1.7), SCurveParams(), SCurveParams())
    out = apply_scurve(rgb, curves)
    np.testing.assert_array_equal(out[:, 1:], rgb[:, 1:])
    assert not np.allclose(out[:, 0], rgb[:, 0])


def test_scurve_s_shape_contrast():
    # slope>1 with soft ends: below pivot pulled down, above pivot pushed up (more contrast)
    p = SCurveParams(toe=0.4, shoulder=0.4, slope=1.8, pivot=0.5)
    y = _channel_curve(p)
    lo = _GRID < 0.5
    hi = _GRID > 0.5
    assert np.all(y[lo] <= _GRID[lo] + 1e-9)
    assert np.all(y[hi] >= _GRID[hi] - 1e-9)


def test_scurve_extrapolation_monotonic_out_of_range():
    p = SCurveParams(toe=0.3, shoulder=0.5, slope=1.4)
    x = np.linspace(-0.2, 1.2, 400)
    y = apply_scurve(np.repeat(x[:, None], 3, axis=1), (p, p, p))[:, 0]
    assert np.all(np.diff(y) > -1e-12)


# ── FilmModel: A -> B composition ─────────────────────────────────────

def test_model_identity_is_bit_for_bit():
    rgb = _rgb_grid(3)
    np.testing.assert_array_equal(FilmModel.identity().forward(rgb), rgb)
    assert FilmModel().is_identity()


def test_model_applies_a_then_b():
    rgb = _rgb_grid(4)
    ct = CrosstalkParams(gr=0.05, rb=0.02)
    curves = (SCurveParams(slope=1.6), SCurveParams(toe=0.3), SCurveParams(shoulder=0.4))
    model = FilmModel(crosstalk=ct, curves=curves)
    expected = apply_scurve(apply_crosstalk(rgb, ct), curves)
    np.testing.assert_array_equal(model.forward(rgb), expected)


def test_model_forward_bad_shape_raises():
    with pytest.raises(ValueError):
        FilmModel(crosstalk=CrosstalkParams(rg=0.1)).forward(np.zeros((5, 2)))


def test_model_neutral_gray_stays_neutral():
    # full model with crosstalk + symmetric curves: gray axis stays gray (no hue cast on neutrals)
    ct = CrosstalkParams(rg=0.03, gr=0.03, rb=0.03, br=0.03, gb=0.03, bg=0.03)
    p = SCurveParams(toe=0.3, shoulder=0.3, slope=1.5)
    out = FilmModel(crosstalk=ct, curves=(p, p, p)).forward(
        np.tile(np.linspace(0, 1, 20)[:, None], (1, 3))
    )
    np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-12)
    np.testing.assert_allclose(out[:, 1], out[:, 2], atol=1e-12)
