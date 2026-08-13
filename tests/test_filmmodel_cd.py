"""Deterministic tests for v2 film model Blocks C (sat-vs-luma) + D (hue zones) and the
DI<->Oklab bridge — ADR-0001 b1.2.

Identity@0 bit-for-bit stays the Golden-Rule guarantee. Smoothness/periodicity tests keep
zone boundaries free of hue breaks (spec §6)."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.perceptual import di_to_oklab, oklab_to_di
from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
    apply_hue_zones,
    apply_sat_luma,
    sat_multiplier,
)
from lutgen.fitter.filmmodel.huezone import _interp_periodic


def _lab_grid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lab = np.empty((512, 3))
    lab[:, 0] = rng.uniform(0.0, 1.0, 512)          # L
    lab[:, 1:] = rng.uniform(-0.25, 0.25, (512, 2))  # a, b (realistic chroma range)
    return lab


# ── DI <-> Oklab bridge ───────────────────────────────────────────────

def test_di_oklab_roundtrip_exact():
    rng = np.random.default_rng(1)
    di = rng.uniform(0.0, 1.0, (1024, 3))
    np.testing.assert_allclose(oklab_to_di(di_to_oklab(di)), di, atol=1e-12)


def test_di_oklab_neutral_axis_has_zero_chroma():
    # atol 1e-6: the Bradford-adapted DWG->709 matrix rows don't sum to exactly 1 in float64,
    # leaving ~1e-7 residual chroma on the gray axis — numerical floor, far below visibility.
    gray = np.tile(np.linspace(0.05, 0.95, 16)[:, None], (1, 3))
    lab = di_to_oklab(gray)
    np.testing.assert_allclose(lab[:, 1:], 0.0, atol=1e-6)


# ── Block C: sat-vs-luma ──────────────────────────────────────────────

def test_satluma_identity_is_bit_for_bit():
    lab = _lab_grid()
    np.testing.assert_array_equal(apply_sat_luma(lab, SatLumaParams()), lab)


def test_satluma_scales_chroma_leaves_l():
    lab = _lab_grid(2)
    out = apply_sat_luma(lab, SatLumaParams(shadow=0.5, mid=1.5, high=0.8))
    np.testing.assert_array_equal(out[:, 0], lab[:, 0])  # L untouched
    # chroma ratio equals the multiplier at that L
    mult = sat_multiplier(lab[:, 0], SatLumaParams(shadow=0.5, mid=1.5, high=0.8))
    np.testing.assert_allclose(np.hypot(out[:, 1], out[:, 2]),
                               np.hypot(lab[:, 1], lab[:, 2]) * mult, atol=1e-12)


def test_satluma_curve_hits_anchors():
    p = SatLumaParams(shadow=0.4, mid=1.3, high=0.7)
    np.testing.assert_allclose(sat_multiplier(np.array([0.0, 0.5, 1.0]), p),
                               [0.4, 1.3, 0.7], atol=1e-12)


def test_satluma_curve_c1_smooth():
    p = SatLumaParams(shadow=0.3, mid=1.8, high=0.5)
    x = np.linspace(0.0, 1.0, 2001)
    d2 = np.diff(sat_multiplier(x, p), 2)
    assert np.max(np.abs(d2)) < 5e-5  # no kink at the mid anchor


def test_satluma_never_negative_chroma():
    p = SatLumaParams(shadow=-2.0, mid=0.0, high=1.0)  # abusive params
    mult = sat_multiplier(np.linspace(0, 1, 100), p)
    assert np.all(mult >= 0.0)


def test_satluma_flat_extrapolation():
    p = SatLumaParams(shadow=0.4, mid=1.0, high=1.6)
    np.testing.assert_allclose(sat_multiplier(np.array([-0.5, 1.5]), p), [0.4, 1.6], atol=1e-12)


# ── Block D: hue zones ────────────────────────────────────────────────

def test_huezone_identity_is_bit_for_bit():
    lab = _lab_grid(3)
    np.testing.assert_array_equal(apply_hue_zones(lab, HueZoneParams()), lab)


def test_huezone_rotates_hue_preserves_l_and_chroma():
    lab = _lab_grid(4)
    p = HueZoneParams(r_shift=0.2, g_shift=-0.1, b_shift=0.15)
    out = apply_hue_zones(lab, p)
    np.testing.assert_array_equal(out[:, 0], lab[:, 0])  # L untouched
    # pure shifts (no trims) preserve chroma exactly
    np.testing.assert_allclose(np.hypot(out[:, 1], out[:, 2]),
                               np.hypot(lab[:, 1], lab[:, 2]), atol=1e-12)


def test_huezone_trim_scales_chroma():
    lab = _lab_grid(5)
    out = apply_hue_zones(lab, HueZoneParams(r_trim=0.5, y_trim=0.5, g_trim=0.5,
                                             c_trim=0.5, b_trim=0.5, m_trim=0.5))
    # uniform +50% trim everywhere -> chroma exactly 1.5x
    np.testing.assert_allclose(np.hypot(out[:, 1], out[:, 2]),
                               1.5 * np.hypot(lab[:, 1], lab[:, 2]), atol=1e-12)


def test_huezone_achromatic_fixed_point():
    gray = np.column_stack([np.linspace(0, 1, 16), np.zeros(16), np.zeros(16)])
    p = HueZoneParams(r_shift=0.3, b_trim=0.4)
    np.testing.assert_array_equal(apply_hue_zones(gray, p), gray)


def test_huezone_interp_periodic_no_wrap_seam():
    vals = np.array([0.3, -0.2, 0.1, 0.0, 0.25, -0.15])
    h = np.linspace(-np.pi, np.pi, 4001)
    y = _interp_periodic(h, vals)
    # continuous everywhere, including the -pi/pi wrap
    assert np.max(np.abs(np.diff(y))) < 5e-3
    # endpoints meet (same angle mod 2pi)
    np.testing.assert_allclose(y[0], y[-1], atol=1e-9)


def test_huezone_interp_plateaus_at_centres():
    from lutgen.fitter.filmmodel.huezone import ZONE_ANGLES
    vals = np.array([0.3, -0.2, 0.1, 0.0, 0.25, -0.15])
    np.testing.assert_allclose(_interp_periodic(ZONE_ANGLES, vals), vals, atol=1e-12)


def test_huezone_never_negative_chroma():
    lab = _lab_grid(6)
    out = apply_hue_zones(lab, HueZoneParams(r_trim=-3.0, g_trim=-3.0))  # abusive
    assert np.all(np.hypot(out[:, 1], out[:, 2]) >= 0.0)


# ── FilmModel: full A->B->C->D composition ────────────────────────────

def test_model_full_identity_is_bit_for_bit():
    rng = np.random.default_rng(7)
    rgb = rng.uniform(0.0, 1.0, (512, 3))
    np.testing.assert_array_equal(FilmModel.identity().forward(rgb), rgb)


def test_model_ab_only_skips_oklab_roundtrip():
    # A/B-only model must equal the 1.1 behaviour exactly (no conversion error added)
    rng = np.random.default_rng(8)
    rgb = rng.uniform(0.0, 1.0, (256, 3))
    ct = CrosstalkParams(gr=0.05)
    curves = (SCurveParams(slope=1.4), SCurveParams(), SCurveParams())
    from lutgen.fitter.filmmodel import apply_crosstalk, apply_scurve
    expected = apply_scurve(apply_crosstalk(rgb, ct), curves)
    np.testing.assert_array_equal(FilmModel(crosstalk=ct, curves=curves).forward(rgb), expected)


def test_model_full_pipeline_order():
    # C/D run in CODE-SPACE Oklab (Oklab on the DI code values — bounded and sane at the
    # lattice corners; scene-referred Oklab exploded there, found by the b1.7 stress harness)
    from lutgen.engine.perceptual import from_oklab, to_oklab

    rng = np.random.default_rng(9)
    rgb = rng.uniform(0.05, 0.95, (256, 3))
    ct = CrosstalkParams(rg=0.03)
    curves = (SCurveParams(toe=0.2), SCurveParams(), SCurveParams(shoulder=0.3))
    sl = SatLumaParams(mid=1.3)
    hz = HueZoneParams(r_shift=0.1)
    model = FilmModel(crosstalk=ct, curves=curves, sat_luma=sl, hue_zones=hz)
    from lutgen.fitter.filmmodel import apply_crosstalk, apply_scurve
    x = apply_scurve(apply_crosstalk(rgb, ct), curves)
    expected = from_oklab(apply_hue_zones(apply_sat_luma(to_oklab(x), sl), hz))
    np.testing.assert_array_equal(model.forward(rgb), expected)


def test_model_cd_neutral_gray_stays_neutral():
    # full model incl. C+D on the gray axis: chroma stays ~0 (no hue cast on neutrals)
    ct = CrosstalkParams(rg=0.03, gr=0.03, rb=0.03, br=0.03, gb=0.03, bg=0.03)
    p = SCurveParams(toe=0.3, shoulder=0.3, slope=1.5)
    model = FilmModel(crosstalk=ct, curves=(p, p, p),
                      sat_luma=SatLumaParams(shadow=0.7, mid=1.4, high=0.8),
                      hue_zones=HueZoneParams(r_shift=0.2, b_trim=0.3))
    gray = np.tile(np.linspace(0.05, 0.95, 16)[:, None], (1, 3))
    out = model.forward(gray)
    np.testing.assert_allclose(out[:, 0], out[:, 1], atol=1e-9)
    np.testing.assert_allclose(out[:, 1], out[:, 2], atol=1e-9)


def test_model_bad_shape_raises():
    with pytest.raises(ValueError):
        FilmModel(sat_luma=SatLumaParams(mid=1.2)).forward(np.zeros((4, 2)))
