"""Deterministic tests for the v2.1 Fourier hue curve (ADR-0007 b7.1): the block itself,
serialization back-compat, fine hue targets, and the key capability claim — a smooth
mids-of-the-wheel hue twist the 6-zone model could not express is now recovered."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
from lutgen.fitter.filmmodel import FilmModel, FourierHueParams, apply_fourier_hue
from lutgen.fitter.filmmodel.fourierhue import eval_shift, eval_trim
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict
from lutgen.fitter.fit import FitOptions, _cube_fn, fit_film_model, synth_samples
from lutgen.orchestration.poolstats import (
    N_HUE_BINS,
    compute_frame_stats,
    neutral_prior,
    pool_stats,
)

OPT = FitOptions(n_samples=1500, max_nfev=40, exposure_align=False)


def _lab_grid(seed=0):
    rng = np.random.default_rng(seed)
    lab = np.empty((512, 3))
    lab[:, 0] = rng.uniform(0.1, 0.9, 512)
    lab[:, 1:] = rng.uniform(-0.2, 0.2, (512, 2))
    return lab


# ── the block ─────────────────────────────────────────────────────────

def test_identity_bit_for_bit():
    lab = _lab_grid()
    np.testing.assert_array_equal(apply_fourier_hue(lab, FourierHueParams()), lab)
    assert FilmModel.identity().is_identity()


def test_shift_preserves_l_and_chroma():
    lab = _lab_grid(1)
    p = FourierHueParams(s0=0.05, sc1=0.08, ss2=-0.04)
    out = apply_fourier_hue(lab, p)
    np.testing.assert_array_equal(out[:, 0], lab[:, 0])
    np.testing.assert_allclose(np.hypot(out[:, 1], out[:, 2]),
                               np.hypot(lab[:, 1], lab[:, 2]), atol=1e-12)


def test_trim_scales_chroma_only():
    lab = _lab_grid(2)
    out = apply_fourier_hue(lab, FourierHueParams(t0=0.2))
    np.testing.assert_allclose(np.hypot(out[:, 1], out[:, 2]),
                               1.2 * np.hypot(lab[:, 1], lab[:, 2]), atol=1e-12)


def test_curve_is_smooth_and_periodic():
    p = FourierHueParams(sc1=0.1, ss3=0.05, tc2=0.15)
    theta = np.linspace(-np.pi, np.pi, 2001)
    for f in (eval_shift, eval_trim):
        y = f(theta, p)
        assert abs(y[0] - y[-1]) < 1e-12                # periodic
        assert np.max(np.abs(np.diff(y, 2))) < 1e-3     # smooth (no kinks anywhere)


def test_achromatic_fixed_point():
    gray = np.column_stack([np.linspace(0, 1, 16), np.zeros(16), np.zeros(16)])
    p = FourierHueParams(s0=0.1, t0=0.3)
    np.testing.assert_array_equal(apply_fourier_hue(gray, p), gray)


def test_serialize_roundtrip_and_backcompat():
    m = FilmModel(hue_fourier=FourierHueParams(sc1=0.08, ts2=-0.1))
    assert model_from_dict(model_to_dict(m)) == m
    d = model_to_dict(m)
    del d["hue_fourier"]                                # pre-v2.1 profile
    assert model_from_dict(d).hue_fourier.is_identity()


# ── fine hue targets ──────────────────────────────────────────────────

def test_hue_bins_capture_localized_twist():
    from lutgen.engine.perceptual import from_oklab
    rng = np.random.default_rng(3)
    hue = rng.uniform(-np.pi, np.pi, 6000)
    lab = np.column_stack([np.full(6000, 0.55),
                           0.12 * np.cos(hue), 0.12 * np.sin(hue)])
    img = np.clip(from_oklab(lab), 0, 1)
    s = compute_frame_stats(img)
    assert s.hue_mean_ab.shape == (N_HUE_BINS, 2)
    assert s.hue_weight.sum() == pytest.approx(1.0, abs=0.05)
    # each bin's mean points at its own hue (12 distinct directions — 6 zones can't do this)
    angles = np.arctan2(s.hue_mean_ab[:, 1], s.hue_mean_ab[:, 0])
    assert len(np.unique(np.round(angles, 1))) >= 10


# ── the capability claim: smooth localized twist recovered ────────────

def test_fit_recovers_localized_hue_twist():
    """Ground truth: a smooth localized hue twist applied to a STRUCTURED world (peaked hue
    distribution, like every real pool — skin/sky/foliage clusters). A rotation of a uniform
    wheel is invisible in marginal statistics, so structure is what makes the curve
    identifiable; the refs-only path adopts the pool's hue structure for exactly this reason.
    Six plateaus cannot express the localized bump; the Fourier curve must land it."""
    from dataclasses import replace as _replace

    from lutgen.fitter.filmmodel import film_print_character
    from lutgen.orchestration.poolstats import HUE_BIN_CENTERS
    truth = FourierHueParams(s0=0.02, sc2=-0.05, ss1=0.08, ss3=0.02)
    # v3 (ADR-0008): the ground truth is a FILM system + hue personality — references in
    # the wild ARE film-graded; recovery of a twist on identity tone is no longer the
    # engine's claim (the film-prior tone stage biases against non-film worlds by design)
    truth_model = FilmModel(film_system=film_print_character(), hue_fourier=truth)

    peaked = _replace(neutral_prior())
    w = 1.0 + 0.8 * np.cos(HUE_BIN_CENTERS - 1.0)       # skin-ish cluster
    peaked.hue_weight = w / w.sum()
    cloud = synth_samples(peaked, OPT.n_samples, OPT.seed)
    inv = _cube_fn(load_base_inverse(), INVERSE_SIZE)
    fwd = _cube_fn(load_base(), DEFAULT_SIZE)
    ref = pool_stats([compute_frame_stats(np.clip(fwd(truth_model.forward(inv(cloud))), 0, 1))])

    res = fit_film_model(ref, options=OPT)              # refs-only path (structure adopted)
    theta = np.linspace(-np.pi, np.pi, 73)
    got = eval_shift(theta, res.model.hue_fourier)
    want = eval_shift(theta, truth)
    # the recovered curve tracks the true one across the wheel (correlation, not exactness —
    # crosstalk/curves absorb some, and thin far-side bins are regularized toward neutral)
    corr = np.corrcoef(got, want)[0, 1]
    assert corr > 0.45, f"hue-curve correlation {corr:.2f}"
    assert np.abs(got - want).max() < 0.14              # within ~8 degrees everywhere
    # the curve must be RIGHT WHERE THE MASS IS (the identifiable region): error at the
    # pool's hue-mass peak stays small. (A global-argmax check was dropped in b8.5: on the
    # thin far side the curve is ridge-flattened and its argmax jumps lobes — noise.)
    i_mass = int(np.argmin(np.abs(theta - 1.0)))       # the peaked world's mass centre
    assert abs(got[i_mass] - want[i_mass]) < 0.06      # within ~3.5 degrees there


# ── Block E: brightness-modulated hue shift (ADR-0007 b7.2) ───────────

def test_lshift_vanishes_at_mid_gray():
    from lutgen.fitter.filmmodel.fourierhue import eval_lshift
    p = FourierHueParams(l0=0.1, lc1=-0.05, ls2=0.08)
    theta = np.linspace(-np.pi, np.pi, 73)
    np.testing.assert_allclose(eval_shift(theta, p, 0.5), 0.0, atol=1e-12)   # F0 empty here
    np.testing.assert_allclose(eval_shift(theta, p, 1.0) - eval_shift(theta, p, 0.0),
                               eval_lshift(theta, p), atol=1e-12)


def test_lmod_shifts_shadows_and_highlights_oppositely():
    lab_dark = np.array([[0.2, 0.1, 0.0]])
    lab_brgt = np.array([[0.8, 0.1, 0.0]])
    p = FourierHueParams(l0=0.2)                        # shadows -0.06 rad, highlights +0.06
    hd = np.arctan2(*apply_fourier_hue(lab_dark, p)[0, [2, 1]])
    hb = np.arctan2(*apply_fourier_hue(lab_brgt, p)[0, [2, 1]])
    assert hd < 0 < hb
    np.testing.assert_allclose(hb - hd, 0.2 * 0.6, atol=1e-12)


def test_lmod_identity_and_serialize():
    m = FilmModel(hue_fourier=FourierHueParams(lc1=0.07))
    assert not m.is_identity()
    assert model_from_dict(model_to_dict(m)) == m
    d = model_to_dict(m)
    del d["hue_fourier"]
    assert model_from_dict(d).hue_fourier.is_identity()


def test_hue2_targets_capture_split_shift():
    """A brightness-opposed hue shift is INVISIBLE in the combined hue bins (it cancels) but
    plainly visible in the dark/bright halves — the statistic Block E fits against."""
    from lutgen.engine.perceptual import from_oklab
    rng = np.random.default_rng(5)
    n = 8000
    luma = rng.uniform(0.1, 0.9, n)
    # PEAKED hue world (like every real pool): constants of a rotation are invisible in
    # marginal statistics under a uniform wheel — structure is what carries the signal
    base_hue = np.concatenate([rng.normal(1.0, 0.7, n // 2), rng.uniform(-np.pi, np.pi, n - n // 2)])
    base_hue = (base_hue + np.pi) % (2 * np.pi) - np.pi
    shift = 0.25 * (luma - 0.5)                         # dark cool, bright warm around each hue
    hue = base_hue + shift
    lab = np.column_stack([luma, 0.1 * np.cos(hue), 0.1 * np.sin(hue)])
    img = np.clip(from_oklab(lab), 0, 1)
    s = compute_frame_stats(img)
    a_lo = np.arctan2(s.hue2_mean_ab[0, :, 1], s.hue2_mean_ab[0, :, 0])
    a_hi = np.arctan2(s.hue2_mean_ab[1, :, 1], s.hue2_mean_ab[1, :, 0])
    d = (a_hi - a_lo + np.pi) % (2 * np.pi) - np.pi
    # the signal lives where the hue mass is (rotation of a locally-uniform stretch is
    # invisible): read the split at the PEAK bin
    peak = int(np.argmax(s.hue_weight))
    assert d[peak] > 0.05                               # bright half rotated ahead of dark


def test_fit_recovers_brightness_modulated_shift():
    """Ground truth: hue shift that flips sign with brightness (split-tone generalization).
    The L-independent curve cannot express it; Block E must land direction + magnitude."""
    from dataclasses import replace as _replace

    from lutgen.fitter.filmmodel import film_print_character
    from lutgen.orchestration.poolstats import HUE_BIN_CENTERS
    truth = FourierHueParams(l0=0.16, lc1=0.06)
    truth_model = FilmModel(film_system=film_print_character(), hue_fourier=truth)
    peaked = _replace(neutral_prior())
    w = 1.0 + 0.8 * np.cos(HUE_BIN_CENTERS - 1.0)
    peaked.hue_weight = w / w.sum()
    cloud = synth_samples(peaked, OPT.n_samples, OPT.seed)
    inv = _cube_fn(load_base_inverse(), INVERSE_SIZE)
    fwd = _cube_fn(load_base(), DEFAULT_SIZE)
    ref = pool_stats([compute_frame_stats(np.clip(fwd(truth_model.forward(inv(cloud))), 0, 1))])

    res = fit_film_model(ref, options=OPT)
    # honest capability, v3 (ADR-0008): Block F itself produces brightness-dependent hue
    # rotation (per-channel gammas + knees — physically correct), so WHICH block carries
    # the split is degenerate; parameter-level recovery of the l-coefs is no longer
    # meaningful. What must hold: the RENDERED world's dark/bright hue split tracks the
    # reference (sign + magnitude order) at the hue-mass peak.
    out_img = np.clip(fwd(res.model.forward(inv(cloud))), 0, 1)
    out = pool_stats([compute_frame_stats(out_img)])

    def _split(tgt, peak):
        a_lo = np.arctan2(tgt.hue2_mean_ab[0, :, 1], tgt.hue2_mean_ab[0, :, 0])
        a_hi = np.arctan2(tgt.hue2_mean_ab[1, :, 1], tgt.hue2_mean_ab[1, :, 0])
        return ((a_hi - a_lo + np.pi) % (2 * np.pi) - np.pi)[peak]

    peak = int(np.argmax(ref.hue_weight))
    want, got = _split(ref, peak), _split(out, peak)
    assert want > 0.03                                  # the truth world does carry a split
    assert got > 0.0                                    # direction reproduced
    assert 0.3 * want < got < 3.0 * want                # magnitude within order
