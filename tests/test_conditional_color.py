"""Deterministic tests for conditional color learning (ADR-0006): per-band balance targets
captured by poolstats and reproduced by the fit; the global-darkening shortcut tamed."""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
from lutgen.engine.perceptual import from_oklab
from lutgen.fitter.filmmodel import FilmModel, SCurveParams
from lutgen.fitter.fit import FitOptions, _cube_fn, fit_film_model, synth_samples
from lutgen.orchestration.poolstats import (
    N_BANDS,
    compute_frame_stats,
    neutral_prior,
    pool_stats,
)

OPT = FitOptions(n_samples=1500, max_nfev=40, exposure_align=False)

_PRIOR = neutral_prior()
_INV = _cube_fn(load_base_inverse(), INVERSE_SIZE)
_FWD = _cube_fn(load_base(), DEFAULT_SIZE)


def _split_tone_image(n=4000, seed=0):
    """Synthetic frame with the director's split: cool shadows, warm highlights."""
    rng = np.random.default_rng(seed)
    luma = rng.uniform(0.02, 0.98, n)
    warmth = (luma - 0.5) * 0.12                        # a: cool below mid, warm above
    lab = np.column_stack([luma, warmth, warmth * 0.6])
    return np.clip(from_oklab(lab), 0, 1)


# ── poolstats captures the conditional signal ─────────────────────────

def test_band_mean_ab_shape_and_gray_zero():
    gray = np.tile(np.linspace(0.05, 0.95, 256)[:, None], (1, 3))
    s = compute_frame_stats(gray)
    assert s.band_mean_ab.shape == (N_BANDS, 2)
    np.testing.assert_allclose(s.band_mean_ab, 0.0, atol=1e-6)


def test_band_mean_ab_captures_split_tone():
    s = compute_frame_stats(_split_tone_image())
    assert s.band_mean_ab[0, 0] < -0.01                 # shadows cool (negative a)
    assert s.band_mean_ab[-1, 0] > 0.02                 # highlights warm (positive a)
    # (no monotonicity assert: the darkest band desaturates toward 0 at the gamut edge)
    # the GLOBAL mean hides it — that was the old failure
    assert abs(s.mean_lab[1]) < 0.02


def test_prior_band_balance_is_gray():
    np.testing.assert_array_equal(neutral_prior().band_mean_ab, np.zeros((N_BANDS, 2)))


# ── the fit reproduces conditional balance, not just the mean ─────────

def _targets_of(display_samples):
    return pool_stats([compute_frame_stats(np.clip(display_samples, 0, 1))])


def test_fit_reproduces_split_tone_statistics():
    # ground truth: warm-shadow model (blue toe) applied to the prior world
    truth = FilmModel(curves=(SCurveParams(slope=1.05), SCurveParams(),
                              SCurveParams(toe=0.5, slope=1.15)))
    cloud = synth_samples(_PRIOR, OPT.n_samples, OPT.seed)
    ref = _targets_of(_FWD(truth.forward(_INV(cloud))))
    spread_ref = ref.band_mean_ab[0] - ref.band_mean_ab[-1]
    assert np.abs(spread_ref).max() > 0.005             # ground truth IS conditional

    res = fit_film_model(ref, options=OPT)
    out = _targets_of(_FWD(res.model.forward(_INV(cloud))))
    # fitted world matches the reference PER BAND (the conditional behaviour), tightly
    # (0.035: the b8.3 tail/spread residuals trade a hair of band balance for punch)
    np.testing.assert_allclose(out.band_mean_ab, ref.band_mean_ab, atol=0.035)
    spread_out = out.band_mean_ab[0] - out.band_mean_ab[-1]
    assert np.dot(spread_out, spread_ref) > 0           # same direction of split


def test_exposure_ridge_prefers_conditional_explanation():
    # a look that darkens ONLY via the film curve (strong print contrast + black
    # convergence — no global shift in truth). v3 truth class (ADR-0008).
    from dataclasses import replace as _r

    from lutgen.fitter.filmmodel import film_print_character
    fs = film_print_character()
    fs = _r(fs, printer=_r(fs.printer, slope=1.7, ptoe=0.9))
    truth = FilmModel(film_system=fs)
    cloud = synth_samples(_PRIOR, OPT.n_samples, OPT.seed)
    ref = _targets_of(_FWD(truth.forward(_INV(cloud))))
    res = fit_film_model(ref, options=OPT)
    # the tamed exposure must not absorb what the curve explains
    assert abs(res.model.global_trim.exposure) < 0.08
    assert res.model.film_system.printer.slope > 1.4    # the curve carries the look
    assert res.model.film_system.printer.ptoe > 0.4


def test_pure_exposure_still_recoverable():
    # C must tame the shortcut, not amputate it: a REAL global shift is still learned
    from lutgen.fitter.filmmodel import GlobalParams
    truth = FilmModel(global_trim=GlobalParams(exposure=-0.15))
    cloud = synth_samples(_PRIOR, OPT.n_samples, OPT.seed)
    ref = _targets_of(_FWD(truth.forward(_INV(cloud))))
    res = fit_film_model(ref, options=OPT)
    assert res.model.global_trim.exposure < -0.08       # majority of the shift still found


# ── ADR-0007: a dark pool of vivid frames must not teach desaturation ─

def test_darkened_world_does_not_desaturate():
    """The user's exact complaint encoded: reference pool = the SAME vivid world, only
    darker (a dark film / dim web stills). The fit must learn the darkening WITHOUT
    learning a saturation cut — colourfulness targets are brightness-invariant."""
    from dataclasses import replace as _r2

    from lutgen.fitter.filmmodel import GlobalParams, film_print_character
    truth = FilmModel(film_system=film_print_character(),
                      global_trim=GlobalParams(exposure=-0.12))   # film + darkening ONLY
    cloud = synth_samples(_PRIOR, OPT.n_samples, OPT.seed)
    ref = _targets_of(_FWD(truth.forward(_INV(cloud))))
    # v3 honest contract (ADR-0008): the guard is the RENDERED OUTCOME, not parameter
    # forensics — Block F cannot null a level-shifted world exactly (physics bounds), so
    # hue-relative trims may wiggle chasing the residue; what must hold is that the
    # rendered world is NOT DULLER than the reference. Level protections stay structural
    # (t0 zeroed, satluma retired from the fit, mean-centered residuals).
    def _mean_sat(img):
        from lutgen.engine.perceptual import to_oklab
        lab = to_oklab(np.clip(img, 0, 1).reshape(-1, 3))
        return float(np.mean(np.hypot(lab[:, 1], lab[:, 2]) / np.maximum(lab[:, 0], 0.05)))

    ref_sat = _mean_sat(_FWD(truth.forward(_INV(cloud))))
    in_sat = _mean_sat(_FWD(FilmModel.identity().forward(_INV(cloud))))
    # C/L is level-dependent, so each path compares at ITS OWN level: the aligned (shipped)
    # path does not darken — footage must keep its own vividness (vs input); the bake path
    # darkens like the truth — rendered saturation must track the reference (vs ref).
    for opts, floor in ((_r2(OPT, exposure_align=True), in_sat), (OPT, ref_sat)):
        res = fit_film_model(ref, options=opts)
        m = res.model
        assert m.hue_fourier.t0 == 0.0                    # sat level structurally unlearnable
        assert m.sat_luma.is_identity()                   # satluma retired from the fit
        out_sat = _mean_sat(_FWD(m.forward(_INV(cloud))))
        assert out_sat > floor * 0.9                      # rendered: no dulling
    # the bake path still learns the darkening itself
    assert res.model.global_trim.exposure < -0.06


# ── ADR-0007: per-hue luminance ("lush vs olive greens") ─────────────

def test_hue_mean_l_captures_bright_greens():
    from lutgen.engine.perceptual import from_oklab
    from lutgen.orchestration.poolstats import HUE_BIN_CENTERS
    rng = np.random.default_rng(9)
    n = 6000
    hue = rng.uniform(-np.pi, np.pi, n)
    green = np.abs(((hue - 2.3) + np.pi) % (2 * np.pi) - np.pi) < 0.6
    luma = np.where(green, 0.7, 0.4) + rng.uniform(-0.05, 0.05, n)   # greens rendered bright
    lab = np.column_stack([luma, 0.1 * np.cos(hue), 0.1 * np.sin(hue)])
    s = compute_frame_stats(np.clip(from_oklab(lab), 0, 1))
    gbin = int(np.argmin(np.abs(HUE_BIN_CENTERS - 2.3)))
    other = [i for i in range(12) if abs(i - gbin) > 2]
    assert s.hue_mean_l[gbin] > s.hue_mean_l[other].mean() + 0.15


def test_fit_lifts_bright_hue_family():
    """Reference world renders greens brighter; curves/crosstalk (stages 1-2) must lift them
    — stage 3 cannot move luminance by construction."""
    from lutgen.fitter.filmmodel import CrosstalkParams
    truth = FilmModel(crosstalk=CrosstalkParams(gr=0.12, gb=0.12))   # green-brightening mix
    cloud = synth_samples(_PRIOR, OPT.n_samples, OPT.seed)
    ref = _targets_of(_FWD(truth.forward(_INV(cloud))))
    res = fit_film_model(ref, options=OPT)
    out = _targets_of(_FWD(res.model.forward(_INV(cloud))))
    rl_ref = ref.hue_mean_l / max(ref.mean_lab[0], 0.05)
    rl_out = out.hue_mean_l / max(out.mean_lab[0], 0.05)
    rl_base = _targets_of(_FWD(_INV(cloud))).hue_mean_l / 0.45
    # the fitted world's per-hue luminance tracks the reference far better than identity
    # 0.8: Block F's physics bounds sculpt per-hue brightness less freely than the
    # legacy 12-param display curves did — the tradeoff bought filmic structure (b8.4)
    assert np.abs(rl_out - rl_ref).mean() < 0.8 * np.abs(rl_base - rl_ref).mean()
