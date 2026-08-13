"""Deterministic tests for fitter/fit.py — the staged bounded fit (ADR-0001 b1.4).

Strategy: build reference targets FROM a known ground-truth FilmModel applied to the same
source cloud the fit uses, then check the fit (a) reproduces the reference statistics,
(b) returns near-identity when the reference IS the source world, (c) keeps thin-data zones
neutral (per-region regularization), (d) is deterministic, (e) respects bounds/monotonicity."""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
)
from lutgen.fitter.fit import FitOptions, _cube_fn, fit_film_model, synth_samples
from lutgen.orchestration.poolstats import (
    compute_frame_stats,
    neutral_prior,
    pool_stats,
)

# Small/fast options for tests — statistics stay over-determined even at this size.
# exposure_align OFF: these tests recover a KNOWN ground-truth model whose source world is
# the prior itself, so raw fitter mechanics are what's under test. The production path keeps
# alignment ON (ADR-0003 R.2 — absolute tone reshape is deliberately not recovered there).
OPT = FitOptions(n_samples=1500, max_nfev=40, exposure_align=False)

_PRIOR = neutral_prior()
_CLOUD = synth_samples(_PRIOR, OPT.n_samples, OPT.seed)
_INV = _cube_fn(load_base_inverse(), INVERSE_SIZE)
_FWD = _cube_fn(load_base(), DEFAULT_SIZE)


def _targets_from_model(model: FilmModel):
    """Reference targets = the known model's output world, measured like poolstats does."""
    out = _FWD(model.forward(_INV(_CLOUD)))
    return pool_stats([compute_frame_stats(np.clip(out, 0.0, 1.0))])


def test_identity_reference_fits_near_identity():
    # v3 (ADR-0008): the fit is anchored to the film-print character, so where the pool's
    # evidence is thick (mids) an identity world is recovered; where it is thin (the
    # extreme highlights of the prior world) the film shoulder is DESIGNED to remain —
    # weak evidence ships film, not identity.
    ref = _targets_from_model(FilmModel.identity())
    res = fit_film_model(ref, options=OPT)
    x = np.linspace(0.05, 0.75, 30)
    grid = np.column_stack([x, x, x])
    np.testing.assert_allclose(res.model.forward(grid), grid, atol=0.04)
    hi = np.column_stack([np.linspace(0.75, 0.98, 12)] * 3)
    out = res.model.forward(hi)
    assert np.all(np.diff(out[:, 1]) > 0)               # smooth monotone film shoulder


def test_fit_reproduces_known_model_statistics():
    truth = FilmModel(
        crosstalk=CrosstalkParams(gr=0.08, rb=0.04),
        curves=(SCurveParams(toe=0.4, slope=1.25), SCurveParams(slope=1.1),
                SCurveParams(shoulder=0.5, slope=0.9)),
        sat_luma=SatLumaParams(shadow=0.85, mid=1.15, high=1.0),   # shape-only (level not learned)
        hue_zones=HueZoneParams(r_shift=0.12, b_trim=0.2),
    )
    ref = _targets_from_model(truth)
    res = fit_film_model(ref, options=OPT)

    out = np.clip(_FWD(res.model.forward(_INV(_CLOUD))), 0.0, 1.0)
    s = compute_frame_stats(out)
    # the fitted model's world must match the reference world's statistics
    assert np.abs(s.channel_quantiles - ref.channel_quantiles).mean() < 0.035
    assert np.abs(s.mean_lab - ref.mean_lab).max() < 0.02
    assert np.abs(s.chroma_by_band - ref.chroma_by_band).mean() < 0.015
    # NOTE: no better-than-identity comparison remains here — the level-free residual design
    # (ADR-0007, the user's vividness contract) deliberately declines to chase mean-colour
    # and chroma LEVELS, which is most of what this particular ground truth's deviation is.
    # The absolute-accuracy clauses above are the meaningful contract.


def test_thin_zone_stays_neutral():
    # reference with the magenta zone erased (weight 0) -> its shift/trim must stay ~0
    truth = FilmModel(curves=(SCurveParams(slope=1.2), SCurveParams(), SCurveParams()),
                      hue_zones=HueZoneParams(r_shift=0.15))
    ref = _targets_from_model(truth)
    ref.zone_weight = ref.zone_weight.copy()
    from lutgen.fitter.filmmodel.huezone import ZONE_NAMES
    m = list(ZONE_NAMES).index("m")
    ref.zone_weight[m] = 0.0                       # no data in this zone
    res = fit_film_model(ref, options=OPT)
    assert abs(res.model.hue_zones.m_shift) < 0.03
    assert abs(res.model.hue_zones.m_trim) < 0.05


def test_fit_is_deterministic():
    ref = _targets_from_model(FilmModel(curves=(SCurveParams(toe=0.3), SCurveParams(),
                                                SCurveParams())))
    a = fit_film_model(ref, options=OPT)
    b = fit_film_model(ref, options=OPT)
    assert a.model == b.model
    assert a.stage_cost == b.stage_cost


def test_fitted_curves_respect_bounds_and_monotonic():
    truth = FilmModel(curves=(SCurveParams(toe=1.5, slope=1.9), SCurveParams(slope=0.6),
                              SCurveParams(shoulder=1.2)))
    res = fit_film_model(_targets_from_model(truth), options=OPT)
    for c in res.model.curves:
        assert 0.0 <= c.toe <= 2.0 and 0.0 <= c.shoulder <= 2.0
        assert 0.5 <= c.slope <= 2.0 and 0.3 <= c.pivot <= 0.7
    # fitted forward transform monotonic per channel on the gray axis
    x = np.linspace(0.0, 1.0, 257)
    out = res.model.forward(np.column_stack([x, x, x]))
    assert np.all(np.diff(out, axis=0) > -1e-9)


def test_progress_reports_stages():
    seen = []
    ref = _targets_from_model(FilmModel.identity())
    fit_film_model(ref, options=FitOptions(n_samples=800, max_nfev=5), progress=seen.append)
    assert seen == ["tone", "crosstalk", "coupling", "huesat", "polish", "huesat", "done"]  # hue re-fit after polish (b8.5)


def test_stage_diagnostics_present():
    ref = _targets_from_model(FilmModel.identity())
    res = fit_film_model(ref, options=FitOptions(n_samples=800, max_nfev=5))
    assert set(res.stage_cost) == {"tone", "crosstalk", "coupling", "huesat", "polish"}
    assert all(v >= 0 for v in res.stage_cost.values())
    assert res.n_frames == 1


def test_synth_samples_deterministic_and_in_range():
    a = synth_samples(_PRIOR, 500, seed=3)
    b = synth_samples(_PRIOR, 500, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (500, 3)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)
