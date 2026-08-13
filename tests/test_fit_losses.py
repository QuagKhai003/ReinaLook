"""b8.3 tests — fit-v2 losses: sat-distribution statistics, tail weighting, Hunt term,
asymmetric under-sat, JPEG tile debias, soft-l1 pass-through."""

from __future__ import annotations

import numpy as np

from lutgen.fitter.fit import FitOptions, _residuals, fit_film_model
from lutgen.orchestration.poolstats import (
    SAT_QUANTILES,
    compute_frame_stats,
    neutral_prior,
    pool_stats,
)


def _cloud(seed=0, n=4000, sat_scale=1.0, level=0.45):
    """A display-space colour cloud with controllable saturation spread."""
    rng = np.random.default_rng(seed)
    base = np.clip(level + rng.normal(0, 0.12, (n, 1)), 0.02, 0.98)
    tint = rng.normal(0, 0.08 * sat_scale, (n, 3))
    return np.clip(base + tint, 0.0, 1.0)


def _targets(img):
    return pool_stats([compute_frame_stats(img)])


# ── statistics ────────────────────────────────────────────────────────

def test_sat_quantiles_monotone_and_scale():
    s1 = compute_frame_stats(_cloud(sat_scale=1.0))
    s2 = compute_frame_stats(_cloud(sat_scale=2.0))
    assert np.all(np.diff(s1.sat_quantiles) >= 0)
    assert s2.sat_quantiles[-1] > s1.sat_quantiles[-1]      # more spread => fatter tail
    assert len(s1.sat_quantiles) == len(SAT_QUANTILES)


def test_tile_p95_resists_dull_dilution():
    """A saturated subject on a big dull wall: frame-level p95 dives when the wall grows;
    the tile estimate keeps reading the subject's tail."""
    rng = np.random.default_rng(3)
    h = w = 64
    img = np.full((h, w, 3), 0.5) + rng.normal(0, 0.005, (h, w, 3))   # dull wall
    img[:16, :16] += np.array([0.25, -0.1, -0.1])                     # saturated subject
    img = np.clip(img, 0, 1)
    s = compute_frame_stats(img)
    assert s.sat_tile_p95 > s.sat_quantiles[-1] * 0.9                 # tail survives tiling
    flat = compute_frame_stats(img.reshape(-1, 3))
    assert flat.sat_tile_p95 == flat.sat_quantiles[-1]                # flat input: global p95


def test_prior_has_sat_distribution_fields():
    p = neutral_prior()
    assert p.sat_quantiles.shape == SAT_QUANTILES.shape
    assert np.all(np.diff(p.sat_quantiles) > 0) and p.sat_tile_p95 > 0


# ── residual behaviour ────────────────────────────────────────────────

def _chroma_res(out, ref, opt):
    return _residuals(out, ref, opt, tone=False, balance=False, chroma=True, zones=False,
                      spread=True)


def test_spread_residual_fires_on_lost_punch():
    """Output with compressed saturation tails vs a punchy reference => nonzero residual,
    larger than the mirrored (over-punchy) case (asymmetry)."""
    opt = FitOptions()
    ref = _targets(_cloud(seed=1, sat_scale=1.6))
    dull = _chroma_res(_cloud(seed=2, sat_scale=0.8), ref, opt)
    assert float(np.abs(dull).sum()) > 0.05
    ref_dull = _targets(_cloud(seed=1, sat_scale=0.8))
    punchy = _chroma_res(_cloud(seed=2, sat_scale=1.6), ref_dull, opt)
    # same magnitude of mismatch, opposite direction: under-sat must cost more
    assert float(np.square(dull).sum()) > float(np.square(punchy).sum())


def test_hunt_term_discounts_brighter_render():
    """Same saturation statistics at a brighter level should read as MORE colourful =>
    with Hunt on, a brighter output needs less measured sat => residual shifts down."""
    opt_on = FitOptions(hunt_alpha=0.15)
    opt_off = FitOptions(hunt_alpha=0.0)
    ref = _targets(_cloud(seed=4, sat_scale=1.3, level=0.4))
    bright = _cloud(seed=5, sat_scale=1.3, level=0.6)
    r_on = _chroma_res(bright, ref, opt_on)
    r_off = _chroma_res(bright, ref, opt_off)
    assert not np.allclose(r_on, r_off)                   # the term does something


def test_tail_weighting_boosts_extreme_quantiles():
    opt = FitOptions(tail_weight=3.0)
    ref = _targets(_cloud(seed=6))
    out = _cloud(seed=7, level=0.55)
    r_flat = _residuals(out, ref, FitOptions(tail_weight=1.0),
                        tone=True, balance=False, chroma=False, zones=False)
    r_tail = _residuals(out, ref, opt,
                        tone=True, balance=False, chroma=False, zones=False)
    q = r_flat[:63].reshape(3, 21)
    qt = r_tail[:63].reshape(3, 21)
    np.testing.assert_allclose(qt[:, 10], q[:, 10], atol=1e-12)   # median untouched
    assert np.abs(qt[:, 0]).sum() > np.abs(q[:, 0]).sum() * 2.5   # ends boosted ~3x


# ── fit smoke with the new losses ─────────────────────────────────────

def test_fit_runs_with_v2_losses():
    ref = _targets(_cloud(seed=8, sat_scale=1.5, level=0.5))
    res = fit_film_model(ref, options=FitOptions(n_samples=400, max_nfev=8))
    assert set(res.stage_cost) == {"tone", "crosstalk", "coupling", "huesat", "polish"}
    assert all(np.isfinite(v) for v in res.stage_cost.values())
    out = res.model.forward(np.random.default_rng(0).uniform(0, 1, (32, 3)))
    assert np.all(np.isfinite(out))


def test_soft_l1_option_passes_through():
    ref = _targets(_cloud(seed=9))
    r1 = fit_film_model(ref, options=FitOptions(n_samples=300, max_nfev=4, loss="soft_l1"))
    r2 = fit_film_model(ref, options=FitOptions(n_samples=300, max_nfev=4, loss="linear"))
    assert all(v >= 0 for v in r1.stage_cost.values())
    assert r1.stage_cost != r2.stage_cost                 # the loss changes the objective
