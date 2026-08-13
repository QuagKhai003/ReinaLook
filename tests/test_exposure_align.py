"""Deterministic tests for exposure-aligned tone targets (ADR-0003 R.2): a dark-scene pool
must fit with curves OFF their bounds and a far lower tone cost than the unaligned fit."""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
from lutgen.fitter.filmmodel import FilmModel
from lutgen.fitter.fit import (
    FitOptions,
    _cube_fn,
    _exposure_aligned,
    fit_film_model,
    synth_samples,
)
from lutgen.orchestration.poolstats import (
    compute_frame_stats,
    neutral_prior,
    pool_stats,
)

# exposure_align is an explicit knob since ADR-0004 (default OFF: Block G's exposure param
# makes the level expressible, so the full look is learned; ON = colour-science-only recipe)
OPT = FitOptions(n_samples=1200, max_nfev=25, exposure_align=True)
OPT_NOALIGN = FitOptions(n_samples=1200, max_nfev=25, exposure_align=False)

_PRIOR = neutral_prior()
_INV = _cube_fn(load_base_inverse(), INVERSE_SIZE)
_FWD = _cube_fn(load_base(), DEFAULT_SIZE)


def _dark_world_targets(scale=0.3):
    """A 'dark film' reference pool: the prior world uniformly darkened — same look (identity
    grade), just dark scenes."""
    cloud = synth_samples(_PRIOR, OPT.n_samples, OPT.seed) * scale
    out = np.clip(_FWD(FilmModel.identity().forward(_INV(cloud))), 0, 1)
    return pool_stats([compute_frame_stats(out)])


def test_aligned_source_adopts_pool_tone_distribution():
    ref = _dark_world_targets()
    aligned = _exposure_aligned(_PRIOR, ref)
    luma_q = ref.channel_quantiles.mean(axis=0)
    for c in range(3):                             # neutral world with the pool's own tones
        np.testing.assert_allclose(aligned.channel_quantiles[c], luma_q, atol=1e-12)
    # chroma/zone stats untouched
    np.testing.assert_array_equal(aligned.chroma_by_band, _PRIOR.chroma_by_band)


def test_dark_pool_fits_off_bounds_with_alignment():
    ref = _dark_world_targets()
    res = fit_film_model(ref, options=OPT)
    for c in res.model.curves:                     # no parameter pinned at its bound
        assert c.slope < 1.99 and c.pivot < 0.699
        assert c.toe < 1.95
    res_no = fit_film_model(ref, options=OPT_NOALIGN)
    assert res.stage_cost["tone"] < 0.5 * res_no.stage_cost["tone"]


def test_bright_pool_aligns_upward():
    cloud = np.clip(synth_samples(_PRIOR, OPT.n_samples, OPT.seed) * 1.6, 0, 1)
    out = np.clip(_FWD(FilmModel.identity().forward(_INV(cloud))), 0, 1)
    ref = pool_stats([compute_frame_stats(out)])
    aligned = _exposure_aligned(_PRIOR, ref)
    mid = aligned.channel_quantiles.shape[1] // 2
    assert aligned.channel_quantiles[:, mid].mean() > _PRIOR.channel_quantiles[:, mid].mean()


def test_explicit_source_pool_never_rescaled():
    ref = _dark_world_targets()
    src = neutral_prior()
    res = fit_film_model(ref, source=src, options=OPT)   # user-measured source: no alignment
    # with an unshifted source and a dark ref, tone params must work hard (proves no rescale)
    assert res.stage_cost["tone"] > fit_film_model(ref, options=OPT).stage_cost["tone"]
