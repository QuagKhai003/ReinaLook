"""Deterministic tests for orchestration/poolstats.py — pooled robust Learn-mode targets +
neutral prior (ADR-0001 b1.3).

The pooling tests prove the spec §4 claim in code: median pooling keeps what the pool shares
and rejects a single outlier frame's excursions."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.perceptual import to_oklab
from lutgen.orchestration.poolstats import (
    L_BAND_EDGES,
    N_BANDS,
    N_ZONES,
    QUANTILES,
    compute_frame_stats,
    neutral_prior,
    pool_stats,
)


def _synth_frame(seed: int, tint: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """A random 'scene' with an optional constant RGB tint (the 'grade')."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.05, 0.95, (64, 64, 3))
    return np.clip(img + np.asarray(tint), 0.0, 1.0)


# ── compute_frame_stats ───────────────────────────────────────────────

def test_frame_stats_shapes():
    s = compute_frame_stats(_synth_frame(0))
    assert s.channel_quantiles.shape == (3, len(QUANTILES))
    assert s.chroma_by_band.shape == (N_BANDS,)
    assert s.band_weight.shape == (N_BANDS,)
    assert s.zone_mean_ab.shape == (N_ZONES, 2)
    assert s.zone_weight.shape == (N_ZONES,)


def test_frame_stats_deterministic():
    a = compute_frame_stats(_synth_frame(1))
    b = compute_frame_stats(_synth_frame(1))
    np.testing.assert_array_equal(a.channel_quantiles, b.channel_quantiles)
    np.testing.assert_array_equal(a.zone_mean_ab, b.zone_mean_ab)


def test_frame_stats_band_weights_sum_to_one():
    s = compute_frame_stats(_synth_frame(2))
    np.testing.assert_allclose(s.band_weight.sum(), 1.0, atol=1e-12)


def test_frame_stats_gray_image_zero_chroma():
    gray = np.tile(np.linspace(0.1, 0.9, 256)[:, None], (1, 3)).reshape(16, 16, 3)
    s = compute_frame_stats(gray)
    np.testing.assert_allclose(s.chroma_by_band, 0.0, atol=1e-6)
    np.testing.assert_allclose(s.zone_weight, 0.0)          # all achromatic -> no zone pixels
    np.testing.assert_allclose(s.mean_lab[1:], 0.0, atol=1e-6)


def test_frame_stats_quantiles_monotonic():
    s = compute_frame_stats(_synth_frame(3))
    assert np.all(np.diff(s.channel_quantiles, axis=1) >= 0)


def test_frame_stats_warm_tint_shows_in_balance():
    neutral = compute_frame_stats(_synth_frame(4))
    warm = compute_frame_stats(_synth_frame(4, tint=(0.08, 0.0, -0.08)))
    assert warm.mean_lab[1] > neutral.mean_lab[1]  # a (red-green) pushed toward red


def test_frame_stats_empty_raises():
    with pytest.raises(ValueError):
        compute_frame_stats(np.empty((0, 3)))


def test_frame_stats_dark_frame_weights_shadow_bands():
    dark = _synth_frame(5) * 0.25   # everything pushed into low L
    s = compute_frame_stats(dark)
    lo_bands = s.band_weight[: len(L_BAND_EDGES) // 2 + 1]
    assert lo_bands.sum() > 0.9


# ── pool_stats: robust pooling ────────────────────────────────────────

def _pool(frames):
    return pool_stats([compute_frame_stats(f) for f in frames])


def test_pool_median_rejects_outlier_frame():
    # 4 frames share a teal tint; 1 outlier is aggressively magenta. Median must stay teal.
    teal = (-0.02, 0.02, 0.05)
    frames = [_synth_frame(i, tint=teal) for i in range(4)]
    outlier = _synth_frame(99, tint=(0.4, -0.3, 0.4))
    with_outlier = _pool(frames + [outlier])
    without = _pool(frames)
    # pooled colour balance barely moves when the outlier joins
    assert np.max(np.abs(with_outlier.mean_lab - without.mean_lab)) < 0.01


def test_pool_mean_would_be_hijacked_median_is_not():
    # sanity contrast: the MEAN of the same stats moves an order of magnitude more
    teal = (-0.02, 0.02, 0.05)
    stats = [compute_frame_stats(_synth_frame(i, tint=teal)) for i in range(4)]
    out_stats = compute_frame_stats(_synth_frame(99, tint=(0.4, -0.3, 0.4)))
    all_stats = stats + [out_stats]
    med = pool_stats(all_stats).mean_lab
    mean = np.mean(np.stack([s.mean_lab for s in all_stats]), axis=0)
    clean = pool_stats(stats).mean_lab
    assert np.linalg.norm(mean - clean) > 5 * np.linalg.norm(med - clean)


def test_pool_shared_grade_survives_different_scenes():
    # different random scenes, same warm grade -> pooled balance clearly warm
    warm = (0.06, 0.01, -0.05)
    pooled = _pool([_synth_frame(i, tint=warm) for i in range(6)])
    assert pooled.mean_lab[1] > 0.005   # a pushed red
    assert pooled.n_frames == 6


def test_pool_single_frame_passthrough():
    s = compute_frame_stats(_synth_frame(7))
    p = pool_stats([s])
    np.testing.assert_array_equal(p.channel_quantiles, s.channel_quantiles)
    np.testing.assert_array_equal(p.zone_mean_ab, s.zone_mean_ab)
    assert p.n_frames == 1


def test_pool_empty_raises():
    with pytest.raises(ValueError):
        pool_stats([])


# ── neutral prior ─────────────────────────────────────────────────────

def test_prior_is_gray_and_sane():
    p = neutral_prior()
    np.testing.assert_allclose(p.mean_lab[1:], 0.0)          # gray balance
    assert 0.0 < p.black_point < 0.05                        # blacks near black
    assert 0.9 < p.white_point <= 1.0
    assert np.all(np.diff(p.channel_quantiles, axis=1) > 0)  # monotonic tone ramp
    np.testing.assert_allclose(p.band_weight.sum(), 1.0)
    np.testing.assert_allclose(p.zone_weight.sum(), 1.0)
    assert p.n_frames == 0


def test_prior_is_constant_saturation_world():
    p = neutral_prior()
    np.testing.assert_allclose(p.sat_by_band, p.sat_by_band[0])   # flat C/L by design
    assert np.all(np.diff(p.chroma_by_band) > 0)                  # chroma grows with L


def test_prior_zone_means_point_at_zone_centres():
    from lutgen.fitter.filmmodel.huezone import ZONE_ANGLES
    p = neutral_prior()
    hues = np.arctan2(p.zone_mean_ab[:, 1], p.zone_mean_ab[:, 0])
    np.testing.assert_allclose(hues, ZONE_ANGLES, atol=1e-12)  # no hue twist in the prior


# ── zone binning consistency with Block D ─────────────────────────────

def test_zone_binning_matches_primary_colors():
    from lutgen.fitter.filmmodel.huezone import ZONE_NAMES
    # a pure-red image lands (almost) all its chromatic weight in the 'r' zone
    red = np.zeros((8, 8, 3))
    red[..., 0] = np.linspace(0.4, 0.9, 64).reshape(8, 8)
    s = compute_frame_stats(red)
    assert s.zone_weight[list(ZONE_NAMES).index("r")] > 0.99


def test_frame_stats_zone_mean_matches_oklab_direction():
    red = np.zeros((4, 4, 3))
    red[..., 0] = 0.8
    s = compute_frame_stats(red)
    from lutgen.fitter.filmmodel.huezone import ZONE_NAMES
    z = list(ZONE_NAMES).index("r")
    lab = to_oklab(np.array([[0.8, 0.0, 0.0]]))[0]
    np.testing.assert_allclose(s.zone_mean_ab[z], lab[1:], atol=1e-12)
