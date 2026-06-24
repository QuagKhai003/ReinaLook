"""Deterministic tests for orchestration/stats.py (ADR-0004 b2.2)."""

from __future__ import annotations

import numpy as np

from lutgen.orchestration.stats import QUANTILES, compute_stats


def test_flat_gray():
    img = np.full((16, 16, 3), 0.5)
    s = compute_stats(img)
    np.testing.assert_allclose(s.channel_quantiles, 0.5)
    assert s.channel_quantiles.shape == (3, len(QUANTILES))
    assert s.saturation_global == 0.0
    np.testing.assert_allclose(s.band_balance, 0.5)  # empty bands fall back to global mean
    assert s.black_point == 0.5 and s.white_point == 0.5


def test_pure_red_is_saturated():
    img = np.zeros((8, 8, 3))
    img[..., 0] = 1.0
    s = compute_stats(img)
    np.testing.assert_allclose(s.saturation_global, 1.0)


def test_warm_image_balance():
    rng = np.random.default_rng(0)
    img = np.clip(rng.random((64, 64, 3)) * 0.5 + np.array([0.2, 0.0, -0.1]), 0, 1)
    s = compute_stats(img)
    # red mean should exceed blue mean across every luma band
    assert np.all(s.band_balance[:, 0] > s.band_balance[:, 2])


def test_endpoints_from_luma():
    img = np.linspace(0, 1, 100).reshape(10, 10)[..., None].repeat(3, axis=2)
    s = compute_stats(img)
    assert 0.0 <= s.black_point < s.white_point <= 1.0


def test_accepts_flat_pixel_list():
    pixels = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    s = compute_stats(pixels)
    assert s.channel_quantiles.shape == (3, len(QUANTILES))
