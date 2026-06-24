"""Tests for engine/perceptual.py — Oklab conversion (ADR-0011)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.perceptual import from_oklab, to_oklab


def test_round_trip():
    rng = np.random.default_rng(0)
    rgb = rng.random((1000, 3))
    np.testing.assert_allclose(from_oklab(to_oklab(rgb)), rgb, atol=1e-6)


def test_black_and_white():
    np.testing.assert_allclose(to_oklab(np.zeros(3))[0], 0.0, atol=1e-9)      # L=0 at black
    np.testing.assert_allclose(to_oklab(np.ones(3)), [1.0, 0.0, 0.0], atol=1e-6)  # white: L≈1, neutral


def test_neutral_is_achromatic():
    # greys must have ~zero a, b
    for v in (0.2, 0.5, 0.8):
        lab = to_oklab(np.array([v, v, v]))
        np.testing.assert_allclose(lab[1:], [0.0, 0.0], atol=1e-6)


def test_shape_preserved():
    img = np.random.default_rng(1).random((5, 7, 3))
    assert to_oklab(img).shape == (5, 7, 3)
