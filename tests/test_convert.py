"""Deterministic tests for engine/convert.py (ADR-0001 batch 0.3)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.convert import convert_base
from lutgen.engine.grid import identity_grid
from lutgen.engine.spaces import di_decode, dwg_to_rec709_linear, rec709_g24_encode


def test_shape_preserved_on_full_grid():
    g = identity_grid()
    out = convert_base(g)
    assert out.shape == g.shape == (274625, 3)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_black_maps_to_black():
    np.testing.assert_allclose(convert_base(np.array([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0], atol=1e-9)


def test_neutral_axis_stays_neutral():
    # Greys (equal RGB in) must come out equal RGB (no color cast) — the base must not tint.
    for x in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        out = convert_base(np.array([x, x, x]))
        np.testing.assert_allclose(out, [out[0]] * 3, atol=1e-9)


def test_neutral_monotonic_increasing():
    codes = np.linspace(0.0, 1.0, 64)
    greys = np.stack([codes, codes, codes], axis=-1)
    lum = convert_base(greys)[:, 0]
    assert np.all(np.diff(lum) > 0)  # strictly increasing along the neutral axis


def test_matches_manual_pipeline():
    # convert_base == compose(spaces fns) exactly, for arbitrary samples.
    rng = np.random.default_rng(1)
    samples = rng.random((500, 3))
    expected = rec709_g24_encode(dwg_to_rec709_linear(di_decode(samples)))
    np.testing.assert_allclose(convert_base(samples), expected, atol=1e-12)


def test_super_white_unclamped():
    # DI code 1.0 -> scene-linear 100 -> g2.4 > 1 (clamping is downstream, not here).
    out = convert_base(np.array([1.0, 1.0, 1.0]))
    assert np.all(out > 1.0)


def test_rejects_bad_shape():
    import pytest

    with pytest.raises(ValueError):
        convert_base(np.zeros((4, 2)))
