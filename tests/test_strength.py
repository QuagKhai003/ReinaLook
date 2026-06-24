"""Deterministic tests for engine/strength.py (ADR-0002 b1.1). The s=0 parity test is the
core Golden-Rule guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.base import load_base
from lutgen.engine.strength import blend


def _dummy_look(base: np.ndarray) -> np.ndarray:
    # An arbitrary but valid "looked" cube: warm shift + contrast, clipped to [0,1].
    rng = np.random.default_rng(7)
    return np.clip(base * 0.9 + 0.05 + rng.normal(0, 0.02, base.shape), 0.0, 1.0)


def test_s0_is_base_bit_for_bit():
    base = load_base()
    look = _dummy_look(base)
    np.testing.assert_array_equal(blend(base, look, 0.0), base)  # exact, the Golden Rule


def test_s1_is_look_bit_for_bit():
    base = load_base()
    look = _dummy_look(base)
    np.testing.assert_array_equal(blend(base, look, 1.0), look)


def test_half_is_midpoint():
    base = load_base()
    look = _dummy_look(base)
    np.testing.assert_allclose(blend(base, look, 0.5), 0.5 * (base + look), atol=1e-12)


def test_strength_clamped():
    base = load_base()
    look = _dummy_look(base)
    np.testing.assert_array_equal(blend(base, look, -3.0), base)  # < 0 -> 0 -> base
    np.testing.assert_array_equal(blend(base, look, 9.0), look)   # > 1 -> 1 -> look


def test_s0_returns_independent_copy():
    base = load_base()
    out = blend(base, _dummy_look(base), 0.0)
    out[0] = [0.5, 0.5, 0.5]
    np.testing.assert_allclose(load_base()[0], [0.0, 0.0, 0.0], atol=1e-6)


def test_monotonic_in_strength_per_node():
    base = load_base()
    look = _dummy_look(base)
    prev = base.copy()
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        cur = blend(base, look, s)
        # each node moves monotonically from base toward look as s grows
        assert np.all((cur - prev) * (look - base) >= -1e-12)
        prev = cur


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        blend(np.zeros((8, 3)), np.zeros((27, 3)), 0.5)
