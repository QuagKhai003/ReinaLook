"""Deterministic tests for app/preview.py (ADR-0007 b5.2)."""

from __future__ import annotations

import numpy as np

from lutgen.app.preview import before_after, make_test_still
from lutgen.engine.base import load_base


def test_make_test_still():
    s = make_test_still(128, 256)
    assert s.shape == (128, 256, 3)
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_before_after_shapes_and_range():
    still = make_test_still(64, 64)
    base = load_base()
    before, after = before_after(still, base, base)
    assert before.shape == after.shape == (64, 64, 3)
    for img in (before, after):
        assert img.min() >= 0.0 - 1e-9 and img.max() <= 1.0 + 1e-9
    # base==final → before and after identical
    np.testing.assert_allclose(before, after, atol=1e-9)


def test_after_reflects_warm_final():
    still = make_test_still(48, 48)
    base = load_base()
    warm_final = np.clip(base + np.array([0.1, 0.0, -0.08]), 0, 1)
    before, after = before_after(still, base, warm_final)
    assert (after[..., 0].mean() - after[..., 2].mean()) > (
        before[..., 0].mean() - before[..., 2].mean()
    )
