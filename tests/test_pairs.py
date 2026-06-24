"""Tests for fitter/pairs.py + render_cube_from_pairs (ADR-0012)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lutgen.engine.base import load_base
from lutgen.fitter.pairs import PairsFitter
from lutgen.orchestration.pipeline import render_cube_from_pairs

GAIN = np.array([1.10, 1.00, 0.90])
BIAS = np.array([0.05, 0.00, -0.03])


def _grade(rgb):
    return np.clip(rgb * GAIN + BIAS, 0.0, 1.0)


def _pair(seed, shape=(200, 200, 3)):
    before = np.random.default_rng(seed).random(shape)
    return before, _grade(before)


def test_recovers_known_grade():
    before, after = _pair(0)
    look = PairsFitter(smoothing=0.5).fit_from_pairs([before], [after])
    test = np.random.default_rng(9).random((2000, 3))
    out = look(test)
    err = np.abs(out - _grade(test))
    assert err.mean() < 0.02 and np.percentile(err, 95) < 0.06


def test_identity_grade():
    before = np.random.default_rng(1).random((150, 150, 3))
    look = PairsFitter(smoothing=0.5).fit_from_pairs([before], [before])  # after == before
    test = np.random.default_rng(2).random((1000, 3))
    np.testing.assert_allclose(look(test), test, atol=0.04)


def test_mismatched_pairs_raises():
    b, a = _pair(0)
    with pytest.raises(ValueError):
        PairsFitter().fit_from_pairs([b, b], [a])           # count mismatch
    with pytest.raises(ValueError):
        PairsFitter().fit_from_pairs([b], [a[:100]])        # shape mismatch


def _write(path, img):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").save(path)


def test_render_from_pairs_end_to_end(tmp_path):
    before, after = _pair(3, (120, 120, 3))
    bp, ap = tmp_path / "b.png", tmp_path / "a.png"
    _write(bp, before)
    _write(ap, after)
    base = load_base()
    cube = render_cube_from_pairs([bp], [ap], 1.0)
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    np.testing.assert_array_equal(render_cube_from_pairs([bp], [ap], 0.0).samples, base)  # s0==base
