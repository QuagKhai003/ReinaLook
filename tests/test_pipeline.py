"""Deterministic tests for orchestration/pipeline.py (ADR-0006 b4.1)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lutgen.engine.base import load_base
from lutgen.orchestration.pipeline import render_cube


def _warm_png(path, seed):
    rng = np.random.default_rng(seed)
    content = rng.random((40, 40, 3)) * 0.6 + 0.2
    looked = np.clip(content + np.array([0.18, 0.0, -0.12]), 0, 1)
    Image.fromarray((looked * 255).astype(np.uint8), "RGB").save(path)


def _refs(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"ref{i}.png"
        _warm_png(p, i)
        paths.append(p)
    return paths


def test_strength_zero_is_base(tmp_path):
    cube = render_cube(_refs(tmp_path), 0.0)
    np.testing.assert_array_equal(cube.samples, load_base())  # Golden Rule end-to-end


def test_render_valid_and_warm(tmp_path):
    base = load_base()
    cube = render_cube(_refs(tmp_path), 1.0, title="warm")
    assert cube.size == 33 and cube.samples.shape == (35937, 3)
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    assert cube.title == "warm"
    warmth = lambda x: x[:, 0].mean() - x[:, 2].mean()
    assert warmth(cube.samples) > warmth(base)


def test_strength_scales_effect(tmp_path):
    refs = _refs(tmp_path)
    base = load_base()
    d = lambda s: np.abs(render_cube(refs, s).samples - base).mean()
    assert d(0.25) < d(0.5) < d(1.0)  # more strength = further from base
