"""Deterministic tests for orchestration/ingest.py (ADR-0004 b2.1)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lutgen.orchestration.ingest import load_image, load_references


def _write_png(path, rgb_uint8):
    Image.fromarray(rgb_uint8, mode="RGB").save(path)


def test_load_normalizes_to_unit_range(tmp_path):
    arr = np.zeros((10, 12, 3), dtype=np.uint8)
    arr[..., 0] = 255  # pure red
    p = tmp_path / "red.png"
    _write_png(p, arr)
    out = load_image(p)
    assert out.shape == (10, 12, 3)
    assert out.dtype == np.float64
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out[0, 0], [1.0, 0.0, 0.0])


def test_downscale_max_dim(tmp_path):
    arr = (np.random.default_rng(0).random((2000, 1000, 3)) * 255).astype(np.uint8)
    p = tmp_path / "big.png"
    _write_png(p, arr)
    out = load_image(p, max_dim=512)
    assert max(out.shape[:2]) == 512


def test_drops_alpha(tmp_path):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 1] = 200
    rgba[..., 3] = 128
    p = tmp_path / "a.png"
    Image.fromarray(rgba, mode="RGBA").save(p)
    out = load_image(p)
    assert out.shape == (8, 8, 3)


def test_load_references_empty_raises():
    with pytest.raises(ValueError):
        load_references([])
