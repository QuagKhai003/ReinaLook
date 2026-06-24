"""Test app/preview.py load_preview_still (ADR-0008)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lutgen.app.preview import load_preview_still


def test_load_preview_still(tmp_path):
    arr = (np.random.default_rng(0).random((300, 500, 3)) * 255).astype(np.uint8)
    p = tmp_path / "dwgdi.png"
    Image.fromarray(arr, "RGB").save(p)
    out = load_preview_still(p, max_dim=128)
    assert out.shape[-1] == 3
    assert max(out.shape[:2]) == 128
    assert out.min() >= 0.0 and out.max() <= 1.0
