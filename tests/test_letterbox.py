"""Deterministic tests for letterbox auto-crop at ingest (ADR-0003 R.1)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lutgen.orchestration.ingest import autocrop_letterbox, load_image


def _scene(h=120, w=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.15, 0.9, (h, w, 3))


def _with_bars(scene, top=0, bottom=0, left=0, right=0):
    h, w = scene.shape[:2]
    out = np.zeros((h + top + bottom, w + left + right, 3))
    out[top:top + h, left:left + w] = scene
    return out


def test_letterbox_bars_cropped():
    scene = _scene()
    barred = _with_bars(scene, top=14, bottom=14)
    out = autocrop_letterbox(barred)
    # bars gone (inset may trim a couple of scene rows too)
    assert out.shape[0] <= scene.shape[0]
    assert out.shape[0] >= scene.shape[0] - 4
    assert (out @ np.array([0.2126, 0.7152, 0.0722])).mean(axis=1).min() > 0.02


def test_pillarbox_bars_cropped():
    out = autocrop_letterbox(_with_bars(_scene(), left=20, right=8))
    assert out.shape[1] <= 200 and out.shape[1] >= 196


def test_no_bars_untouched():
    scene = _scene(seed=1)
    np.testing.assert_array_equal(autocrop_letterbox(scene), scene)


def test_dark_scene_not_cropped():
    # genuinely dark frame (night scene): low luma everywhere but ABOVE the bar threshold
    dark = np.full((100, 160, 3), 0.04)
    np.testing.assert_array_equal(autocrop_letterbox(dark), dark)


def test_near_black_frame_refuses_absurd_crop():
    # a frame that is almost entirely black must not be cropped to nothing
    black = np.zeros((100, 160, 3))
    black[45:55, :, :] = 0.5                      # thin bright band
    out = autocrop_letterbox(black)
    assert out.shape == black.shape               # crop would exceed 50% -> refused


def test_load_image_autocrops_by_default(tmp_path):
    barred = _with_bars(_scene(seed=2), top=16, bottom=16)
    p = tmp_path / "barred.png"
    Image.fromarray((barred * 255).astype(np.uint8)).save(p)
    out = load_image(p, max_dim=None)
    assert out.shape[0] <= 120                    # bars removed
    raw = load_image(p, max_dim=None, autocrop=False)
    assert raw.shape[0] == 152                    # opt-out keeps them


@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
def test_single_sided_bar(side):
    kw = {side: 18}
    scene = _scene(seed=3)
    out = autocrop_letterbox(_with_bars(scene, **kw))
    if side in ("top", "bottom"):
        assert out.shape[0] <= scene.shape[0] and out.shape[1] == scene.shape[1]
    else:
        assert out.shape[1] <= scene.shape[1] and out.shape[0] == scene.shape[0]
