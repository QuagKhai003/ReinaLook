"""Deterministic tests for fitter/mid.py (ADR-0005 b3.2/3.3)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import load_base
from lutgen.engine.cube_io import Cube
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.interface import LookFitter
from lutgen.fitter.mid import MidFitter
from lutgen.orchestration.consensus import build_consensus
from lutgen.orchestration.stats import compute_stats


def _mean_warmth(rgb):  # R - B
    return float(rgb[:, 0].mean() - rgb[:, 2].mean())


def _mean_sat(rgb):
    cmax = rgb.max(axis=1)
    cmin = rgb.min(axis=1)
    return float(((cmax - cmin) / np.maximum(cmax, 1e-6)).mean())


def _warm_refs(n=5):
    out = []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        content = rng.random((48, 48, 3)) * 0.6 + 0.2
        # warm + boosted saturation look
        looked = content + np.array([0.18, 0.0, -0.12])
        gray = looked.mean(axis=2, keepdims=True)
        looked = gray + (looked - gray) * 1.5  # raise chroma
        out.append(np.clip(looked, 0, 1))
    return out


def test_midfitter_satisfies_interface():
    assert isinstance(MidFitter(), LookFitter)


def test_identity_when_consensus_is_base():
    base = load_base()
    consensus = build_consensus([compute_stats(base)])
    look = MidFitter().fit(consensus)
    np.testing.assert_allclose(look(base), base, atol=1e-6)  # ~identity anchor


def test_warm_consensus_makes_base_warmer():
    base = load_base()
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    out = MidFitter().fit(consensus)(base)
    assert _mean_warmth(out) > _mean_warmth(base)   # warm cast imposed


def test_saturation_tracks_consensus():
    # Relative: a more-saturated consensus yields a more-saturated result than a flat one.
    base = load_base()
    rng = np.random.default_rng(3)
    content = rng.random((48, 48, 3)) * 0.5 + 0.25
    gray = content.mean(axis=2, keepdims=True)
    vivid = [np.clip(gray + (content - gray) * 2.2, 0, 1) for _ in range(4)]
    flat = [np.clip(gray + (content - gray) * 0.2, 0, 1) for _ in range(4)]
    out_hi = MidFitter().fit(build_consensus([compute_stats(r) for r in vivid]))(base)
    out_lo = MidFitter().fit(build_consensus([compute_stats(r) for r in flat]))(base)
    assert _mean_sat(out_hi) > _mean_sat(out_lo)


def test_tone_zero_preserves_luma_exactly():
    base = load_base()
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    out = MidFitter(tone_strength=0.0).fit(consensus)(base)
    w = np.array([0.2126, 0.7152, 0.0722])
    np.testing.assert_allclose(out @ w, base @ w, atol=1e-9)  # color cast only, exposure kept


def test_lower_tone_strength_darkens_less():
    base = load_base()
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    w = np.array([0.2126, 0.7152, 0.0722])
    base_l = (base @ w).mean()
    full = (MidFitter(tone_strength=1.0).fit(consensus)(base) @ w).mean()
    soft = (MidFitter(tone_strength=0.3).fit(consensus)(base) @ w).mean()
    assert abs(soft - base_l) < abs(full - base_l)  # softer tone stays closer to base exposure


def test_neutral_axis_stays_monotone():
    base = load_base()
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    look = MidFitter().fit(consensus)
    out = look(base)
    from lutgen.engine.grid import reshape_to_lattice

    diag = reshape_to_lattice(out)[np.arange(65), np.arange(65), np.arange(65)]
    luma = diag @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(luma) > -1e-6)  # non-decreasing brightness along greys


def test_end_to_end_render_valid_cube():
    base = load_base()
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    look_samples = MidFitter().fit(consensus)(base)
    final = regularize(blend(base, look_samples, 1.0))
    cube = Cube(size=65, samples=final)
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    assert _mean_warmth(cube.samples) > _mean_warmth(base)


def test_strength_zero_ignores_look():
    base = load_base()
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    look_samples = MidFitter().fit(consensus)(base)
    np.testing.assert_array_equal(blend(base, look_samples, 0.0), base)  # Golden Rule
