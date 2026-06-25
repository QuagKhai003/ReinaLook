"""Deterministic tests for fitter/rich.py — MKL optimal transport (ADR-0010)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lutgen.engine.base import load_base
from lutgen.engine.strength import blend
from lutgen.fitter.interface import LookFitter
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.consensus import build_consensus
from lutgen.orchestration.stats import compute_stats


def _warm_refs(n=5):
    out = []
    for seed in range(n):
        rng = np.random.default_rng(seed)
        content = rng.random((48, 48, 3)) * 0.5 + 0.25
        out.append(np.clip(content + np.array([0.15, 0.0, -0.1]), 0, 1))
    return out


def test_richfitter_satisfies_interface():
    assert isinstance(RichFitter(), LookFitter)


def test_rgb_mkl_matches_target_mean_and_covariance():
    rng = np.random.default_rng(0)
    source = rng.random((4000, 3))
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    out = RichFitter(space="rgb").fit(consensus, source_samples=source)(source)
    np.testing.assert_allclose(out.mean(axis=0), consensus.mean, atol=1e-2)
    np.testing.assert_allclose(np.cov(out, rowvar=False), consensus.covariance, atol=1e-2)


def test_oklab_mkl_matches_target_in_oklab():
    from lutgen.engine.perceptual import to_oklab

    rng = np.random.default_rng(2)
    source = rng.random((4000, 3))
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    out = RichFitter(space="oklab").fit(consensus, source_samples=source)(source)
    lab = to_oklab(out)
    np.testing.assert_allclose(lab.mean(axis=0), consensus.mean_oklab, atol=2e-2)
    np.testing.assert_allclose(np.cov(lab, rowvar=False), consensus.cov_oklab, atol=2e-2)


def test_identity_when_consensus_is_source():
    base = load_base()
    consensus = build_consensus([compute_stats(base)])
    for space in ("rgb", "oklab"):
        out = RichFitter(space=space, method="mkl").fit(consensus, source_samples=base)(base)
        np.testing.assert_allclose(out, base, atol=2e-3)  # target==source → ~identity


def test_tone_zero_preserves_luma_rgb():
    base = load_base()
    w = np.array([0.2126, 0.7152, 0.0722])
    consensus = build_consensus([compute_stats(r) for r in _warm_refs()])
    out = RichFitter(tone_strength=0.0, space="rgb", method="mkl").fit(consensus)(base)
    np.testing.assert_allclose(out @ w, base @ w, atol=1e-9)


def test_idt_transports_toward_target():
    from lutgen.fitter.rich import _idt

    rng = np.random.default_rng(0)
    source = rng.random((3000, 3))
    target = rng.random((3000, 3)) * 0.3 + np.array([0.5, 0.1, 0.0])  # shifted/narrowed
    out = _idt(source, target, iterations=20)
    # marginal means + stds move toward the target
    assert np.abs(out.mean(0) - target.mean(0)).sum() < np.abs(source.mean(0) - target.mean(0)).sum()
    assert np.abs(out.std(0) - target.std(0)).sum() < np.abs(source.std(0) - target.std(0)).sum()


def test_pdf_fitter_end_to_end(tmp_path):
    from lutgen.orchestration.pipeline import render_cube

    paths = []
    for i, rimg in enumerate(_warm_refs(3)):
        p = tmp_path / f"p{i}.png"
        Image.fromarray((rimg * 255).astype(np.uint8), "RGB").save(p)
        paths.append(p)
    base = load_base()
    cube = render_cube(paths, 1.0, fitter=RichFitter(method="pdf", iterations=8))
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    np.testing.assert_array_equal(
        render_cube(paths, 0.0, fitter=RichFitter(method="pdf", iterations=8)).samples, base)


def test_end_to_end_render(tmp_path):
    from lutgen.orchestration.pipeline import render_cube

    paths = []
    for i, r in enumerate(_warm_refs(3)):
        p = tmp_path / f"r{i}.png"
        Image.fromarray((r * 255).astype(np.uint8), "RGB").save(p)
        paths.append(p)
    base = load_base()
    cube = render_cube(paths, 1.0, fitter=RichFitter())
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0
    np.testing.assert_array_equal(render_cube(paths, 0.0, fitter=RichFitter()).samples, base)  # s0==base
