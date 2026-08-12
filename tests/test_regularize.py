"""Deterministic tests for engine/regularize.py (ADR-0002 b1.2)."""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import load_base
from lutgen.engine.grid import flatten_lattice, identity_grid, reshape_to_lattice
from lutgen.engine.regularize import (
    clamp,
    enforce_neutral_monotonic,
    gamut_clamp,
    regularize,
)

_W = np.array([0.2126, 0.7152, 0.0722])


def test_gamut_clamp_in_range_identity():
    base = load_base()
    np.testing.assert_array_equal(gamut_clamp(base), base)   # byte-identical → s=0 stays exact


def test_gamut_clamp_preserves_luma_and_hue():
    c = np.array([1.4, 0.2, 0.1])            # out-of-range saturated red
    out = gamut_clamp(c)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out @ _W, c @ _W, atol=1e-6)  # luma preserved (hard clip would not)
    # desaturates (chroma shrinks) but keeps red dominant
    assert out[0] > out[1] > out[2]
    assert (out.max() - out.min()) < (c.max() - min(c[1], c[2]))


def test_clamp_range():
    x = np.array([[-0.5, 0.5, 1.5], [2.0, -1.0, 0.25]])
    out = clamp(x)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out[0], [0.0, 0.5, 1.0])


def test_neutral_monotonic_removes_inversion():
    size = 3
    samples = identity_grid(size)
    lat = reshape_to_lattice(samples, size)
    # break the grey diagonal: make the k=2 grey darker than k=1 (an inversion)
    lat[2, 2, 2] = [0.1, 0.1, 0.1]
    broken = flatten_lattice(lat)
    fixed = reshape_to_lattice(enforce_neutral_monotonic(broken, size), size)
    diag = fixed[np.arange(size), np.arange(size), np.arange(size)]
    assert np.all(np.diff(diag[:, 0]) >= 0.0)  # non-decreasing
    assert diag[2, 0] >= diag[1, 0]


def test_endpoints_preserved():
    base = load_base()
    out = regularize(base)
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0], atol=1e-6)    # black
    np.testing.assert_allclose(out[-1], base[-1], atol=1e-6)          # white node unchanged


def test_regularize_in_range():
    base = load_base()
    look = np.clip(base * 1.3 + 0.1, -0.2, 1.2)  # out-of-range "look"
    out = regularize(look)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_regularize_idempotent_on_clean_base():
    base = load_base()
    once = regularize(base)
    twice = regularize(once)
    np.testing.assert_array_equal(once, twice)


def test_clean_base_already_neutral_monotonic():
    # The verified base must already pass the neutral-monotonic guard unchanged.
    base = load_base()
    np.testing.assert_allclose(enforce_neutral_monotonic(base), base, atol=1e-12)
