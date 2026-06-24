"""Deterministic tests for engine/spaces.py (ADR-0001 batch 0.2)."""

from __future__ import annotations

import colour
import numpy as np

from lutgen.engine.spaces import (
    DWG_TO_REC709_MATRIX,
    GAMMA,
    di_decode,
    di_encode,
    dwg_to_rec709_linear,
    rec709_g24_decode,
    rec709_g24_encode,
)


def test_di_matches_colour_within_tol():
    code = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(
        di_decode(code), colour.models.oetf_inverse_DaVinciIntermediate(code), atol=1e-6
    )


def test_di_round_trip():
    lin = np.linspace(0.0, 10.0, 100)
    np.testing.assert_allclose(di_decode(di_encode(lin)), lin, atol=1e-9)


def test_di_anchors():
    assert di_decode(0.0) == 0.0  # black stays black


def test_matrix_shape_and_white_to_white():
    assert DWG_TO_REC709_MATRIX.shape == (3, 3)
    # D65 -> D65: linear white (1,1,1) maps to white (rows each sum to 1).
    np.testing.assert_allclose(
        dwg_to_rec709_linear(np.array([1.0, 1.0, 1.0])), [1.0, 1.0, 1.0], atol=1e-9
    )


def test_neutral_axis_preserved():
    # Equal-RGB linear (a grey) stays equal-RGB through the gamut matrix (same white point).
    for x in (0.0, 0.18, 0.5, 1.0):
        out = dwg_to_rec709_linear(np.array([x, x, x]))
        np.testing.assert_allclose(out, [x, x, x], atol=1e-9)


def test_matrix_batched():
    grid = np.random.default_rng(0).random((1000, 3))
    out = dwg_to_rec709_linear(grid)
    assert out.shape == (1000, 3)
    # matches per-row matmul
    np.testing.assert_allclose(out[0], DWG_TO_REC709_MATRIX @ grid[0], atol=1e-12)


def test_g24_round_trip():
    code = np.linspace(0.0, 1.0, 100)
    np.testing.assert_allclose(rec709_g24_decode(rec709_g24_encode(code)), code, atol=1e-12)


def test_g24_anchors_and_value():
    assert GAMMA == 2.4
    np.testing.assert_allclose(rec709_g24_encode(0.0), 0.0)
    np.testing.assert_allclose(rec709_g24_encode(1.0), 1.0)
    np.testing.assert_allclose(rec709_g24_encode(0.5), 0.5 ** (1.0 / 2.4), atol=1e-12)


def test_g24_encode_guards_negatives():
    # negative linear -> 0, no NaN
    out = rec709_g24_encode(np.array([-0.2, 0.0, 0.25]))
    assert not np.isnan(out).any()
    assert out[0] == 0.0
