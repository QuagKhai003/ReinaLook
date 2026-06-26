"""Tests for engine/filmprint.py — film-print (PFE) conversion base (ADR-0022)."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.cube_io import write_cube
from lutgen.engine.grid import identity_grid


def _identity_pfe(tmp_path):
    """Write an identity 17-point .cube to stand in for a real PFE LUT."""
    p = tmp_path / "identity_pfe.cube"
    write_cube(p, identity_grid(17), size=17, title="identity")
    return str(p)


def test_film_base_shape_and_range(tmp_path):
    from lutgen.engine.filmprint import build_film_base

    base = build_film_base(_identity_pfe(tmp_path), size=33)
    assert base.shape == (33 ** 3, 3)
    assert base.min() >= 0.0 and base.max() <= 1.0


def test_film_base_is_deterministic(tmp_path):
    from lutgen.engine.filmprint import build_film_base

    p = _identity_pfe(tmp_path)
    np.testing.assert_array_equal(build_film_base(p, 33), build_film_base(p, 33))


def test_exposure_shifts_brightness(tmp_path):
    from lutgen.engine.filmprint import build_film_base

    p = _identity_pfe(tmp_path)
    dark = build_film_base(p, 33, exposure=-1.0)
    bright = build_film_base(p, 33, exposure=1.0)
    assert bright.mean() > dark.mean()           # +1 stop is brighter than −1 stop


def test_real_2383_anchors_midgrey():
    """If the Juan Melara 2383 Standard LUT is present, 18% grey lands near 0.46 (film mid)."""
    import os

    import colour

    pfe = "LUT/JuanMelara/FilmUnlimited_2383_Standard.cube"
    if not os.path.exists(pfe):
        pytest.skip("2383 PFE LUT not present")
    from lutgen.engine.apply import apply_cube
    from lutgen.engine.filmprint import build_film_base

    base = build_film_base(pfe, 65)
    dwg = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
    di_mid = dwg.cctf_encoding(np.array([[[0.18, 0.18, 0.18]]]))   # DI value of 18% grey
    out = apply_cube(di_mid, base, 65).ravel()
    assert 0.42 < out.mean() < 0.50               # 2383 print mid grey
