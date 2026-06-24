"""Deterministic tests for engine/cube_io.py (ADR-0001 batch 0.4)."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.engine.convert import convert_base
from lutgen.engine.cube_io import Cube, read_cube, write_cube
from lutgen.engine.grid import identity_grid, reshape_to_lattice


def test_write_read_round_trip(tmp_path):
    g = identity_grid()
    samples = convert_base(g)
    p = tmp_path / "base.cube"
    write_cube(p, samples, title="LookForge test")
    cube = read_cube(p)
    assert cube.size == 65
    assert cube.title == "LookForge test"
    assert cube.domain_min == (0.0, 0.0, 0.0)
    assert cube.domain_max == (1.0, 1.0, 1.0)
    np.testing.assert_allclose(cube.samples, samples, atol=1e-6)


def test_header_present(tmp_path):
    p = tmp_path / "h.cube"
    write_cube(p, identity_grid(), title="X")
    text = p.read_text(encoding="utf-8")
    assert 'TITLE "X"' in text
    assert "LUT_3D_SIZE 65" in text
    assert text.splitlines()[1].startswith("LUT_3D_SIZE")  # after TITLE


def test_ordering_ramp_red_fastest(tmp_path):
    # Write a tiny identity cube; the file's line order must be red-fastest (Resolve) so that
    # reshaping the parsed samples to a [blue,green,red] lattice recovers each coordinate.
    size = 3
    p = tmp_path / "ramp.cube"
    write_cube(p, identity_grid(size), size=size)
    parsed = read_cube(p).samples
    lat = reshape_to_lattice(parsed, size)
    step = 1.0 / (size - 1)
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                np.testing.assert_allclose(lat[bi, gi, ri], [ri * step, gi * step, bi * step], atol=1e-6)
    # And explicitly: first `size` data lines vary red only.
    data = [ln for ln in p.read_text().splitlines() if ln[0].isdigit()]
    assert data[0].split() == ["0.000000", "0.000000", "0.000000"]
    assert data[1].split() == ["0.500000", "0.000000", "0.000000"]  # red moved first


def test_clamp_super_white(tmp_path):
    samples = convert_base(identity_grid())
    assert samples.max() > 1.0  # raw stage-A has super-white
    p = tmp_path / "clamped.cube"
    write_cube(p, samples, clamp=True)
    out = read_cube(p).samples
    assert out.max() <= 1.0 + 1e-9 and out.min() >= 0.0


def test_count_mismatch_raises(tmp_path):
    p = tmp_path / "bad.cube"
    p.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")  # needs 8 lines
    with pytest.raises(ValueError):
        read_cube(p)


def test_cube_shape_validation():
    with pytest.raises(ValueError):
        Cube(size=2, samples=np.zeros((4, 3)))  # 2**3 == 8 expected
