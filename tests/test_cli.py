"""Deterministic tests for cli.py (ADR-0006 b4.3)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from lutgen.cli import main
from lutgen.engine.base import load_base
from lutgen.engine.cube_io import read_cube
from lutgen.orchestration.preset import load_preset


def _png(path, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.random((32, 32, 3)) * 0.6 + np.array([0.15, 0.0, -0.1]), 0, 1)
    Image.fromarray((img * 255).astype(np.uint8), "RGB").save(path)


def _two_refs(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _png(a, 1)
    _png(b, 2)
    return str(a), str(b)


def test_render_writes_valid_cube(tmp_path):
    a, b = _two_refs(tmp_path)
    out = tmp_path / "look.cube"
    rc = main(["render", "--refs", a, b, "--strength", "1.0", "--out", str(out), "--title", "T"])
    assert rc == 0
    cube = read_cube(out)
    assert cube.size == 33 and cube.title == "T"
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0


def test_render_strength_zero_is_base(tmp_path):
    a, b = _two_refs(tmp_path)
    out = tmp_path / "base.cube"
    assert main(["render", "--refs", a, b, "--strength", "0", "--out", str(out)]) == 0
    np.testing.assert_allclose(read_cube(out).samples, load_base(), atol=1e-6)


def test_save_then_use_preset(tmp_path):
    a, b = _two_refs(tmp_path)
    preset = tmp_path / "p.json"
    out1 = tmp_path / "o1.cube"
    main(["render", "--refs", a, b, "--strength", "0.7", "--out", str(out1),
          "--save-preset", str(preset)])
    saved = load_preset(preset)
    assert saved["refs"] == [a, b] and saved["strength"] == 0.7

    out2 = tmp_path / "o2.cube"
    assert main(["render", "--preset", str(preset), "--out", str(out2)]) == 0
    np.testing.assert_array_equal(read_cube(out1).samples, read_cube(out2).samples)


def test_render_look_writes_valid_cube(tmp_path):
    a, b = _two_refs(tmp_path)
    out = tmp_path / "look.cube"
    rc = main(["render-look", "--refs", a, b, "--strength", "1.0", "--tone", "0", "--out", str(out)])
    assert rc == 0
    cube = read_cube(out)
    assert cube.size == 33
    assert cube.samples.min() >= 0.0 and cube.samples.max() <= 1.0


def test_render_look_strength_zero_is_identity(tmp_path):
    from lutgen.engine.grid import identity_grid

    a, b = _two_refs(tmp_path)
    out = tmp_path / "id.cube"
    assert main(["render-look", "--refs", a, b, "--strength", "0", "--out", str(out)]) == 0
    np.testing.assert_allclose(read_cube(out).samples, identity_grid(), atol=1e-6)


def test_no_refs_errors(tmp_path):
    assert main(["render", "--out", str(tmp_path / "x.cube")]) == 2


def test_missing_out_exits(tmp_path):
    a, _ = _two_refs(tmp_path)
    with pytest.raises(SystemExit):  # argparse: --out required
        main(["render", "--refs", a])
