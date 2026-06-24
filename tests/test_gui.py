"""Offscreen smoke test for the GUI shell (ADR-0007 b5.3). Skips if PySide6 is absent."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from lutgen.engine.cube_io import read_cube  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _drain(app, w):
    """Let any background compute thread finish (offscreen)."""
    for _ in range(100):
        app.processEvents()
        if w._thread is None or not w._thread.isRunning():
            break
        w._thread.wait(50)
    app.processEvents()


def test_window_builds_and_previews(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    assert w.windowTitle() == "LookForge"
    np.testing.assert_array_equal(w._final_samples(), w._base)  # no refs → base
    w._on_strength(50)   # must not raise (debounced)
    _drain(app, w)
    w.close()


def test_mode_and_fitter_controls(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    w._fitter.setCurrentText("Mid"); _drain(app, w)
    w._fitter.setCurrentText("Rich"); w._method.setCurrentText("pdf"); _drain(app, w)
    w._space.setCurrentText("rgb"); _drain(app, w)
    np.testing.assert_array_equal(w._final_samples(), w._base)   # no refs → base
    w._mode.setCurrentIndex(1); _drain(app, w)                   # Neutral+Graded pools
    assert w._is_pairs() and w._fitter.isEnabled()
    assert w._final_samples().shape == (274625, 3)
    w.close()


def test_threaded_compute_builds_look(app, tmp_path):
    from PIL import Image

    from lutgen.app.main_window import MainWindow

    paths = []
    for i in range(2):
        rng = np.random.default_rng(i)
        img = np.clip(rng.random((32, 32, 3)) * 0.6 + np.array([0.18, 0.0, -0.12]), 0, 1)
        p = tmp_path / f"r{i}.png"
        Image.fromarray((img * 255).astype(np.uint8), "RGB").save(p)
        paths.append(str(p))
    w = MainWindow()
    w._refs = paths
    w._refs_list.addItems(paths)
    w._launch_compute()           # Compute button → off-thread build + spinner; grays controls
    assert not w._fitter.isEnabled()          # controls disabled during compute
    _drain(app, w)
    assert w._fitter.isEnabled()              # re-enabled when done
    assert w._look_samples is not None and w._look_samples.shape == (274625, 3)
    w.close()


def test_placement_switch_refreshes_preview_without_dirty(app, tmp_path):
    from PIL import Image

    from lutgen.app.main_window import MainWindow

    paths = []
    for i in range(2):
        rng = np.random.default_rng(10 + i)
        img = np.clip(rng.random((24, 24, 3)) * 0.6 + 0.1, 0, 1)
        p = tmp_path / f"r{i}.png"
        Image.fromarray((img * 255).astype(np.uint8), "RGB").save(p)
        paths.append(str(p))
    w = MainWindow()
    w._refs = paths
    w._refs_list.addItems(paths)
    w._launch_compute(); _drain(app, w)               # compute the look
    assert w._look_samples is not None and not w._dirty
    look_before = w._look_samples
    w._placement.setCurrentIndex(1)                   # switch to "between" → preview-only refresh
    _drain(app, w)
    assert not w._dirty                               # placement is preview-only, look stays current
    assert w._look_samples is look_before             # not rebuilt
    w.close()


def test_export_writes_valid_cube(app, tmp_path):
    from lutgen.app.main_window import MainWindow
    from lutgen.engine.cube_io import write_cube

    w = MainWindow()
    out = tmp_path / "gui.cube"
    write_cube(out, w._final_samples(), title="LookForge")  # same path the Export button uses
    assert read_cube(out).size == 65
