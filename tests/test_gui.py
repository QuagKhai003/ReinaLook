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
    for _ in range(600):                 # up to ~30s — pdf (default) is slower than mkl
        app.processEvents()
        if w._thread is None or not w._thread.isRunning():
            break
        w._thread.wait(50)
    app.processEvents()


def test_window_builds_and_previews(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    assert w.windowTitle().startswith("ReinaLook")   # "ReinaLook — <mode>" since b2.4
    np.testing.assert_array_equal(w._final_samples(), w._base)  # no refs → base
    w._on_strength(50)   # must not raise (debounced)
    _drain(app, w)
    w.close()


def _neutral_graded(tmp_path, seed=0):
    """Write 2 neutral + 2 graded frames; return (neutral_paths, graded_paths)."""
    from PIL import Image

    def write(prefix, shift):
        out = []
        for i in range(2):
            rng = np.random.default_rng(seed + i)
            img = np.clip(rng.random((28, 28, 3)) * 0.6 + shift, 0, 1)
            p = tmp_path / f"{prefix}{i}.png"
            Image.fromarray((img * 255).astype(np.uint8), "RGB").save(p)
            out.append(str(p))
        return out
    return write("n", 0.2), write("g", np.array([0.18, 0.0, -0.12]))


def test_placement_controls(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    w._placement.setCurrentIndex(1); _drain(app, w)             # Between CSTs
    w._placement.setCurrentIndex(0); _drain(app, w)             # Replace CSTout
    np.testing.assert_array_equal(w._final_samples(), w._base)   # no inputs → base
    assert w._final_samples().shape == (274625, 3)
    w.close()


def test_threaded_compute_builds_look(app, tmp_path):
    from lutgen.app.main_window import MainWindow

    neutral, graded = _neutral_graded(tmp_path)
    w = MainWindow()
    w._before = neutral; w._before_list.addItems(neutral)
    w._after = graded; w._after_list.addItems(graded)
    w._launch_compute()           # Compute button → off-thread build + spinner; grays controls
    assert not w._placement.isEnabled()       # controls disabled during compute
    _drain(app, w)
    assert w._placement.isEnabled()           # re-enabled when done
    assert w._look_samples is not None and w._look_samples.shape == (274625, 3)
    w.close()


def test_placement_switch_refreshes_preview_without_dirty(app, tmp_path):
    from lutgen.app.main_window import MainWindow

    neutral, graded = _neutral_graded(tmp_path, seed=10)
    w = MainWindow()
    w._before = neutral; w._before_list.addItems(neutral)
    w._after = graded; w._after_list.addItems(graded)
    w._launch_compute(); _drain(app, w)               # compute the look (pdf default)
    assert w._look_samples is not None and not w._dirty
    look_before = w._look_samples
    w._placement.setCurrentIndex(1)                   # switch to "between" → preview-only refresh
    _drain(app, w)
    assert not w._dirty                               # placement is preview-only, look stays current
    assert w._look_samples is look_before             # not rebuilt
    w.close()


def test_refresh_renders_still_even_with_inputs_pending(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    w._before = ["a.png"]; w._after = ["b.png"]   # inputs added but not computed → look pending
    w._mark_dirty()
    assert w._dirty and w._look_samples is None
    w._still = np.random.default_rng(0).random((20, 30, 3))   # a freshly loaded still
    w._refresh_preview()           # was a no-op before the fix (guard skipped it)
    _drain(app, w)
    assert w._before_lbl.pixmap() is not None and not w._before_lbl.pixmap().isNull()
    w.close()


def test_remove_in_neutral_graded_mode(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    w._before = ["a.png", "b.png"]
    w._before_list.addItems(w._before)
    w._before_list.item(0).setSelected(True)
    w._remove_before()                              # was a no-op (lambda/bool bug)
    assert w._before == ["b.png"]
    assert w._before_list.count() == 1
    w.close()


def test_export_writes_valid_cube(app, tmp_path):
    from lutgen.app.main_window import MainWindow
    from lutgen.engine.cube_io import write_cube

    w = MainWindow()
    out = tmp_path / "gui.cube"
    write_cube(out, w._final_samples(), title="LookForge")  # same path the Export button uses
    assert read_cube(out).size == 65
