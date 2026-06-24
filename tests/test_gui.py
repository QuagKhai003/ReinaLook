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


def test_window_builds_and_previews(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    assert w.windowTitle() == "LookForge"
    # no refs yet → final == base
    np.testing.assert_array_equal(w._final_samples(), w._base)
    w._on_strength(50)  # must not raise; updates preview


def test_mode_and_fitter_controls(app):
    from lutgen.app.main_window import MainWindow

    w = MainWindow()
    # toggle fitter/method/space — must not raise; no refs so look stays None (base)
    w._fitter.setCurrentText("Mid"); app.processEvents()
    w._fitter.setCurrentText("Rich"); w._method.setCurrentText("pdf"); app.processEvents()
    w._space.setCurrentText("rgb"); app.processEvents()
    np.testing.assert_array_equal(w._final_samples(), w._base)
    # switch to Neutral+Graded pools — page swaps, fitter controls stay active
    w._mode.setCurrentIndex(1); app.processEvents()
    assert w._is_pairs() and w._fitter.isEnabled()
    assert w._final_samples().shape == (35937, 3)


def test_export_writes_valid_cube(app, tmp_path):
    from lutgen.app.main_window import MainWindow
    from lutgen.engine.cube_io import write_cube

    w = MainWindow()
    out = tmp_path / "gui.cube"
    write_cube(out, w._final_samples(), title="LookForge")  # same path the Export button uses
    assert read_cube(out).size == 33
