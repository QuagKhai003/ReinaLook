"""Offscreen tests for the Apply tab (ADR-0002 b2.2): profile load + recents, cached-endpoint
preview, instant strength lerp, §6-gated export."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from lutgen.engine.cube_io import read_cube
from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    HueZoneParams,
    SCurveParams,
)
from lutgen.orchestration.profile import LookProfile, save_profile


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Keep QSettings out of the real registry."""
    from PySide6 import QtCore
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat,
                             QtCore.QSettings.Scope.UserScope, str(tmp_path / "settings"))
    yield


def _drain(app, tab, timeout_s=60):
    for _ in range(timeout_s * 20):
        app.processEvents()
        if tab._thread is None or not tab._thread.isRunning():
            break
        tab._thread.wait(50)
    app.processEvents()


def _good_profile(tmp_path, name="warmlook"):
    prof = LookProfile(
        model=FilmModel(crosstalk=CrosstalkParams(gr=0.06),
                        curves=(SCurveParams(toe=0.3, slope=1.2), SCurveParams(),
                                SCurveParams(shoulder=0.4))),
        name=name, n_frames=6,
    )
    path = tmp_path / f"{name}.json"
    save_profile(path, prof)
    return str(path)


def _broken_profile(tmp_path):
    prof = LookProfile(model=FilmModel(hue_zones=HueZoneParams(r_shift=3.0, c_shift=-3.0)),
                       name="broken")
    path = tmp_path / "broken.json"
    save_profile(path, prof)
    return str(path)


def test_apply_tab_builds_disabled(app):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    assert not tab._export_btn.isEnabled()
    assert tab._after_cache == {}


def test_load_profile_bakes_and_enables(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    tab._open_profile(_good_profile(tmp_path))
    _drain(app, tab)
    assert tab._export_btn.isEnabled()
    assert tab._after_cache.get(0) is not None
    assert "warmlook" in tab._info.text()
    assert tab._editor.isEnabled()
    assert tab._editor.model().curves[0].toe == pytest.approx(0.3, abs=1e-4)
    assert tab._recents.itemText(0).endswith("warmlook.json")


def test_strength_tick_is_pure_lerp_no_thread(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    tab._open_profile(_good_profile(tmp_path))
    _drain(app, tab)
    thread_before = tab._thread
    tab._strength.setValue(35)                          # slider tick
    assert tab._thread is thread_before                 # no new bake
    assert "0.35" in tab._strength_lbl.text()


def test_placement_switch_rebakes(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    tab._open_profile(_good_profile(tmp_path))
    _drain(app, tab)
    after_node2 = tab._after_cache[0].copy()
    tab._placement.setCurrentIndex(1)                   # Between CSTs
    tab._bake_timer.stop(); tab._bake_endpoints()       # skip debounce in test
    _drain(app, tab)
    assert not np.allclose(tab._after_cache[0], after_node2)


def test_export_good_profile_writes_cube(app, tmp_path, monkeypatch):
    from lutgen.app import apply_tab as mod
    tab = mod.ApplyTab()
    tab._open_profile(_good_profile(tmp_path))
    _drain(app, tab)
    out = tmp_path / "out.cube"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Cube (*.cube)")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    tab._export()
    assert out.exists()
    assert read_cube(out).size == 65


def test_export_broken_profile_gated(app, tmp_path, monkeypatch):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    tab._open_profile(_broken_profile(tmp_path))
    tab._placement.setCurrentIndex(1)                   # 'between' shows the breakage
    _drain(app, tab)
    seen = {}

    def fake_confirm(details):
        seen["details"] = details
        return False                                    # user declines

    tab._confirm_force = fake_confirm
    out = tmp_path / "never.cube"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Cube (*.cube)")))
    tab._export()
    assert not out.exists()                             # gate held
    assert "hue zones" in seen["details"]               # block named


def test_export_broken_profile_forced(app, tmp_path, monkeypatch):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    tab._open_profile(_broken_profile(tmp_path))
    tab._placement.setCurrentIndex(1)
    _drain(app, tab)
    tab._confirm_force = lambda details: True           # explicit override
    out = tmp_path / "forced.cube"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Cube (*.cube)")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    tab._export()
    assert out.exists()


def test_main_window_modes_and_title_follow_tab(app):
    from lutgen.app.main_window import MainWindow
    w = MainWindow()
    labels = [w._tabs.tabText(i) for i in range(w._tabs.count())]
    assert labels == ["Learn", "Apply", "Match (legacy)"]
    w._tabs.setCurrentIndex(1)
    assert w.windowTitle() == "ReinaLook — Apply"
    w._tabs.setCurrentIndex(2)
    assert w.windowTitle() == "ReinaLook — Match (legacy)"
    w.close()


def test_main_window_restores_last_mode(app):
    from PySide6 import QtCore

    from lutgen.app.main_window import MainWindow
    QtCore.QSettings("ReinaLook", "ReinaLook").setValue("last_mode", 1)
    w = MainWindow()
    assert w._tabs.currentIndex() == 1                  # Apply restored
    w.close()
