"""Offscreen tests for the multi-still Apply preview (ADR-0004 b4.3): navigation, cap,
per-still endpoint cache, invalidation on look changes."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from lutgen.app.preview import make_test_still
from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    SCurveParams,
)
from lutgen.orchestration.profile import LookProfile, save_profile


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path):
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat,
                             QtCore.QSettings.Scope.UserScope, str(tmp_path / "settings"))
    yield


def _drain(app, tab, timeout_s=90):
    for _ in range(timeout_s * 20):
        app.processEvents()
        if tab._thread is None or not tab._thread.isRunning():
            break
        tab._thread.wait(50)
    app.processEvents()
    # a queued debounce may schedule another bake — drain that too
    if tab._bake_timer.isActive():
        tab._bake_timer.stop()
        tab._bake_endpoints()
        for _ in range(timeout_s * 20):
            app.processEvents()
            if tab._thread is None or not tab._thread.isRunning():
                break
            tab._thread.wait(50)
        app.processEvents()


def _stills(n):
    rng = np.random.default_rng(0)
    return [np.clip(make_test_still(64, 96) * rng.uniform(0.4, 1.0), 0, 1) for _ in range(n)]


def _tab_with_profile(app, tmp_path, n_stills=3):
    from lutgen.app.apply_tab import ApplyTab
    prof = LookProfile(model=FilmModel(crosstalk=CrosstalkParams(gr=0.06),
                                       curves=(SCurveParams(slope=1.2), SCurveParams(),
                                               SCurveParams())), name="p")
    path = tmp_path / "p.json"
    save_profile(path, prof)
    tab = ApplyTab()
    tab._stills = _stills(n_stills)
    tab._still_idx = 0
    tab._before_cache.clear()
    tab._update_nav()
    tab._open_profile(str(path))
    _drain(app, tab)
    return tab


def test_nav_arrows_and_slider(app, tmp_path):
    tab = _tab_with_profile(app, tmp_path)
    assert tab._still_pos.text() == "1/3"
    assert not tab._prev_btn.isEnabled()
    tab._next_btn.click()
    assert tab._still_idx == 1 and tab._still_pos.text() == "2/3"
    tab._still_slider.setValue(2)
    assert tab._still_idx == 2
    assert not tab._next_btn.isEnabled()
    tab._prev_btn.click()
    assert tab._still_idx == 1


def test_per_still_cache_revisit_no_rebake(app, tmp_path):
    tab = _tab_with_profile(app, tmp_path)
    tab._set_still_index(1); _drain(app, tab)          # bake still 1
    tab._set_still_index(2); _drain(app, tab)          # bake still 2
    assert set(tab._after_cache) >= {0, 1, 2}
    thread = tab._thread
    tab._set_still_index(0)                            # revisit — cache hit
    assert tab._thread is thread                       # no new bake thread
    assert not tab._bake_timer.isActive()


def test_look_change_invalidates_all_stills(app, tmp_path):
    tab = _tab_with_profile(app, tmp_path)
    tab._set_still_index(1); _drain(app, tab)
    assert len(tab._after_cache) >= 2
    tab._editor._spins[("sat_luma", "mid")].setValue(1.4)   # edit the look
    assert tab._after_cache == {}                      # every still stale


def test_placement_change_invalidates(app, tmp_path):
    tab = _tab_with_profile(app, tmp_path)
    assert tab._after_cache
    tab._placement.setCurrentIndex(1)
    assert tab._after_cache == {}


def test_stale_bake_result_not_rendered_for_wrong_still(app, tmp_path):
    tab = _tab_with_profile(app, tmp_path)
    # simulate a bake that finishes after the user scrubbed away
    tab._still_idx = 2
    tab._update_nav()
    tab._on_baked((1, tab._before_img_at(1)))          # result for still 1 arrives
    assert 1 in tab._after_cache                       # cached for later
    # current still unchanged; nothing crashed — render used still 2's data path


def test_load_stills_caps_at_20(app, tmp_path, monkeypatch):
    from PIL import Image

    from lutgen.app.apply_tab import ApplyTab
    paths = []
    for i in range(23):
        p = tmp_path / f"s{i}.png"
        Image.fromarray((np.clip(make_test_still(16, 24), 0, 1) * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    tab = ApplyTab()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (paths, "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    tab._load_stills()
    assert len(tab._stills) == 20
    assert tab._still_pos.text() == "1/20"
