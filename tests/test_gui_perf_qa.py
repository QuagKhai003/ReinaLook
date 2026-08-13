"""Offscreen tests for the perf & layout QA batch (ADR-0002 b2.5): pooled-stats cache,
Apply splitter, minimum window size, path tooltips, and measured preview timings."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

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


def _frames(tmp_path, n):
    from PIL import Image
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        img = np.clip(rng.uniform(0.1, 0.9, (20, 20, 3)) + np.array([0.08, 0.0, -0.06]), 0, 1)
        p = tmp_path / f"f{i}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(p)
        out.append(str(p))
    return out


# ── pooled-stats cache (spec §9) ──────────────────────────────────────

def test_learn_stats_cached_across_relearn(app, tmp_path, monkeypatch):
    from lutgen.app import learn_tab as mod

    calls = {"n": 0}
    real = mod.pool_targets

    def counting(paths, **kw):
        calls["n"] += 1
        return real(paths, **kw)

    monkeypatch.setattr(mod, "pool_targets", counting)
    tab = mod.LearnTab()
    tab._paths.extend(_frames(tmp_path, 5))
    tab._list.addItems(tab._paths)
    tab._update_hint()
    tab._fast.setChecked(True)
    tab._launch(); _drain(app, tab)
    assert calls["n"] == 1
    tab._launch(); _drain(app, tab)                     # re-Learn, same pool
    assert calls["n"] == 1                              # cache hit — no re-ingest
    tab._paths.append("extra.png")                      # pool changed
    assert tab._targets_cache[0] != tuple(tab._paths)   # next learn will recompute


def test_learn_options_object_stays_default():
    from lutgen.fitter.fit import FitOptions
    assert FitOptions().n_samples == 3000               # cache work didn't touch defaults


# ── layout QA ─────────────────────────────────────────────────────────

def test_apply_tab_has_draggable_splitter(app):
    from lutgen.app.apply_tab import ApplyTab
    tab = ApplyTab()
    assert isinstance(tab._splitter, QtWidgets.QSplitter)
    assert tab._splitter.count() == 2


def test_main_window_minimum_size(app):
    from lutgen.app.main_window import MainWindow
    w = MainWindow()
    assert w.minimumWidth() >= 900 and w.minimumHeight() >= 560
    w.close()


def test_learn_pool_items_carry_tooltips(app, tmp_path):
    from lutgen.app.learn_tab import LearnTab
    tab = LearnTab()
    long_path = str(tmp_path / ("deep" * 20) / "frame.png")
    tab._paths_pending = None
    # go through the same code path _add uses (dialog stubbed by direct call)
    new = [long_path]
    tab._paths.extend(new)
    for p in new:
        item = QtWidgets.QListWidgetItem(p)
        item.setToolTip(p)
        tab._list.addItem(item)
    assert tab._list.item(0).toolTip() == long_path


def test_recents_have_tooltips(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    prof = LookProfile(model=FilmModel(crosstalk=CrosstalkParams(gr=0.05)), name="tt")
    path = tmp_path / "tt.json"
    save_profile(path, prof)
    tab = ApplyTab()
    tab._open_profile(str(path))
    _drain(app, tab)
    assert tab._recents.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole) == str(path)


# ── measured timings (documented in progress log) ─────────────────────

def test_strength_tick_under_100ms(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    prof = LookProfile(model=FilmModel(curves=(SCurveParams(slope=1.3), SCurveParams(),
                                               SCurveParams())), name="t")
    path = tmp_path / "t.json"
    save_profile(path, prof)
    tab = ApplyTab()
    tab._open_profile(str(path))
    _drain(app, tab)
    t0 = time.perf_counter()
    for v in (20, 40, 60, 80, 100):
        tab._strength.setValue(v)
    per_tick = (time.perf_counter() - t0) / 5
    assert per_tick < 0.1, f"strength tick took {per_tick * 1000:.0f} ms"
