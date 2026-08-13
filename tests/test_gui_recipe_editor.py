"""Offscreen tests for the recipe edit layer (ADR-0002 b2.3): RecipeEditor round-trip +
the ApplyTab edit → modified → re-bake → save-as flow."""

from __future__ import annotations

import math
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
)
from lutgen.orchestration.profile import (
    LookProfile,
    load_profile,
    save_profile,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path):
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat,
                             QtCore.QSettings.Scope.UserScope, str(tmp_path / "settings"))
    yield


def _model() -> FilmModel:
    return FilmModel(
        crosstalk=CrosstalkParams(gr=0.08, rb=-0.02),
        curves=(SCurveParams(toe=0.4, slope=1.25, pivot=0.45), SCurveParams(slope=1.1),
                SCurveParams(shoulder=0.5, slope=0.9)),
        sat_luma=SatLumaParams(shadow=0.7, mid=1.25, high=0.85),
        hue_zones=HueZoneParams(r_shift=0.12, b_trim=0.2),
    )


def _drain(app, tab, timeout_s=60):
    for _ in range(timeout_s * 20):
        app.processEvents()
        if tab._thread is None or not tab._thread.isRunning():
            break
        tab._thread.wait(50)
    app.processEvents()


# ── RecipeEditor alone ────────────────────────────────────────────────

def test_editor_roundtrip_within_display_precision(app):
    from lutgen.app.recipe_editor import RecipeEditor
    ed = RecipeEditor()
    m = _model()
    ed.set_model(m)
    out = ed.model()
    assert out.crosstalk.gr == pytest.approx(0.08, abs=1e-4)
    assert out.curves[0].toe == pytest.approx(0.4, abs=1e-4)
    assert out.curves[0].pivot == pytest.approx(0.45, abs=1e-4)
    assert out.sat_luma.mid == pytest.approx(1.25, abs=1e-4)
    assert out.hue_zones.r_shift == pytest.approx(0.12, abs=1e-3)   # degrees round-trip
    assert out.hue_zones.b_trim == pytest.approx(0.2, abs=1e-3)


def test_editor_identity_roundtrip(app):
    from lutgen.app.recipe_editor import RecipeEditor
    ed = RecipeEditor()
    ed.set_model(FilmModel.identity())
    assert ed.model().is_identity()


def test_editor_set_model_does_not_emit_edited(app):
    from lutgen.app.recipe_editor import RecipeEditor
    ed = RecipeEditor()
    hits = []
    ed.edited.connect(lambda: hits.append(1))
    ed.set_model(_model())
    assert hits == []


def test_editor_user_change_emits_and_reflects(app):
    from lutgen.app.recipe_editor import RecipeEditor
    ed = RecipeEditor()
    ed.set_model(FilmModel.identity())
    hits = []
    ed.edited.connect(lambda: hits.append(1))
    spin = ed._spins[("curves", "g", "slope")]
    spin.setValue(1.3)
    assert hits, "edited signal must fire on user change"
    assert ed.model().curves[1].slope == pytest.approx(1.3, abs=1e-6)


def test_editor_degree_percent_units(app):
    from lutgen.app.recipe_editor import RecipeEditor
    ed = RecipeEditor()
    ed.set_model(FilmModel(hue_zones=HueZoneParams(r_shift=math.radians(10), r_trim=0.25)))
    assert ed._spins[("hue_zones", "r_shift")].value() == pytest.approx(10.0, abs=1e-3)
    assert ed._spins[("hue_zones", "r_trim")].value() == pytest.approx(25.0, abs=1e-3)


def test_editor_ranges_mirror_fit_bounds(app):
    from lutgen.app.recipe_editor import RecipeEditor
    ed = RecipeEditor()
    s = ed._spins[("curves", "r", "slope")]
    assert (s.minimum(), s.maximum()) == (0.5, 2.0)
    ct = ed._spins[("crosstalk", "rg")]
    assert (ct.minimum(), ct.maximum()) == (-0.25, 0.25)


# ── ApplyTab integration ──────────────────────────────────────────────

def _loaded_tab(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    path = tmp_path / "p.json"
    save_profile(path, LookProfile(model=_model(), name="p", n_frames=5))
    tab = ApplyTab()
    tab._open_profile(str(path))
    _drain(app, tab)
    return tab


def test_apply_edit_marks_modified_and_rebakes(app, tmp_path):
    tab = _loaded_tab(app, tmp_path)
    assert not tab._saveas_btn.isEnabled()
    spin = tab._editor._spins[("sat_luma", "mid")]
    spin.setValue(1.28)                                 # user edit (1.25 = loaded value)
    assert tab._modified and tab._saveas_btn.isEnabled()
    assert "(modified)" in tab._info.text()
    assert tab._bake_timer.isActive()                   # debounced re-bake scheduled
    assert tab._profile.model.sat_luma.mid == pytest.approx(1.28, abs=1e-6)


def test_apply_edited_model_is_exported_model(app, tmp_path, monkeypatch):
    from lutgen.engine.cube_io import read_cube
    tab = _loaded_tab(app, tmp_path)
    tab._editor._spins[("curves", "r", "slope")].setValue(1.9)
    tab._bake_timer.stop()                              # don't wait for the debounce
    out = tmp_path / "edited.cube"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Cube (*.cube)")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    tab._export()
    assert out.exists() and read_cube(out).size == 65


def test_apply_save_as_roundtrip(app, tmp_path, monkeypatch):
    tab = _loaded_tab(app, tmp_path)
    tab._editor._spins[("hue_zones", "g_shift")].setValue(-5.0)   # degrees
    out = tmp_path / "tweaked.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Look Profile (*.json)")))
    tab._save_as()
    assert out.exists()
    loaded = load_profile(out)
    assert loaded.name == "tweaked"
    assert loaded.model.hue_zones.g_shift == pytest.approx(math.radians(-5.0), abs=1e-6)
    assert not tab._modified
    assert "(modified)" not in tab._info.text()
