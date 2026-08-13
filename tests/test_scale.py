"""Deterministic tests for recipe scaling (ADR-0005): pure scaled_model + the Apply tab
Tone/Color amount dials."""

from __future__ import annotations

import os

import numpy as np
import pytest

from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    GlobalParams,
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
)
from lutgen.fitter.filmmodel.scale import scaled_model


def _model() -> FilmModel:
    return FilmModel(
        global_trim=GlobalParams(exposure=-0.12),
        crosstalk=CrosstalkParams(gr=0.08, rb=-0.04),
        curves=(SCurveParams(toe=0.4, shoulder=0.2, slope=1.4, pivot=0.6),
                SCurveParams(slope=0.8), SCurveParams(toe=0.2, slope=1.2, pivot=0.4)),
        sat_luma=SatLumaParams(shadow=0.7, mid=1.3, high=0.8),
        hue_zones=HueZoneParams(r_shift=0.2, b_trim=0.3),
    )


# ── pure scaling ──────────────────────────────────────────────────────

def test_full_amounts_reproduce_model():
    assert scaled_model(_model(), 1.0, 1.0) == _model()


def test_zero_amounts_are_identity():
    assert scaled_model(_model(), 0.0, 0.0).is_identity()


def test_tone_scaling_shrinks_shared_shape_keeps_cast():
    # tone 0.5: exposure halves, the channels' MEAN shape halves toward neutral,
    # but the between-channel differences (the cast) stay at full amount
    m0, m = _model(), scaled_model(_model(), 0.5, 1.0)
    assert m.global_trim.exposure == pytest.approx(-0.06)
    slopes0 = [c.slope for c in m0.curves]
    slopes = [c.slope for c in m.curves]
    mean0 = sum(slopes0) / 3
    assert sum(slopes) / 3 == pytest.approx(1 + (mean0 - 1) * 0.5)      # shared part halved
    assert slopes[0] - slopes[1] == pytest.approx(slopes0[0] - slopes0[1])  # cast intact
    assert m.crosstalk == m0.crosstalk                  # colour untouched
    assert m.hue_zones == m0.hue_zones


def test_color_zero_removes_cast_keeps_contrast():
    # colour 0: all three curves collapse to the SHARED shape — no channel cast — while
    # tone (exposure + mean contrast) stays at full amount. This is the user's yellow-cast dial.
    m0, m = _model(), scaled_model(_model(), 1.0, 0.0)
    assert m.curves[0] == m.curves[1] == m.curves[2]    # identical channels = no cast
    mean_slope0 = sum(c.slope for c in m0.curves) / 3
    assert m.curves[0].slope == pytest.approx(mean_slope0)
    assert m.global_trim == m0.global_trim              # tone untouched
    assert m.crosstalk.is_identity() and m.hue_zones.is_identity()


def test_color_only_scaling_blocks():
    m = scaled_model(_model(), 1.0, 0.5)
    assert m.crosstalk.gr == pytest.approx(0.04)
    assert m.sat_luma.mid == pytest.approx(1.15)        # 1 + (1.3-1)*0.5
    assert m.hue_zones.r_shift == pytest.approx(0.1)
    assert m.global_trim == _model().global_trim        # tone untouched


def test_amounts_clamped_no_overdrive():
    m = scaled_model(_model(), 2.0, -1.0)
    assert m.curves[0].slope <= 1.4 + 1e-12             # no extrapolation past the fit
    assert m.crosstalk.is_identity() and m.hue_zones.is_identity()


def test_recomposed_params_stay_in_fit_bounds():
    # deviation-only mix (tone 0, colour 1) is not convex — the clamp must hold the bounds
    for t in (0.0, 0.3, 0.7, 1.0):
        for c in (0.0, 0.3, 0.7, 1.0):
            m = scaled_model(_model(), t, c)
            for cv in m.curves:
                assert 0.0 <= cv.toe <= 2.0 and 0.0 <= cv.shoulder <= 2.0
                assert 0.5 <= cv.slope <= 2.0 and 0.3 <= cv.pivot <= 0.7


def test_scaled_curves_stay_monotonic():
    x = np.linspace(0, 1, 257)
    grid = np.column_stack([x, x, x])
    for t, c in ((0.25, 1.0), (0.5, 0.5), (0.75, 0.0), (0.0, 1.0)):
        out = scaled_model(_model(), t, c).forward(grid)
        assert np.all(np.diff(out, axis=0) > -1e-9)


# ── Apply tab dials (offscreen) ───────────────────────────────────────

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

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


def _drain(app, tab, timeout_s=90):
    for _ in range(timeout_s * 20):
        app.processEvents()
        if tab._thread is None or not tab._thread.isRunning():
            break
        tab._thread.wait(50)
    app.processEvents()


def _loaded_tab(app, tmp_path):
    from lutgen.app.apply_tab import ApplyTab
    path = tmp_path / "p.json"
    save_profile(path, LookProfile(model=_model(), name="p"))
    tab = ApplyTab()
    tab._open_profile(str(path))
    _drain(app, tab)
    return tab


def test_dials_default_full_and_scale_effective_model(app, tmp_path):
    from dataclasses import replace
    tab = _loaded_tab(app, tmp_path)
    # 100/100 = the model itself, minus film brightness (opt-in, default off)
    assert tab._effective_model() == replace(_model(), global_trim=GlobalParams())
    tab._bake_exposure.setChecked(True)
    assert tab._effective_model() == _model()
    tab._tone_amt.setValue(50)
    assert "50%" in tab._tone_amt_lbl.text()
    assert tab._effective_model().global_trim.exposure == pytest.approx(-0.06)
    assert tab._after_cache == {}                       # dial invalidated the endpoints
    assert tab._bake_timer.isActive()


def test_export_uses_scaled_model(app, tmp_path, monkeypatch):
    from lutgen.engine.base import DEFAULT_SIZE, load_base
    from lutgen.engine.cube_io import read_cube
    tab = _loaded_tab(app, tmp_path)
    tab._tone_amt.setValue(0)
    tab._color_amt.setValue(0)                          # amounts 0/0 -> identity model
    tab._bake_timer.stop()
    out = tmp_path / "flat.cube"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Cube (*.cube)")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    tab._export()                                       # identity look at strength 1 = base
    # atol 1e-6: .cube text format carries 6 decimals (file round-trip is ~5e-7)
    np.testing.assert_allclose(read_cube(out).samples, load_base(DEFAULT_SIZE), atol=1e-6)


def test_save_as_stays_unscaled(app, tmp_path, monkeypatch):
    tab = _loaded_tab(app, tmp_path)
    tab._tone_amt.setValue(30)
    tab._editor._spins[("sat_luma", "mid")].setValue(1.3)   # ensure modified/save enabled
    tab._bake_timer.stop()
    out = tmp_path / "saved.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "Look Profile (*.json)")))
    tab._save_as()
    saved = load_profile(out)
    assert saved.model.global_trim.exposure == pytest.approx(-0.12, abs=1e-6)  # NOT scaled


def test_film_brightness_opt_in_default_off(app, tmp_path):
    tab = _loaded_tab(app, tmp_path)
    assert not tab._bake_exposure.isChecked()           # default: footage keeps its exposure
    assert tab._effective_model().global_trim.is_identity()
    tab._bake_exposure.setChecked(True)
    assert tab._effective_model().global_trim.exposure == pytest.approx(-0.12)
    assert tab._after_cache == {}                       # toggle invalidated endpoints


def test_cli_apply_film_brightness_flag(tmp_path, monkeypatch):
    from PySide6 import QtWidgets as _qt  # noqa: F401 (env already offscreen)

    from lutgen.cli import main
    from lutgen.engine.base import DEFAULT_SIZE, load_base
    from lutgen.engine.cube_io import read_cube
    path = tmp_path / "p.json"
    save_profile(path, LookProfile(model=FilmModel(
        global_trim=__import__("lutgen.fitter.filmmodel", fromlist=["GlobalParams"]).GlobalParams(exposure=-0.15)),
        name="p"))
    out_def = tmp_path / "def.cube"
    assert main(["apply", "--profile", str(path), "--out", str(out_def)]) == 0
    # default: exposure NOT baked -> cube == base
    np.testing.assert_allclose(read_cube(out_def).samples, load_base(DEFAULT_SIZE), atol=1e-6)
    out_fb = tmp_path / "fb.cube"
    assert main(["apply", "--profile", str(path), "--out", str(out_fb), "--film-brightness"]) == 0
    assert not np.allclose(read_cube(out_fb).samples, load_base(DEFAULT_SIZE), atol=1e-3)
