"""Offscreen tests for the Learn tab (ADR-0002 b2.1) + pure recipe-summary formatting.
Qt parts skip if PySide6 is absent."""

from __future__ import annotations

import os

import numpy as np
import pytest

from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
)
from lutgen.orchestration.profile import LookProfile, load_profile

# ── pure recipe summary (no Qt needed) ────────────────────────────────

def test_recipe_summary_shows_nonneutral_hides_neutral():
    from lutgen.app.recipe import recipe_summary
    prof = LookProfile(
        model=FilmModel(
            crosstalk=CrosstalkParams(gr=0.04),
            curves=(SCurveParams(toe=0.31, slope=1.24), SCurveParams(), SCurveParams()),
            sat_luma=SatLumaParams(shadow=0.82),
            hue_zones=HueZoneParams(r_shift=0.04, b_trim=0.11),
        ),
        name="x", n_frames=6, stage_cost={"tone": 0.012},
    )
    s = recipe_summary(prof)
    assert "toe +0.31" in s and "slope 1.24" in s
    assert "G→R +0.040" in s
    assert "shadow -18%" in s
    assert "R: hue +2.3°" in s and "B: sat +11%" in s
    assert "6 frames" in s
    assert s.count("neutral") == 3          # empty film-system + split-tone + hue-curve groups...
    # ...but an identity profile collapses to all-neutral groups:
    assert recipe_summary(LookProfile(model=FilmModel.identity())).count("neutral") == 7


# ── Qt widget tests (offscreen) ───────────────────────────────────────

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _drain(app, tab, timeout_s=60):
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


def _add_paths(tab, paths):
    tab._paths.extend(paths)
    tab._list.addItems(paths)
    tab._update_hint()


def test_learn_tab_builds_and_hint_tiers(app):
    from lutgen.app.learn_tab import LearnTab, _hint_color
    tab = LearnTab()
    assert not tab._learn_btn.isEnabled()               # empty pool -> can't learn
    _add_paths(tab, ["a.png"])
    assert "absorb" in tab._hint.text()                 # single-image wall surfaced
    assert _hint_color(1) == "#c33"
    _add_paths(tab, [f"{i}.png" for i in range(5)])
    assert _hint_color(len(tab._paths)) == "#2a2"
    assert tab._learn_btn.isEnabled()


def test_learn_tab_remove_updates_hint(app):
    from lutgen.app.learn_tab import LearnTab
    tab = LearnTab()
    _add_paths(tab, ["a.png", "b.png"])
    tab._list.item(0).setSelected(True)
    tab._remove()
    assert tab._paths == ["b.png"] and tab._list.count() == 1


def test_learn_tab_fits_and_saves(app, tmp_path):
    from lutgen.app.learn_tab import LearnTab
    from lutgen.orchestration.profile import save_profile as _  # noqa: F401
    tab = LearnTab()
    _add_paths(tab, _frames(tmp_path, 5))
    tab._fast.setChecked(True)                          # draft fit keeps the test quick
    tab._launch()
    assert not tab._cancel_btn.isHidden()               # running state (offscreen: isHidden, not isVisible)
    _drain(app, tab)
    assert isinstance(tab._profile, LookProfile)
    assert tab._profile.n_frames == 5
    assert tab._save_btn.isEnabled()
    assert "Fit: 5 frames" in tab._summary.toPlainText()

    # save via the same path the dialog handler uses
    out = tmp_path / "mylook.json"
    tab._profile.name = out.stem
    from lutgen.orchestration.profile import save_profile
    save_profile(out, tab._profile)
    assert load_profile(out).n_frames == 5


def test_learn_tab_cancel_mid_fit(app, tmp_path):
    from lutgen.app.learn_tab import LearnTab
    tab = LearnTab()
    _add_paths(tab, _frames(tmp_path, 5))
    tab._launch()
    tab._on_cancel()                                    # cancel immediately (stage-boundary)
    _drain(app, tab)
    assert tab._profile is None                         # nothing produced
    assert not tab._save_btn.isEnabled()
    assert tab._learn_btn.isEnabled()                   # UI back to idle


def test_main_window_has_learn_tab(app):
    from lutgen.app.main_window import MainWindow
    w = MainWindow()
    labels = [w._tabs.tabText(i) for i in range(w._tabs.count())]
    assert "Learn" in labels and "Match (legacy)" in labels
    w.close()
