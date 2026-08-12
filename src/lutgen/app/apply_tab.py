"""apply_tab — the Apply mode page: saved Look Profile -> preview -> validated .cube export.

@context  Second slice of the Learn/Apply UI (ADR-0002 b2.2, spec §4/§9): load a profile from
          the library (recents persisted), preview it on a still with a zero-lag strength
          slider (cached endpoint images, lerp per tick), pick placement, and export through
          the §6 stress gate — a failing profile shows the offending block and needs an
          explicit "Export anyway".
@done     ApplyTab: load + recents (QSettings), EDITABLE recipe (RecipeEditor — edits re-bake
          the preview debounced, mark modified, save-as), MULTI-STILL preview (up to 20 DWG
          stills, prev/next + index slider, per-still endpoint cache — revisits instant,
          look/placement changes invalidate), TONE/COLOR AMOUNT dials (ADR-0005 — scale the
          recipe toward neutral per group; bake + export use the scaled model, save-as stays
          unscaled), threaded endpoint bake, instant strength lerp, gated export.
@todo     Source-adaptive trim (spec Phase 4).
@limits   GUI-only (Qt); no color math — bake/validate via orchestration/learn.py. The
          strength lerp is exact for "between" and a close approximation under node2's gamut
          clamp; EXPORT always bakes the exact cube at the chosen strength.
@affects  Mounted as the "Apply" tab by main_window.py. Uses app/worker + app/qt_image +
          app/recipe + app/preview, orchestration/learn.py + profile.py. ADR-0002 b2.2.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import DEFAULT_SIZE, load_base
from lutgen.engine.cube_io import write_cube
from lutgen.fitter.filmmodel.scale import scaled_model
from lutgen.orchestration.learn import (
    diagnose_model,
    render_cube_from_profile,
    validate_baked_cube,
)
from lutgen.orchestration.profile import LookProfile, load_profile, save_profile

from .preview import load_preview_still, make_test_still
from .qt_image import to_pixmap
from .recipe_editor import RecipeEditor
from .worker import ComputeThread

_IMG_FILTER = ("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);; All files (*)")
_PREVIEW_W = 520
_MAX_RECENTS = 8
_MAX_STILLS = 20


class ApplyTab(QtWidgets.QWidget):
    """Apply mode: profile -> preview -> gated export. Bakes on a worker thread; the strength
    slider only lerps two cached images (instant)."""

    def __init__(self) -> None:
        super().__init__()
        self._profile: LookProfile | None = None
        self._profile_path: str | None = None
        self._modified = False
        self._base = load_base()
        # multi-still preview (ADR-0004 b4.3): up to 20 DWG/DI stills, navigated by
        # arrows/slider; per-still endpoint cache so revisiting a still is instant.
        self._stills: list[np.ndarray] = [make_test_still()]
        self._still_idx = 0
        self._before_cache: dict[int, np.ndarray] = {0: apply_cube(self._stills[0], self._base)}
        self._after_cache: dict[int, np.ndarray] = {}   # idx -> strength-1 endpoint image
        self._thread: ComputeThread | None = None
        self._settings = QtCore.QSettings("ReinaLook", "ReinaLook")

        # placement/still changes re-bake the endpoints — debounced
        self._bake_timer = QtCore.QTimer(self, singleShot=True, interval=180)
        self._bake_timer.timeout.connect(self._bake_endpoints)

        self._build_ui()
        self._reload_recents()
        self._update_nav()
        self._render()

    def _on_placement_changed(self, _=None) -> None:
        self._invalidate_after_cache()                 # a placement flip stales every endpoint
        self._bake_timer.start()

    # — UI —
    def _build_ui(self) -> None:
        load_btn = QtWidgets.QPushButton("Load profile…")
        load_btn.clicked.connect(self._browse_profile)
        self._recents = QtWidgets.QComboBox()
        self._recents.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._recents.activated.connect(self._open_recent)

        self._info = QtWidgets.QLabel("No profile loaded.")
        self._info.setWordWrap(True)

        # the editable recipe (b2.3) — edits re-bake the preview debounced + enable save-as
        self._editor = RecipeEditor()
        self._editor.setEnabled(False)
        self._editor.edited.connect(self._on_edited)

        self._saveas_btn = QtWidgets.QPushButton("Save edited profile as…")
        self._saveas_btn.setEnabled(False)
        self._saveas_btn.clicked.connect(self._save_as)

        self._placement = QtWidgets.QComboBox()
        self._placement.addItems(["Replace CSTout", "Between CSTs"])
        self._placement.currentIndexChanged.connect(self._on_placement_changed)

        self._strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._strength.setRange(0, 100)
        self._strength.setValue(100)
        self._strength.valueChanged.connect(self._on_strength)
        self._strength_lbl = QtWidgets.QLabel("Strength: 1.00")

        # ADR-0005: scale the RECIPE toward neutral, per group — keep the film's palette
        # while relaxing its tonal mood (or vice versa). Re-bakes debounced; export uses
        # the scaled model. Save-as always saves the editor's model UNscaled.
        self._tone_amt = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._tone_amt.setRange(0, 100)
        self._tone_amt.setValue(100)
        self._tone_amt.valueChanged.connect(self._on_amounts)
        self._tone_amt_lbl = QtWidgets.QLabel("Tone amount: 100%  (exposure + contrast)")
        self._color_amt = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._color_amt.setRange(0, 100)
        self._color_amt.setValue(100)
        self._color_amt.valueChanged.connect(self._on_amounts)
        self._color_amt_lbl = QtWidgets.QLabel("Color amount: 100%  (palette + hue/sat)")

        still_btn = QtWidgets.QPushButton("Load preview stills (DWG/DI frames, up to 20)…")
        still_btn.clicked.connect(self._load_stills)

        # still navigation: ◀ ▶ + an index slider for fast scrubbing
        self._prev_btn = QtWidgets.QPushButton("◀")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(lambda _=None: self._set_still_index(self._still_idx - 1))
        self._next_btn = QtWidgets.QPushButton("▶")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(lambda _=None: self._set_still_index(self._still_idx + 1))
        self._still_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._still_slider.setRange(0, 0)
        self._still_slider.valueChanged.connect(self._set_still_index)
        self._still_pos = QtWidgets.QLabel("1/1")
        self._nav_row = QtWidgets.QHBoxLayout()
        self._nav_row.addWidget(self._prev_btn)
        self._nav_row.addWidget(self._still_slider, 1)
        self._nav_row.addWidget(self._next_btn)
        self._nav_row.addWidget(self._still_pos)

        self._busy = QtWidgets.QLabel("⏳ baking preview…")
        self._busy.setStyleSheet("color: #d80;")
        self._busy.setVisible(False)

        self._export_btn = QtWidgets.QPushButton("Export .cube…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)

        self._before_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._after_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(load_btn)
        controls.addWidget(QtWidgets.QLabel("Recent profiles"))
        controls.addWidget(self._recents)
        controls.addWidget(self._info)
        controls.addWidget(QtWidgets.QLabel("Recipe (editable)"))
        controls.addWidget(self._editor, 1)
        form = QtWidgets.QFormLayout()
        form.addRow("Placement", self._placement)
        controls.addLayout(form)
        controls.addWidget(self._tone_amt_lbl)
        controls.addWidget(self._tone_amt)
        controls.addWidget(self._color_amt_lbl)
        controls.addWidget(self._color_amt)
        controls.addWidget(self._strength_lbl)
        controls.addWidget(self._strength)
        controls.addWidget(self._saveas_btn)
        controls.addWidget(self._export_btn)

        cw = QtWidgets.QWidget()
        cw.setLayout(controls)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(cw)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(380)

        preview = QtWidgets.QVBoxLayout()
        preview.addWidget(still_btn)
        preview.addLayout(self._nav_row)
        preview.addWidget(self._busy)
        preview.addWidget(QtWidgets.QLabel("Before (base conversion)"))
        preview.addWidget(self._before_lbl, 1)
        preview.addWidget(QtWidgets.QLabel("After (profile applied)"))
        preview.addWidget(self._after_lbl, 1)
        pw = QtWidgets.QWidget()
        pw.setLayout(preview)

        # controls | preview in a draggable splitter (spec §9 resizable split)
        scroll.setMaximumWidth(16777215)               # splitter governs the width now
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._splitter.addWidget(scroll)
        self._splitter.addWidget(pw)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([370, 700])
        root = QtWidgets.QHBoxLayout(self)
        root.addWidget(self._splitter)

    # — profile loading + recents —
    def _browse_profile(self, _=None) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Look Profile", "", "Look Profile (*.json)")
        if path:
            self._open_profile(path)

    def _open_recent(self, index: int) -> None:
        path = self._recents.itemText(index)
        if path:
            self._open_profile(path)

    def _open_profile(self, path: str) -> None:
        try:
            profile = load_profile(path)
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "ReinaLook", f"Could not load profile:\n{exc}")
            return
        self._profile = profile
        self._profile_path = path
        self._modified = False
        self._info.setText(f"<b>{profile.name}</b> — learned from {profile.n_frames} frames")
        self._editor.set_model(profile.model)
        self._editor.setEnabled(True)
        self._saveas_btn.setEnabled(False)
        self._export_btn.setEnabled(True)
        self._push_recent(path)
        self._invalidate_after_cache()                 # new look: every still's endpoint stale
        self._bake_endpoints()

    # — recipe editing (b2.3) —
    def _on_edited(self) -> None:
        if self._profile is None:
            return
        self._profile.model = self._editor.model()
        self._modified = True
        self._saveas_btn.setEnabled(True)
        self._info.setText(f"<b>{self._profile.name}</b> — learned from "
                           f"{self._profile.n_frames} frames <i>(modified)</i>")
        self._invalidate_after_cache()                 # edited look: endpoints stale
        self._bake_timer.start()                       # debounced preview re-bake

    def _save_as(self, _=None) -> None:
        if self._profile is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Look Profile", f"{self._profile.name}-edited.json",
            "Look Profile (*.json)")
        if path:
            from pathlib import Path
            self._profile.name = Path(path).stem
            save_profile(path, self._profile)
            self._modified = False
            self._info.setText(f"<b>{self._profile.name}</b> — learned from "
                               f"{self._profile.n_frames} frames")
            self._push_recent(path)

    def _push_recent(self, path: str) -> None:
        recents = [p for p in self._read_recents() if p != path]
        recents.insert(0, path)
        self._settings.setValue("recent_profiles", recents[:_MAX_RECENTS])
        self._reload_recents()

    def _read_recents(self) -> list[str]:
        val = self._settings.value("recent_profiles", [])
        if isinstance(val, str):                       # QSettings may collapse a 1-list to str
            val = [val]
        return list(val or [])

    def _reload_recents(self) -> None:
        self._recents.clear()
        for p in self._read_recents():
            self._recents.addItem(p)
            self._recents.setItemData(self._recents.count() - 1, p,
                                      QtCore.Qt.ItemDataRole.ToolTipRole)

    # — preview (multi-still, cached endpoints + instant lerp) —
    def _placement_key(self) -> str:
        return "between" if self._placement.currentIndex() == 1 else "node2"

    def _effective_model(self):
        """The editor's model scaled by the Tone/Color amount dials (bake + export path)."""
        return scaled_model(self._profile.model,
                            self._tone_amt.value() / 100.0,
                            self._color_amt.value() / 100.0)

    def _on_amounts(self, _=None) -> None:
        self._tone_amt_lbl.setText(
            f"Tone amount: {self._tone_amt.value()}%  (exposure + contrast)")
        self._color_amt_lbl.setText(
            f"Color amount: {self._color_amt.value()}%  (palette + hue/sat)")
        if self._profile is not None:
            self._invalidate_after_cache()             # scaled look: endpoints stale
            self._bake_timer.start()

    def _before_img_at(self, idx: int) -> np.ndarray:
        if idx not in self._before_cache:              # base never changes: cache per still
            self._before_cache[idx] = apply_cube(self._stills[idx], self._base)
        return self._before_cache[idx]

    def _invalidate_after_cache(self) -> None:
        """The look changed (profile load / edit / placement): every still's endpoint is stale."""
        self._after_cache.clear()

    def _set_still_index(self, idx: int) -> None:
        idx = max(0, min(idx, len(self._stills) - 1))
        if idx == self._still_idx and self._still_slider.value() == idx:
            self._update_nav()
            return
        self._still_idx = idx
        self._update_nav()
        if idx in self._after_cache or self._profile is None:
            self._render()                             # cache hit (or no look) — instant
        else:
            self._render()                             # show 'before' immediately...
            self._bake_timer.start()                   # ...and bake this still's endpoint

    def _update_nav(self) -> None:
        n = len(self._stills)
        self._still_slider.blockSignals(True)
        self._still_slider.setRange(0, n - 1)
        self._still_slider.setValue(self._still_idx)
        self._still_slider.blockSignals(False)
        self._still_pos.setText(f"{self._still_idx + 1}/{n}")
        self._prev_btn.setEnabled(self._still_idx > 0)
        self._next_btn.setEnabled(self._still_idx < n - 1)

    def _bake_endpoints(self, _=None) -> None:
        """Bake the CURRENT still's strength-1 endpoint on the worker thread (look/placement/
        still changed). Strength ticks never come here — they lerp the cached images."""
        if self._profile is None:
            self._render()
            return
        if self._still_idx in self._after_cache:
            self._render()
            return
        if self._thread is not None and self._thread.isRunning():
            self._bake_timer.start()                   # busy — try again shortly
            return
        model = self._effective_model()
        placement = self._placement_key()
        idx = self._still_idx
        still = self._stills[idx]

        def payload(_report):
            cube = render_cube_from_profile(model, 1.0, placement=placement)
            after = apply_cube(still, cube.samples, cube.size)
            if placement == "between":                 # cube is DWG/DI→DWG/DI: show Rec.709
                after = apply_cube(after, self._base, DEFAULT_SIZE)
            return idx, after

        self._busy.setVisible(True)
        thread = ComputeThread(payload)
        thread.done.connect(self._on_baked)
        self._thread = thread
        thread.start()

    def _on_baked(self, result) -> None:
        self._busy.setVisible(False)
        if isinstance(result, Exception):
            QtWidgets.QMessageBox.warning(self, "ReinaLook", f"Preview failed:\n{result}")
            return
        idx, after = result
        self._after_cache[idx] = after
        if idx == self._still_idx:                     # user may have scrubbed away meanwhile
            self._render()

    def _strength_value(self) -> float:
        return self._strength.value() / 100.0

    def _on_strength(self, _=None) -> None:
        self._strength_lbl.setText(f"Strength: {self._strength_value():.2f}")
        self._render()                                 # cheap image lerp — instant

    def _render(self) -> None:
        before = self._before_img_at(self._still_idx)
        self._show(self._before_lbl, before)
        after_full = self._after_cache.get(self._still_idx)
        if after_full is None:
            self._show(self._after_lbl, before)
            return
        s = self._strength_value()
        self._show(self._after_lbl, (1.0 - s) * before + s * after_full)

    def _show(self, lbl, img) -> None:
        lbl.setPixmap(to_pixmap(img).scaledToWidth(
            _PREVIEW_W, QtCore.Qt.TransformationMode.SmoothTransformation))

    def _load_stills(self, _=None) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load DWG/DI preview stills (up to 20)", "", _IMG_FILTER)
        if not paths:
            return
        if len(paths) > _MAX_STILLS:
            QtWidgets.QMessageBox.information(
                self, "ReinaLook", f"Keeping the first {_MAX_STILLS} of {len(paths)} stills.")
            paths = paths[:_MAX_STILLS]
        self._stills = [load_preview_still(p) for p in paths]
        self._still_idx = 0
        self._before_cache.clear()
        self._invalidate_after_cache()
        self._update_nav()
        self._render()
        self._bake_timer.start()

    # — export (the §6 gate) —
    def _confirm_force(self, details: str) -> bool:
        """Failing validation: show the per-block blame; only an explicit override exports.
        Split out so tests can stub the dialog."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("ReinaLook — validation failed")
        box.setText("This look failed the stress validation and may band or tear on footage.")
        box.setDetailedText(details)
        force = box.addButton("Export anyway", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is force

    def _export(self, _=None) -> None:
        if self._profile is None:
            return
        placement = self._placement_key()
        cube = render_cube_from_profile(self._effective_model(), self._strength_value(),
                                        title=self._profile.name, placement=placement)
        report = validate_baked_cube(cube, placement)   # §6: mandatory before export
        if not report.ok:
            blamed = diagnose_model(self._effective_model(), self._strength_value(),
                                    placement=placement)
            lines = [f"{block}: {v}" for block, vs in blamed.items() for v in vs]
            unattributed = [str(v) for v in report.violations
                            if not any(str(v) in vs for vs in blamed.values())]
            details = "\n".join(lines + unattributed) or report.summary()
            if not self._confirm_force(details):
                return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export .cube", f"{self._profile.name}.cube", "Cube (*.cube)")
        if path:
            write_cube(path, cube.samples, cube.size, title=cube.title)
            QtWidgets.QMessageBox.information(self, "ReinaLook", f"Wrote {path}")
