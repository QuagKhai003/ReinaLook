"""main_window — the PySide6 desktop shell (thin; all logic is in the engine).

@context  Wires widgets to the tested fitters: references OR before/after pairs, fitter/method/
          space/tone/strength controls, before/after preview, export, save preset. No color math
          here (Plan §3, ADR-0007/0014).
@done     MainWindow: references + pairs modes; Mid/Rich(mkl|pdf, oklab|rgb) controls; tone +
          strength sliders; live preview; export; save preset.
@todo     Thumbnails, drag-drop, packaging (PyInstaller).
@limits   Imports PySide6 (only when the GUI runs). Look refit on any look control; re-blend on strength.
@affects  Uses fitters (mid/rich/pairs), engine.base/strength/regularize, app.preview, preset.
          Launched by app/run.py. See ADR-0007/0014.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from lutgen.engine.base import load_base
from lutgen.engine.cube_io import write_cube
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.mid import MidFitter
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.consensus import build_consensus
from lutgen.orchestration.ingest import load_references
from lutgen.orchestration.preset import save_preset
from lutgen.orchestration.stats import compute_stats

from .preview import before_after, load_preview_still, make_test_still

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.exr)"
_PREVIEW_W = 460


def _to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w, _ = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)


def _file_list():
    w = QtWidgets.QListWidget()
    w.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    w.setMaximumHeight(110)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LookForge")
        self._base = load_base()
        self._still = make_test_still()
        self._look_samples: np.ndarray | None = None
        self._refs: list[str] = []
        self._before: list[str] = []
        self._after: list[str] = []

        self._build_ui()
        self._sync_controls()
        self._update_preview()

    # — UI —
    def _build_ui(self) -> None:
        self._mode = QtWidgets.QComboBox()
        self._mode.addItems(["References (graded only)", "Neutral + Graded (unpaired)"])
        self._mode.currentIndexChanged.connect(self._on_mode)

        # references page
        self._refs_list = _file_list()
        refs_add = QtWidgets.QPushButton("+ Add references…")
        refs_rm = QtWidgets.QPushButton("Remove selected")
        refs_add.clicked.connect(self._add_refs)
        refs_rm.clicked.connect(self._remove_refs)
        refs_page = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(refs_page)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel("Reference images (the target look)"))
        rl.addWidget(self._refs_list)
        rl.addWidget(refs_add)
        rl.addWidget(refs_rm)

        # pairs page
        self._before_list = _file_list()
        self._after_list = _file_list()
        before_add = QtWidgets.QPushButton("+ Add NEUTRAL (your footage)…")
        after_add = QtWidgets.QPushButton("+ Add GRADED (the look)…")
        before_rm = QtWidgets.QPushButton("Remove selected neutral")
        after_rm = QtWidgets.QPushButton("Remove selected graded")
        before_add.clicked.connect(self._add_before)
        after_add.clicked.connect(self._add_after)
        before_rm.clicked.connect(lambda: self._remove(self._before, self._before_list))
        after_rm.clicked.connect(lambda: self._remove(self._after, self._after_list))
        pairs_page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pairs_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(QtWidgets.QLabel("NEUTRAL images (your ungraded footage)"))
        pl.addWidget(self._before_list)
        pl.addWidget(before_add)
        pl.addWidget(before_rm)
        pl.addWidget(QtWidgets.QLabel("GRADED images (the target look)"))
        pl.addWidget(self._after_list)
        pl.addWidget(after_add)
        pl.addWidget(after_rm)
        hint = QtWidgets.QLabel("Unpaired pools — different scenes/lighting OK, counts need not "
                                "match. Transports your neutral colors toward the graded look. "
                                "Fitter/Method/Space/Tone all apply.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        pl.addWidget(hint)

        self._pages = QtWidgets.QStackedWidget()
        self._pages.addWidget(refs_page)
        self._pages.addWidget(pairs_page)

        # fitter controls (references mode)
        self._fitter = QtWidgets.QComboBox(); self._fitter.addItems(["Rich", "Mid"])
        self._method = QtWidgets.QComboBox(); self._method.addItems(["mkl", "pdf"])
        self._space = QtWidgets.QComboBox(); self._space.addItems(["oklab", "rgb"])
        for c in (self._fitter, self._method, self._space):
            c.currentIndexChanged.connect(self._on_look_changed)

        self._tone = self._slider(100, self._on_look_changed)
        self._tone_lbl = QtWidgets.QLabel("Tone (exposure match): 1.00")
        self._strength = self._slider(80, self._on_strength)
        self._strength_lbl = QtWidgets.QLabel("Strength: 0.80")

        form = QtWidgets.QFormLayout()
        form.addRow("Fitter", self._fitter)
        form.addRow("Method", self._method)
        form.addRow("Space", self._space)

        export_btn = QtWidgets.QPushButton("Export .cube…")
        preset_btn = QtWidgets.QPushButton("Save preset…")
        export_btn.clicked.connect(self._export)
        preset_btn.clicked.connect(self._save_preset)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Mode"))
        left.addWidget(self._mode)
        left.addWidget(self._pages)
        left.addSpacing(8)
        left.addLayout(form)
        left.addWidget(self._tone_lbl)
        left.addWidget(self._tone)
        left.addWidget(self._strength_lbl)
        left.addWidget(self._strength)
        left.addStretch(1)
        left.addWidget(export_btn)
        left.addWidget(preset_btn)

        self._before_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._after_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        still_btn = QtWidgets.QPushButton("Load preview still (DWG/DI frame)…")
        still_btn.clicked.connect(self._load_still)
        preview = QtWidgets.QVBoxLayout()
        preview.addWidget(still_btn)
        preview.addWidget(QtWidgets.QLabel("Before (your original / base)"))
        preview.addWidget(self._before_lbl, 1)
        preview.addWidget(QtWidgets.QLabel("After (final look)"))
        preview.addWidget(self._after_lbl, 1)

        root = QtWidgets.QHBoxLayout()
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setMaximumWidth(340)
        root.addWidget(lw)
        root.addLayout(preview, 1)
        central = QtWidgets.QWidget(); central.setLayout(root)
        self.setCentralWidget(central)

    def _slider(self, value, slot):
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(0, 100); s.setValue(value); s.valueChanged.connect(slot)
        return s

    # — state —
    def _tone_value(self) -> float:
        return self._tone.value() / 100.0

    def _strength_value(self) -> float:
        return self._strength.value() / 100.0

    def _is_pairs(self) -> bool:
        return self._mode.currentIndex() == 1

    def _build_fitter(self):
        tone = self._tone_value()
        if self._fitter.currentText() == "Mid":
            return MidFitter(tone_strength=tone)
        return RichFitter(tone_strength=tone, space=self._space.currentText(),
                          method=self._method.currentText())

    def _refit(self) -> None:
        try:
            if self._is_pairs():           # unpaired Neutral + Graded pools (ADR-0016)
                if not self._before or not self._after:
                    self._look_samples = None
                    return
                targets = load_references(self._after)
                consensus = build_consensus([compute_stats(i) for i in targets])
                src = np.concatenate([i.reshape(-1, 3) for i in load_references(self._before)])
                if src.shape[0] > 200_000:
                    src = src[np.random.default_rng(0).choice(src.shape[0], 200_000, replace=False)]
                look = self._build_fitter().fit(consensus, source_samples=src)
            else:
                if not self._refs:
                    self._look_samples = None
                    return
                consensus = build_consensus([compute_stats(i) for i in load_references(self._refs)])
                look = self._build_fitter().fit(consensus)
            self._look_samples = look(self._base)
        except Exception as exc:  # surface fitter/IO errors without crashing the UI
            self._look_samples = None
            QtWidgets.QMessageBox.warning(self, "LookForge", f"Could not build look:\n{exc}")

    def _final_samples(self) -> np.ndarray:
        if self._look_samples is None:
            return self._base
        return regularize(blend(self._base, self._look_samples, self._strength_value()))

    def _update_preview(self) -> None:
        before, after = before_after(self._still, self._base, self._final_samples())
        for lbl, img in ((self._before_lbl, before), (self._after_lbl, after)):
            lbl.setPixmap(_to_pixmap(img).scaledToWidth(
                _PREVIEW_W, QtCore.Qt.TransformationMode.SmoothTransformation))

    def _sync_controls(self) -> None:
        # both modes use the OT fitters, so fitter controls are always active.
        rich = self._fitter.currentText() == "Rich"
        self._pages.setCurrentIndex(1 if self._is_pairs() else 0)
        self._method.setEnabled(rich)
        self._space.setEnabled(rich)

    # — slots —
    def _on_mode(self, _=None) -> None:
        self._sync_controls(); self._refit(); self._update_preview()

    def _on_look_changed(self, _=None) -> None:
        self._tone_lbl.setText(f"Tone (exposure match): {self._tone_value():.2f}")
        self._sync_controls(); self._refit(); self._update_preview()

    def _on_strength(self, _=None) -> None:
        self._strength_lbl.setText(f"Strength: {self._strength_value():.2f}")
        self._update_preview()

    def _add(self, title, store, listw) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, "", _IMG_FILTER)
        new = [p for p in paths if p not in store]   # dedup: skip already-added files
        if new:
            store.extend(new)
            listw.addItems(new)
            self._refit(); self._update_preview()

    def _add_refs(self) -> None:
        self._add("Add references", self._refs, self._refs_list)

    def _add_before(self) -> None:
        self._add("Add BEFORE (neutral) frames", self._before, self._before_list)

    def _add_after(self) -> None:
        self._add("Add AFTER (graded) frames", self._after, self._after_list)

    def _remove(self, store, listw) -> None:
        for item in listw.selectedItems():
            if item.text() in store:
                store.remove(item.text())
            listw.takeItem(listw.row(item))
        self._refit(); self._update_preview()

    def _remove_refs(self) -> None:
        self._remove(self._refs, self._refs_list)

    def _load_still(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load DWG/DI preview still", "", _IMG_FILTER)
        if path:
            self._still = load_preview_still(path)
            self._update_preview()

    def _export(self) -> None:
        if self._look_samples is None:
            if self._is_pairs():
                msg = (f"Add NEUTRAL and GRADED images first — now {len(self._before)} neutral, "
                       f"{len(self._after)} graded (need at least one of each).")
            else:
                msg = "Add reference images first."
            QtWidgets.QMessageBox.warning(self, "LookForge", msg)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export .cube", "look.cube", "Cube (*.cube)")
        if path:
            write_cube(path, self._final_samples(), title="LookForge")
            QtWidgets.QMessageBox.information(self, "LookForge", f"Wrote {path}")

    def _save_preset(self) -> None:
        if self._is_pairs():
            QtWidgets.QMessageBox.information(self, "LookForge", "Presets apply to References mode.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "look.json", "JSON (*.json)")
        if path:
            save_preset(path, self._refs, self._strength_value(), title="LookForge")
