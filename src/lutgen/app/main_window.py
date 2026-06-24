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

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import load_base
from lutgen.engine.cube_io import write_cube
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.mid import MidFitter
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.consensus import build_consensus
from lutgen.orchestration.ingest import load_image
from lutgen.orchestration.pipeline import _assemble
from lutgen.orchestration.preset import save_preset
from lutgen.orchestration.stats import compute_stats

from .preview import load_preview_still, make_test_still

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.exr)"
_PREVIEW_W = 520
_DEBOUNCE_MS = 2000   # wait this long after the LAST slider/control change, then compute


def _to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w, _ = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)


class _ComputeThread(QtCore.QThread):
    """Run the (possibly heavy) look + preview computation off the UI thread."""

    done = QtCore.Signal(object)    # result tuple or Exception
    progress = QtCore.Signal(int)   # 0..100

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(self.progress.emit))   # fn receives a report(pct) callback
        except Exception as exc:   # report back to the UI thread
            self.done.emit(exc)


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
        self._before_img = apply_cube(self._still, self._base)  # cached preview "before"
        self._look_samples: np.ndarray | None = None
        self._refs: list[str] = []
        self._before: list[str] = []
        self._after: list[str] = []

        # nothing computes automatically — only the Compute button triggers a (threaded) compute.
        self._still_dirty = True        # recompute "before" on the first/next compute
        self._dirty = False             # settings changed since last compute
        self._thread: _ComputeThread | None = None

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
        self._placement = QtWidgets.QComboBox()
        self._placement.addItems(["Replace Node 2", "Between Node 1 & 2"])
        for c in (self._fitter, self._method, self._space):
            c.currentIndexChanged.connect(self._on_fitter_changed)
        self._placement.currentIndexChanged.connect(self._on_placement)

        self._tone = self._slider(100, self._on_tone)
        self._tone_lbl = QtWidgets.QLabel("Tone (exposure match): 1.00")
        self._strength = self._slider(80, self._on_strength)
        self._strength_lbl = QtWidgets.QLabel("Strength: 0.80")

        form = QtWidgets.QFormLayout()
        form.addRow("Placement", self._placement)
        form.addRow("Fitter", self._fitter)
        form.addRow("Method", self._method)
        form.addRow("Space", self._space)

        self._compute_btn = QtWidgets.QPushButton("Compute preview")
        self._compute_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._compute_btn.clicked.connect(self._launch_compute)
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
        left.addWidget(self._compute_btn)
        left.addWidget(export_btn)
        left.addWidget(preset_btn)

        self._before_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._after_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        still_btn = QtWidgets.QPushButton("Load preview still (DWG/DI frame)…")
        still_btn.clicked.connect(self._load_still)

        # busy indicator: a percentage progress bar + label, shown while computing
        self._busy = QtWidgets.QProgressBar()
        self._busy.setRange(0, 100)
        self._busy.setValue(0)
        self._busy.setTextVisible(True)
        self._busy.setFormat("%p%")
        self._busy.setMaximumHeight(16)
        self._busy_lbl = QtWidgets.QLabel("⏳ computing…")
        self._busy_lbl.setStyleSheet("color: #d80;")
        busy_row = QtWidgets.QHBoxLayout()
        busy_row.addWidget(self._busy_lbl)
        busy_row.addWidget(self._busy, 1)
        self._set_busy(False)

        preview = QtWidgets.QVBoxLayout()
        preview.addWidget(still_btn)
        preview.addLayout(busy_row)
        preview.addWidget(QtWidgets.QLabel("Before (your original / base)"))
        preview.addWidget(self._before_lbl, 1)
        preview.addWidget(QtWidgets.QLabel("After (final look)"))
        preview.addWidget(self._after_lbl, 1)

        root = QtWidgets.QHBoxLayout()
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setMaximumWidth(340)
        root.addWidget(lw)
        root.addLayout(preview, 1)
        self._central = QtWidgets.QWidget(); self._central.setLayout(root)
        self.setCentralWidget(self._central)

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

    def _has_inputs(self) -> bool:
        return (self._before and self._after) if self._is_pairs() else bool(self._refs)

    def _make_fitter(self, fitter, method, space, tone):
        if fitter == "Mid":
            return MidFitter(tone_strength=tone)
        return RichFitter(tone_strength=tone, space=space, method=method)

    # — building the look (pure; safe to run off the UI thread) —
    def _build_look(self, snap, report) -> np.ndarray | None:
        fitter = self._make_fitter(snap["fitter"], snap["method"], snap["space"], snap["tone"])

        def _stats(paths, lo, hi):
            stats = []
            for i, p in enumerate(paths):
                stats.append(compute_stats(load_image(p)))
                report(lo + int((hi - lo) * (i + 1) / len(paths)))
            return stats

        if snap["pairs"]:
            if not snap["before"] or not snap["after"]:
                return None
            consensus = build_consensus(_stats(snap["after"], 5, 30))
            src = np.concatenate([load_image(p).reshape(-1, 3) for p in snap["before"]])
            if src.shape[0] > 200_000:
                src = src[np.random.default_rng(0).choice(src.shape[0], 200_000, replace=False)]
            report(40)
            look = fitter.fit(consensus, source_samples=src)
        else:
            if not snap["refs"]:
                return None
            consensus = build_consensus(_stats(snap["refs"], 5, 40))
            look = fitter.fit(consensus)
        report(60)
        return look(self._base)

    def _placement_key(self) -> str:
        return "between" if self._placement.currentIndex() == 1 else "node2"

    def _final_samples(self) -> np.ndarray:
        if self._look_samples is None:
            return self._base
        return _assemble(self._look_samples, self._strength_value(), self._placement_key(), 33)

    def _snapshot(self, refit: bool) -> dict:
        return dict(
            refit=refit, pairs=self._is_pairs(), refs=list(self._refs),
            before=list(self._before), after=list(self._after),
            fitter=self._fitter.currentText(), method=self._method.currentText(),
            space=self._space.currentText(), tone=self._tone_value(),
            strength=self._strength_value(), still=self._still, placement=self._placement_key(),
            still_dirty=self._still_dirty, look=self._look_samples,
        )

    # — the worker payload (runs in _ComputeThread) —
    def _compute(self, snap, report):
        report(1)
        look = self._build_look(snap, report) if snap["refit"] else snap["look"]
        report(66)
        final = self._base if look is None else _assemble(look, snap["strength"], snap["placement"], 33)
        report(70)
        before = None
        if snap["still_dirty"]:
            before = apply_cube(snap["still"], self._base, progress=lambda f: report(70 + int(10 * f)))
            after = self._apply_final(snap, final, look, report, 80, 99)
        else:
            after = self._apply_final(snap, final, look, report, 70, 99)
        report(100)
        return look, before, after, snap["refit"]

    def _apply_final(self, snap, final, look, report, lo, hi):
        """Apply the final cube to the preview still. For 'between' placement the cube is
        DWG/DI→DWG/DI, so Node 2 (base) is applied after to show the Rec.709 result."""
        if snap["placement"] == "between" and look is not None:
            mid = (lo + hi) // 2
            looked = apply_cube(snap["still"], final, progress=lambda f: report(lo + int((mid - lo) * f)))
            return apply_cube(looked, self._base, progress=lambda f: report(mid + int((hi - mid) * f)))
        return apply_cube(snap["still"], final, progress=lambda f: report(lo + int((hi - lo) * f)))

    # — manual compute (Compute button only) —
    def _set_busy(self, on: bool) -> None:
        if on:
            self._busy.setValue(0)
        self._busy.setVisible(on); self._busy_lbl.setVisible(on)

    def _set_controls_enabled(self, on: bool) -> None:
        for cls in (QtWidgets.QComboBox, QtWidgets.QSlider, QtWidgets.QPushButton, QtWidgets.QListWidget):
            for w in self._central.findChildren(cls):
                w.setEnabled(on)
        if on:
            self._sync_controls()                 # restore method/space enabled-state

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._compute_btn.setText("Compute preview ●")   # ● = changes pending

    def _launch_compute(self, _=None, *, refit: bool = True) -> None:
        """Compute on a worker thread. ``refit`` True (Compute button) rebuilds the look; False
        (placement switch / still load) only re-assembles + re-applies the cached look to the still."""
        if self._thread is not None and self._thread.isRunning():
            return
        if not refit and self._look_samples is None and self._dirty:
            return                                # nothing cached + pending → wait for Compute
        snap = self._snapshot(refit=refit)
        self._still_dirty = False
        self._compute_btn.setText("Computing…")
        self._set_controls_enabled(False)         # gray everything out
        self._set_busy(True)
        self._thread = _ComputeThread(lambda report: self._compute(snap, report))
        self._thread.progress.connect(self._busy.setValue)
        self._thread.done.connect(self._on_computed)
        self._thread.start()

    def _refresh_preview(self) -> None:
        """Re-render the preview for a preview-only change (placement, still) — no look rebuild."""
        self._launch_compute(refit=False)

    def _on_computed(self, result) -> None:
        self._set_busy(False)
        self._set_controls_enabled(True)
        if isinstance(result, Exception):
            self._compute_btn.setText("Compute preview ●" if self._dirty else "Compute preview")
            QtWidgets.QMessageBox.warning(self, "LookForge", f"Could not build look:\n{result}")
            return
        look, before, after, was_refit = result
        self._look_samples = look
        if was_refit:
            self._dirty = False                   # look is now current
        self._compute_btn.setText("Compute preview ●" if self._dirty else "Compute preview")
        if before is not None:
            self._before_img = before
        self._show(self._before_lbl, self._before_img)
        self._show(self._after_lbl, after)

    def _show(self, lbl, img) -> None:
        lbl.setPixmap(_to_pixmap(img).scaledToWidth(
            _PREVIEW_W, QtCore.Qt.TransformationMode.SmoothTransformation))

    def _update_preview(self) -> None:   # initial synchronous render (synthetic still is small)
        self._show(self._before_lbl, self._before_img)
        self._show(self._after_lbl, apply_cube(self._still, self._final_samples()))

    def _sync_controls(self) -> None:
        rich = self._fitter.currentText() == "Rich"
        self._pages.setCurrentIndex(1 if self._is_pairs() else 0)
        self._method.setEnabled(rich)
        self._space.setEnabled(rich)

    # — slots (no auto-compute; everything just marks "changes pending") —
    def _on_mode(self, _=None) -> None:
        self._sync_controls(); self._mark_dirty()

    def _on_fitter_changed(self, _=None) -> None:
        self._sync_controls(); self._mark_dirty()

    def _on_placement(self, _=None) -> None:
        # placement is preview-only (cached look); refresh now unless the look itself is stale.
        if self._look_samples is not None and not self._dirty:
            self._refresh_preview()
        else:
            self._mark_dirty()

    def _on_tone(self, _=None) -> None:
        self._tone_lbl.setText(f"Tone (exposure match): {self._tone_value():.2f}")
        self._mark_dirty()

    def _on_strength(self, _=None) -> None:
        self._strength_lbl.setText(f"Strength: {self._strength_value():.2f}")
        self._mark_dirty()

    def _add(self, title, store, listw) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, "", _IMG_FILTER)
        new = [p for p in paths if p not in store]   # dedup: skip already-added files
        if new:
            store.extend(new)
            listw.addItems(new)
            self._mark_dirty()

    def _add_refs(self) -> None:
        self._add("Add references", self._refs, self._refs_list)

    def _add_before(self) -> None:
        self._add("Add NEUTRAL (your footage) frames", self._before, self._before_list)

    def _add_after(self) -> None:
        self._add("Add GRADED (the look) frames", self._after, self._after_list)

    def _remove(self, store, listw) -> None:
        for item in listw.selectedItems():
            if item.text() in store:
                store.remove(item.text())
            listw.takeItem(listw.row(item))
        self._mark_dirty()

    def _remove_refs(self) -> None:
        self._remove(self._refs, self._refs_list)

    def _load_still(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load DWG/DI preview still", "", _IMG_FILTER)
        if path:
            self._still = load_preview_still(path)   # full resolution (see preview.load_preview_still)
            self._still_dirty = True                 # "before" recomputed now
            self._refresh_preview()                  # show the loaded still immediately

    def _export(self) -> None:
        if self._look_samples is None and self._has_inputs():
            try:                                    # ensure the look is current before export
                self._set_busy(True); QtWidgets.QApplication.processEvents()
                self._look_samples = self._build_look(self._snapshot(True), lambda _p: None)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "LookForge", f"Could not build look:\n{exc}")
            finally:
                self._set_busy(False)
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

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)
        super().closeEvent(event)

    def _save_preset(self) -> None:
        if self._is_pairs():
            QtWidgets.QMessageBox.information(self, "LookForge", "Presets apply to References mode.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "look.json", "JSON (*.json)")
        if path:
            save_preset(path, self._refs, self._strength_value(), title="LookForge")
