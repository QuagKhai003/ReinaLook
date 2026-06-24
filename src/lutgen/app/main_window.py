"""main_window — the PySide6 desktop shell (thin; all logic is in the engine).

@context  Wires widgets to the tested pipeline: references list, strength slider, before/after
          preview, export, save preset. No color math here (Plan §3, ADR-0007).
@done     MainWindow with add/remove refs, live preview, export, save preset.
@todo     Thumbnails, drag-drop polish, packaging (PyInstaller).
@limits   Imports PySide6 (only when the GUI runs). Look refit on ref change; re-blend on strength.
@affects  Uses orchestration.pipeline/preset, engine.base/strength/regularize, app.preview.
          Launched by app/run.py. See ADR-0007.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from lutgen.engine.base import load_base
from lutgen.engine.cube_io import write_cube
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.consensus import build_consensus
from lutgen.orchestration.ingest import load_references
from lutgen.orchestration.preset import save_preset
from lutgen.orchestration.stats import compute_stats

from .preview import before_after, load_preview_still, make_test_still

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.exr)"
_PREVIEW_W = 460  # displayed preview width (px)


def _to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    arr = np.ascontiguousarray(np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w, _ = arr.shape
    qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LookForge")
        self._refs: list[str] = []
        self._base = load_base()
        self._still = make_test_still()
        self._look_samples: np.ndarray | None = None

        self._refs_list = QtWidgets.QListWidget()
        add_btn = QtWidgets.QPushButton("+ Add references…")
        rm_btn = QtWidgets.QPushButton("Remove selected")
        add_btn.clicked.connect(self._add_refs)
        rm_btn.clicked.connect(self._remove_selected)

        self._strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._strength.setRange(0, 100)
        self._strength.setValue(80)
        self._strength_lbl = QtWidgets.QLabel("Strength: 0.80")
        self._strength.valueChanged.connect(self._on_strength)

        export_btn = QtWidgets.QPushButton("Export .cube…")
        preset_btn = QtWidgets.QPushButton("Save preset…")
        export_btn.clicked.connect(self._export)
        preset_btn.clicked.connect(self._save_preset)

        self._before = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self._after = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        still_btn = QtWidgets.QPushButton("Load preview still (DWG/DI frame)…")
        still_btn.clicked.connect(self._load_still)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("References"))
        left.addWidget(self._refs_list, 1)
        left.addWidget(add_btn)
        left.addWidget(rm_btn)
        left.addSpacing(12)
        left.addWidget(self._strength_lbl)
        left.addWidget(self._strength)
        left.addStretch(1)
        left.addWidget(export_btn)
        left.addWidget(preset_btn)

        preview = QtWidgets.QVBoxLayout()
        preview.addWidget(still_btn)
        preview.addWidget(QtWidgets.QLabel("Before (base)"))
        preview.addWidget(self._before, 1)
        preview.addWidget(QtWidgets.QLabel("After (look)"))
        preview.addWidget(self._after, 1)

        root = QtWidgets.QHBoxLayout()
        root.addLayout(left, 0)
        root.addLayout(preview, 1)
        central = QtWidgets.QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._update_preview()

    # — state —
    def _strength_value(self) -> float:
        return self._strength.value() / 100.0

    def _refit(self) -> None:
        if not self._refs:
            self._look_samples = None
            return
        images = load_references(self._refs)
        consensus = build_consensus([compute_stats(i) for i in images])
        self._look_samples = RichFitter().fit(consensus)(self._base)

    def _final_samples(self) -> np.ndarray:
        if self._look_samples is None:
            return self._base
        return regularize(blend(self._base, self._look_samples, self._strength_value()))

    def _update_preview(self) -> None:
        before, after = before_after(self._still, self._base, self._final_samples())
        self._before.setPixmap(_to_pixmap(before).scaledToWidth(
            _PREVIEW_W, QtCore.Qt.TransformationMode.SmoothTransformation))
        self._after.setPixmap(_to_pixmap(after).scaledToWidth(
            _PREVIEW_W, QtCore.Qt.TransformationMode.SmoothTransformation))

    # — slots —
    def _add_refs(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add references", "", _IMG_FILTER)
        for p in paths:
            self._refs.append(p)
            self._refs_list.addItem(p)
        if paths:
            self._refit()
            self._update_preview()

    def _remove_selected(self) -> None:
        for item in self._refs_list.selectedItems():
            self._refs.remove(item.text())
            self._refs_list.takeItem(self._refs_list.row(item))
        self._refit()
        self._update_preview()

    def _on_strength(self, value: int) -> None:
        self._strength_lbl.setText(f"Strength: {value / 100.0:.2f}")
        self._update_preview()

    def _load_still(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load DWG/DI preview still", "", _IMG_FILTER)
        if path:
            self._still = load_preview_still(path)
            self._update_preview()

    def _export(self) -> None:
        if not self._refs:
            QtWidgets.QMessageBox.warning(self, "LookForge", "Add references first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export .cube", "look.cube", "Cube (*.cube)")
        if path:
            write_cube(path, self._final_samples(), title="LookForge")
            QtWidgets.QMessageBox.information(self, "LookForge", f"Wrote {path}")

    def _save_preset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save preset", "look.json", "JSON (*.json)")
        if path:
            save_preset(path, self._refs, self._strength_value(), title="LookForge")
