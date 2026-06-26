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

from lutgen.engine.adjust import Adjustments, apply_adjustments
from lutgen.engine.apply import apply_cube
from lutgen.engine.base import DEFAULT_SIZE, load_base
from lutgen.engine.film import FilmStock, apply_film
from lutgen.engine.cube_io import write_cube
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.consensus import build_consensus
from lutgen.orchestration.ingest import load_references
from lutgen.orchestration.pipeline import _assemble
from lutgen.orchestration.stats import compute_stats_batch

from .preview import load_preview_still, make_test_still

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.exr)"
_PREVIEW_W = 520
_DEBOUNCE_MS = 2000   # wait this long after the LAST slider/control change, then compute


_DITHER = np.random.default_rng(0).uniform(-0.5, 0.5, (2160, 3840, 1))  # fixed triangular-ish dither


def _to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    # The preview is 8-bit (Qt displays 8-bit), so smooth gradients band on screen even though the
    # exported 65-point cube is smooth. Add ±0.5-level dither before quantizing to hide the banding
    # — purely cosmetic, makes the preview match how Resolve renders the cube on real footage.
    img = np.clip(img, 0.0, 1.0) * 255.0
    h, w = img.shape[:2]
    if h <= _DITHER.shape[0] and w <= _DITHER.shape[1]:
        img = img + _DITHER[:h, :w]
    arr = np.ascontiguousarray(np.clip(np.round(img), 0, 255).astype(np.uint8))
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
        self.setWindowTitle("ReinaLook")
        self._base = load_base()
        self._still = make_test_still()
        self._before_img = apply_cube(self._still, self._base)   # preview at strength 0
        self._prev_look_img = self._before_img                   # preview at strength 1 (no look yet)
        self._look_samples: np.ndarray | None = None
        self._adj = Adjustments()                                # creative adjustments (manual grade)
        self._film = FilmStock()                                 # film-stock transfer
        self._before: list[str] = []
        self._after: list[str] = []

        # nothing computes automatically — only the Compute button triggers a (threaded) compute.
        self._still_dirty = True        # recompute "before" on the first/next compute
        self._dirty = False             # settings changed since last compute
        self._thread: _ComputeThread | None = None

        # debounce the (heavier) preview-endpoint rebuild used by adjustments/placement/still —
        # rebuilding the look cube per slider tick lagged; fire ~180ms after the last change.
        self._preview_timer = QtCore.QTimer(self, singleShot=True, interval=180)
        self._preview_timer.timeout.connect(self._do_refresh)
        self._base_timer = QtCore.QTimer(self, singleShot=True, interval=250)   # debounce PFE rebuild
        self._base_timer.timeout.connect(self._rebuild_base)

        self._build_ui()
        self._sync_mode_labels()
        self._update_preview()
        # (no background colour-import warmup — threading a heavy frozen import at startup races with
        # Qt init and can crash the exe. First PFE load pays the ~1-2s import once, under a wait cursor.)

    # — UI —
    def _build_ui(self) -> None:
        # Single mode: Before/After (neutral footage → graded reference look).

        self._before_list = _file_list()
        self._after_list = _file_list()
        self._before_lbl_w = QtWidgets.QLabel()
        self._after_lbl_w = QtWidgets.QLabel()
        self._before_add = QtWidgets.QPushButton()
        self._after_add = QtWidgets.QPushButton()
        self._before_rm = QtWidgets.QPushButton("Remove selected (top list)")
        after_rm = QtWidgets.QPushButton("Remove selected (bottom list)")
        self._before_add.clicked.connect(self._add_before)
        self._after_add.clicked.connect(self._add_after)
        self._before_rm.clicked.connect(self._remove_before)
        after_rm.clicked.connect(self._remove_after)
        pairs_page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pairs_page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self._before_lbl_w)
        pl.addWidget(self._before_list)
        pl.addWidget(self._before_add)
        pl.addWidget(self._before_rm)
        pl.addWidget(self._after_lbl_w)
        pl.addWidget(self._after_list)
        pl.addWidget(self._after_add)
        pl.addWidget(after_rm)
        self._mode_hint = QtWidgets.QLabel()
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setStyleSheet("color: gray;")
        pl.addWidget(self._mode_hint)

        self._pairs_page = pairs_page

        # look engine is fixed: Rich / pdf / Oklab (the best, only combination).
        self._placement = QtWidgets.QComboBox()
        self._placement.addItems(["Replace CSTout", "Between CSTs"])
        self._placement.currentIndexChanged.connect(self._on_placement)

        # conversion: DaVinci CST (default base) or a film-print PFE LUT (DWG/DI→Cineon→PFE→Rec.709)
        self._conversion = QtWidgets.QComboBox()
        self._conversion.addItems(["DaVinci CST (Resolve)", "Film print (PFE LUT)"])
        self._conversion.currentIndexChanged.connect(self._on_conversion)
        self._pfe_btn = QtWidgets.QPushButton("Load PFE .cube…")
        self._pfe_btn.clicked.connect(self._load_pfe)
        self._pfe_btn.setEnabled(False)
        self._pfe_path: str | None = None
        self._film_exposure = self._slider(50, self._on_film_exposure)   # 0..100 → −2..+2 stops
        self._film_exposure.setEnabled(False)
        self._film_exposure_lbl = QtWidgets.QLabel("Film exposure: 0.0 stop")

        self._tone = self._slider(100, self._on_tone)
        self._tone_lbl = QtWidgets.QLabel("Tone (exposure match): 1.00")
        self._strength = self._slider(80, self._on_strength)
        self._strength_lbl = QtWidgets.QLabel("Strength: 0.80")

        form = QtWidgets.QFormLayout()
        form.addRow("Conversion", self._conversion)
        form.addRow("", self._pfe_btn)
        form.addRow("Placement", self._placement)

        film_box = self._build_film_panel()
        adjust_box = self._build_adjust_panel()

        self._compute_btn = QtWidgets.QPushButton("Compute preview")
        self._compute_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._compute_btn.clicked.connect(self._launch_compute)
        export_btn = QtWidgets.QPushButton("Export .cube…")
        export_btn.clicked.connect(self._export)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._pairs_page)
        left.addSpacing(8)
        left.addLayout(form)
        left.addWidget(self._film_exposure_lbl)
        left.addWidget(self._film_exposure)
        left.addWidget(self._tone_lbl)
        left.addWidget(self._tone)
        left.addWidget(self._strength_lbl)
        left.addWidget(self._strength)
        left.addWidget(film_box)
        left.addWidget(adjust_box)
        left.addStretch(1)
        left.addWidget(self._compute_btn)
        left.addWidget(export_btn)

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
        lw = QtWidgets.QWidget(); lw.setLayout(left)
        scroll = QtWidgets.QScrollArea()         # left controls scroll if taller than the window
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360); scroll.setMaximumWidth(380)
        root.addWidget(scroll)
        root.addLayout(preview, 1)
        self._central = QtWidgets.QWidget(); self._central.setLayout(root)
        self.setCentralWidget(self._central)

    def _slider(self, value, slot):
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(0, 100); s.setValue(value); s.valueChanged.connect(slot)
        return s

    # — creative adjustments panel (manual grade; works with or without references) —
    _ADJ_SPECS = [
        ("contrast", "Contrast", -100, 100, 0),
        ("saturation", "Saturation", -100, 100, 0),
        ("temperature", "Temperature", -100, 100, 0),
        ("tint", "Tint", -100, 100, 0),
        ("shadows", "Shadows", -100, 100, 0),
        ("highlights", "Highlights", -100, 100, 0),
        ("highlight_rolloff", "Highlight roll-off", 0, 100, 0),
    ]

    def _build_adjust_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Adjustments (manual grade — optional)")
        box.setCheckable(True)
        box.setChecked(False)                         # collapsed/off by default
        v = QtWidgets.QVBoxLayout(box)
        self._adj_sliders = {}
        for field, label, lo, hi, default in self._ADJ_SPECS:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label); lbl.setMinimumWidth(110)
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(lo, hi); sl.setValue(default)
            sl.valueChanged.connect(self._on_adjust)
            row.addWidget(lbl); row.addWidget(sl, 1)
            v.addLayout(row)
            self._adj_sliders[field] = sl
        reset = QtWidgets.QPushButton("Reset adjustments")
        reset.clicked.connect(self._reset_adjust)
        v.addWidget(reset)
        box.toggled.connect(self._on_adjust)          # toggling on/off updates the look
        self._adjust_box = box
        return box

    def _read_adjust(self) -> Adjustments:
        if not self._adjust_box.isChecked():
            return Adjustments()                      # panel off → no adjustments
        return Adjustments(**{f: s.value() / 100.0 for f, s in self._adj_sliders.items()})

    def _on_adjust(self, _=None) -> None:
        self._adj = self._read_adjust()
        self._refresh_preview()                       # cheap: re-derive endpoints + render

    def _reset_adjust(self) -> None:
        for s in self._adj_sliders.values():
            s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        self._on_adjust()

    # — film-stock transfer panel (reshapes the colour science; works with or without a look) —
    _FILM_SPECS = [
        ("contrast", "Contrast (S-curve)", -100, 100, 0),
        ("toe", "Toe (matte blacks)", 0, 100, 0),
        ("shoulder", "Shoulder (roll-off)", 0, 100, 0),
        ("highlight_bleach", "Highlight bleach", 0, 100, 0),
        ("split_warm", "Split-tone (warm/cool)", -100, 100, 0),
        ("saturation", "Saturation", -100, 100, 0),
    ]

    def _build_film_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Film stock (transfer — optional)")
        box.setCheckable(True); box.setChecked(False)
        v = QtWidgets.QVBoxLayout(box)
        self._film_sliders = {}
        for field, label, lo, hi, default in self._FILM_SPECS:
            row = QtWidgets.QHBoxLayout()
            w = QtWidgets.QLabel(label); w.setMinimumWidth(130)
            sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sl.setRange(lo, hi); sl.setValue(default)
            sl.valueChanged.connect(self._on_film)
            row.addWidget(w); row.addWidget(sl, 1)
            v.addLayout(row)
            self._film_sliders[field] = sl
        reset = QtWidgets.QPushButton("Reset film")
        reset.clicked.connect(self._reset_film)
        v.addWidget(reset)
        box.toggled.connect(self._on_film)
        self._film_box = box
        return box

    def _read_film(self) -> FilmStock:
        if not self._film_box.isChecked():
            return FilmStock()
        return FilmStock(**{f: s.value() / 100.0 for f, s in self._film_sliders.items()})

    def _on_film(self, _=None) -> None:
        self._film = self._read_film()
        self._refresh_preview()

    def _reset_film(self) -> None:
        for s in self._film_sliders.values():
            s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        self._on_film()

    # — state —
    def _tone_value(self) -> float:
        return self._tone.value() / 100.0

    def _strength_value(self) -> float:
        return self._strength.value() / 100.0

    def _has_inputs(self) -> bool:
        return bool(self._before and self._after)

    # — building the look (pure; safe to run off the UI thread) —
    def _build_look(self, snap, report) -> np.ndarray | None:
        # Before/After: transport the neutral pool → graded pool, baked through a gamut-aware
        # bounded grade cube (ADR-0023), applied to the base. The foundation look.
        if not snap["before"] or not snap["after"]:
            return None
        from lutgen.fitter._gradecube import learn_grade_cube_bounded
        report(5)
        consensus = build_consensus(compute_stats_batch(load_references(snap["after"])))
        report(30)
        src = np.concatenate([i.reshape(-1, 3) for i in load_references(snap["before"])])
        if src.shape[0] > 200_000:
            src = src[np.random.default_rng(0).choice(src.shape[0], 200_000, replace=False)]
        look = RichFitter(tone_strength=snap["tone"]).fit(consensus, source_samples=src)
        moved = look(src)
        report(55)
        grade = learn_grade_cube_bounded(src, moved, DEFAULT_SIZE, smoothing=0.025)
        return apply_cube(self._base, grade, DEFAULT_SIZE)

    def _placement_key(self) -> str:
        return "between" if self._placement.currentIndex() == 1 else "node2"

    def _looked(self) -> np.ndarray | None:
        """Full-strength looked base incl. film transfer + creative adjustments. None = pure base."""
        src = self._look_samples
        if self._film.is_identity() and self._adj.is_identity():
            return src                                   # fitter look, or None (no inputs)
        out = src if src is not None else self._base
        out = apply_film(out, self._film)                # film/adjustments work with or without a look
        return apply_adjustments(out, self._adj)

    def _final_at(self, strength: float) -> np.ndarray:
        looked = self._looked()
        if looked is None:
            return self._base
        return _assemble(looked, strength, self._placement_key(), DEFAULT_SIZE, base=self._base)

    def _final_samples(self) -> np.ndarray:   # exact cube at the current strength (for export)
        return self._final_at(self._strength_value())

    def _snapshot(self, refit: bool) -> dict:
        return dict(
            refit=refit,
            before=list(self._before), after=list(self._after),
            tone=self._tone_value(),
            strength=self._strength_value(), still=self._still, placement=self._placement_key(),
            still_dirty=self._still_dirty, look=self._look_samples,
        )

    # — the worker payload (runs in _ComputeThread): builds the heavy LOOK only —
    def _compute(self, snap, report):
        report(5)
        look = self._build_look(snap, report)   # the only expensive step (stats + fit)
        report(100)
        return look

    # — preview cache: precompute the still at strength 0 and 1, so the strength slider is a
    #   cheap image lerp (trilinear apply is linear in the cube → exact for 'between', a close
    #   approximation under node2's gamut clamp; export always uses the exact _final_samples). —
    def _apply_to_still(self, final):
        """Apply a cube to the still. For 'between' the cube is DWG/DI→DWG/DI, so apply Node 2
        (base) after to show the Rec.709 result."""
        looked = apply_cube(self._still, final)
        has_look = (self._look_samples is not None or not self._adj.is_identity()
                    or not self._film.is_identity())
        if self._placement_key() == "between" and has_look:
            looked = apply_cube(looked, self._base)
        return looked

    def _rebuild_preview_cache(self) -> None:
        """Recompute the two endpoint preview images (heavy). Call when look / placement / still
        changes — NOT on every strength tick."""
        self._before_img = apply_cube(self._still, self._base)          # strength 0
        self._prev_look_img = self._apply_to_still(self._final_at(1.0))  # strength 1

    def _render_preview(self) -> None:
        s = self._strength_value()
        after = (1.0 - s) * self._before_img + s * self._prev_look_img   # cheap lerp — instant
        self._show(self._before_lbl, self._before_img)
        self._show(self._after_lbl, after)

    # — manual compute (Compute button only) —
    def _set_busy(self, on: bool) -> None:
        if on:
            self._busy.setValue(0)
        self._busy.setVisible(on); self._busy_lbl.setVisible(on)

    def _set_controls_enabled(self, on: bool) -> None:
        for cls in (QtWidgets.QComboBox, QtWidgets.QSlider, QtWidgets.QPushButton, QtWidgets.QListWidget):
            for w in self._central.findChildren(cls):
                w.setEnabled(on)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._compute_btn.setText("Compute preview ●")   # ● = changes pending

    def _launch_compute(self, _=None) -> None:
        """Compute button: rebuild the heavy LOOK on a worker thread (spinner + gray-out)."""
        if self._thread is not None and self._thread.isRunning():
            return
        snap = self._snapshot(refit=True)
        self._compute_btn.setText("Computing…")
        self._set_controls_enabled(False)         # gray everything out
        self._set_busy(True)
        self._thread = _ComputeThread(lambda report: self._compute(snap, report))
        self._thread.progress.connect(self._busy.setValue)
        self._thread.done.connect(self._on_computed)
        self._thread.start()

    def _refresh_preview(self) -> None:
        """Preview-only change that alters the look image (adjustments / placement / still). Rebuilds
        the cached endpoints — debounced, since that re-derives the look cube (heavy for 'between')."""
        self._preview_timer.start()

    def _do_refresh(self) -> None:
        self._rebuild_preview_cache()
        self._render_preview()

    def _on_computed(self, result) -> None:
        self._set_busy(False)
        self._set_controls_enabled(True)
        if isinstance(result, Exception):
            self._compute_btn.setText("Compute preview ●" if self._dirty else "Compute preview")
            QtWidgets.QMessageBox.warning(self, "ReinaLook", f"Could not build look:\n{result}")
            return
        self._look_samples = result
        self._dirty = False                       # look is now current
        self._compute_btn.setText("Compute preview")
        self._rebuild_preview_cache()             # new look → recompute endpoints (once)
        self._render_preview()

    def _show(self, lbl, img) -> None:
        lbl.setPixmap(_to_pixmap(img).scaledToWidth(
            _PREVIEW_W, QtCore.Qt.TransformationMode.SmoothTransformation))

    def _update_preview(self) -> None:
        self._rebuild_preview_cache()
        self._render_preview()

    # — slots (no auto-compute; everything just marks "changes pending") —
    def _sync_mode_labels(self) -> None:
        self._before_lbl_w.setText("NEUTRAL frames (your ungraded footage)")
        self._after_lbl_w.setText("GRADED reference (the film/look)")
        self._before_add.setText("+ Add NEUTRAL (your footage)…")
        self._after_add.setText("+ Add GRADED (the film look)…")
        self._mode_hint.setText("Before/After foundation: add NEUTRAL frames (your footage) + GRADED "
                                "frames (the film look). Builds one cube — your foundation. Apply to "
                                "all clips, then tweak each. Neutral should represent your footage.")

    # — conversion: DaVinci base vs film-print PFE base —
    def _film_exposure_value(self) -> float:
        return (self._film_exposure.value() - 50) / 25.0   # 0..100 → −2..+2 stops

    def _is_film_print(self) -> bool:
        return self._conversion.currentIndex() == 1 and self._pfe_path is not None

    def _rebuild_base(self) -> None:
        """Set self._base to the DaVinci base or the film-print PFE base, then recompute the preview
        'before' (base-converted still). Any look must be recomputed (base changed) → mark dirty.
        Building the film base applies the PFE to the 65-point grid (~0.5s) — show a wait cursor."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if self._is_film_print():
                from lutgen.engine.filmprint import build_film_base
                try:
                    self._base = build_film_base(self._pfe_path, DEFAULT_SIZE, self._film_exposure_value())
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "ReinaLook", f"Could not build film base:\n{exc}")
                    return
            else:
                self._base = load_base()
            self._before_img = apply_cube(self._still, self._base)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._mark_dirty()                        # look is fit on the base → rebuild on Compute
        self._refresh_preview()

    def _on_conversion(self, _=None) -> None:
        film = self._conversion.currentIndex() == 1
        self._pfe_btn.setEnabled(film)
        self._film_exposure.setEnabled(film)
        # film print is a Replace-CSTout conversion; Between uses the DaVinci inverse (force node2)
        self._placement.setEnabled(not film)
        if film:
            self._placement.setCurrentIndex(0)
        self._rebuild_base()

    def _load_pfe(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load PFE .cube (Cineon-input)", "", "Cube (*.cube)")
        if path:
            self._pfe_path = path
            self._rebuild_base()

    def _on_film_exposure(self, _=None) -> None:
        self._film_exposure_lbl.setText(f"Film exposure: {self._film_exposure_value():+.1f} stop")
        if self._is_film_print():
            self._base_timer.start()              # debounced: rebuild once dragging stops

    def _on_placement(self, _=None) -> None:
        self._refresh_preview()                   # changes the look image → rebuild endpoints (once)

    def _on_tone(self, _=None) -> None:
        self._tone_lbl.setText(f"Tone (exposure match): {self._tone_value():.2f}")
        self._mark_dirty()                        # tone is inside the fitter → needs Compute

    def _on_strength(self, _=None) -> None:
        self._strength_lbl.setText(f"Strength: {self._strength_value():.2f}")
        self._render_preview()                    # cheap lerp between cached endpoints — instant

    def _add(self, title, store, listw) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, "", _IMG_FILTER)
        new = [p for p in paths if p not in store]   # dedup: skip already-added files
        if new:
            store.extend(new)
            listw.addItems(new)
            self._mark_dirty()

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

    def _remove_before(self, _=None) -> None:
        self._remove(self._before, self._before_list)

    def _remove_after(self, _=None) -> None:
        self._remove(self._after, self._after_list)

    def _load_still(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load DWG/DI preview still", "", _IMG_FILTER)
        if path:
            self._still = load_preview_still(path)
            self._refresh_preview()                  # rebuild endpoints for the new still + render

    def _export(self) -> None:
        if self._look_samples is None and self._has_inputs():
            try:                                    # ensure the look is current before export
                self._set_busy(True); QtWidgets.QApplication.processEvents()
                self._look_samples = self._build_look(self._snapshot(True), lambda _p: None)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "ReinaLook", f"Could not build look:\n{exc}")
            finally:
                self._set_busy(False)
        if self._look_samples is None and self._adj.is_identity() and self._film.is_identity():
            msg = (f"Add images first — now {len(self._before)} / {len(self._after)} "
                   f"(need at least one of each), or open Film stock / Adjustments for a manual grade.")
            QtWidgets.QMessageBox.warning(self, "ReinaLook", msg)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export .cube", "look.cube", "Cube (*.cube)")
        if path:
            write_cube(path, self._final_samples(), title="ReinaLook")
            QtWidgets.QMessageBox.information(self, "ReinaLook", f"Wrote {path}")

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)
        super().closeEvent(event)

