"""learn_tab — the Learn mode page: reference pool -> staged fit -> saved Look Profile.

@context  Phase 2's first slice (ADR-0002 b2.1, spec §4/§9): the user adds 5-15 graded frames,
          sees honest frame-count guidance inline (colored, no modal nagging), runs the staged
          fit on a worker thread (per-stage progress + Cancel), reads the learned recipe as
          text, and saves the profile JSON.
@done     LearnTab widget: pool list + add/remove, colored frame_count_hint, draft-fit
          checkbox, threaded learn with stage progress + stage-granular Cancel, recipe
          summary, save-profile dialog.
@todo     Recipe EDITING is 2.3; profile library / Apply side is 2.2.
@limits   GUI-only (Qt); no color math — calls orchestration/learn + profile only. Cancel is
          cooperative at stage boundaries (tone/crosstalk/huesat), granular enough for a
          seconds-long fit.
@affects  Mounted as the "Learn (v2)" tab by main_window.py. Uses app/worker.py + app/recipe.py,
          orchestration/learn.py + profile.py, fitter/fit.FitOptions. ADR-0002 b2.1.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from lutgen.fitter.fit import FitOptions
from lutgen.orchestration.learn import frame_count_hint, learn_profile
from lutgen.orchestration.profile import LookProfile, save_profile

from .recipe import recipe_summary
from .worker import Cancelled, ComputeThread

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
# stage -> progress % shown while that stage runs
_STAGE_PCT = {"tone": 10, "crosstalk": 45, "huesat": 75, "done": 100}
_STAGE_TEXT = {"tone": "Fitting tone…", "crosstalk": "Fitting crosstalk…",
               "huesat": "Fitting hue/sat detail…", "done": "Done"}


def _hint_color(n: int) -> str:
    if n <= 1:
        return "#c33"      # warning: scene contamination
    if n < 5:
        return "#c80"      # usable, add more
    return "#2a2"          # good pool


class LearnTab(QtWidgets.QWidget):
    """Learn mode: pool -> fit -> profile. All heavy work on a ComputeThread."""

    def __init__(self) -> None:
        super().__init__()
        self._paths: list[str] = []
        self._profile: LookProfile | None = None
        self._thread: ComputeThread | None = None
        self._cancel = False
        self._build_ui()
        self._update_hint()

    # — UI —
    def _build_ui(self) -> None:
        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        add_btn = QtWidgets.QPushButton("+ Add graded reference frames…")
        add_btn.clicked.connect(self._add)
        rm_btn = QtWidgets.QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove)

        self._hint = QtWidgets.QLabel()
        self._hint.setWordWrap(True)

        self._fast = QtWidgets.QCheckBox("Draft fit (quicker, rougher)")

        self._learn_btn = QtWidgets.QPushButton("Learn look")
        self._learn_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._learn_btn.clicked.connect(self._launch)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setVisible(False)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._stage_lbl = QtWidgets.QLabel("")
        self._stage_lbl.setVisible(False)

        self._summary = QtWidgets.QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setPlaceholderText("The learned recipe appears here.")

        self._save_btn = QtWidgets.QPushButton("Save profile…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("GRADED reference frames (varied scenes, same look)"))
        lay.addWidget(self._list, 1)
        lay.addWidget(add_btn)
        lay.addWidget(rm_btn)
        lay.addWidget(self._hint)
        lay.addWidget(self._fast)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._learn_btn, 1)
        row.addWidget(self._cancel_btn)
        lay.addLayout(row)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self._stage_lbl)
        prow.addWidget(self._progress, 1)
        lay.addLayout(prow)
        lay.addWidget(QtWidgets.QLabel("Learned recipe"))
        lay.addWidget(self._summary, 1)
        lay.addWidget(self._save_btn)

    # — pool —
    def _add(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add graded reference frames", "", _IMG_FILTER)
        new = [p for p in paths if p not in self._paths]
        if new:
            self._paths.extend(new)
            self._list.addItems(new)
            self._update_hint()

    def _remove(self) -> None:
        for item in self._list.selectedItems():
            if item.text() in self._paths:
                self._paths.remove(item.text())
            self._list.takeItem(self._list.row(item))
        self._update_hint()

    def _update_hint(self) -> None:
        n = len(self._paths)
        self._hint.setText(frame_count_hint(n) if n else
                           "Add graded frames from the film / video whose look you want.")
        self._hint.setStyleSheet(f"color: {_hint_color(n)};" if n else "color: gray;")
        self._learn_btn.setEnabled(n > 0)

    # — learn (threaded) —
    def _launch(self, _=None) -> None:
        if not self._paths or (self._thread is not None and self._thread.isRunning()):
            return
        paths = list(self._paths)
        options = FitOptions(n_samples=1200, max_nfev=30) if self._fast.isChecked() else None
        self._cancel = False
        self._set_running(True)

        thread = ComputeThread(lambda report: self._fit_payload(paths, options))
        thread.stage.connect(self._on_stage)
        thread.done.connect(self._on_done)
        self._thread = thread
        thread.start()

    def _fit_payload(self, paths, options) -> LookProfile:
        """Worker-thread payload. Stage callback also carries the cooperative cancel."""
        def progress(stage: str) -> None:
            if self._cancel:
                raise Cancelled()
            self._thread.stage.emit(stage)
        return learn_profile(paths, name="untitled", options=options, progress=progress)

    def _on_stage(self, stage: str) -> None:
        self._stage_lbl.setText(_STAGE_TEXT.get(stage, stage))
        self._progress.setValue(_STAGE_PCT.get(stage, 0))

    def _on_cancel(self, _=None) -> None:
        self._cancel = True
        self._cancel_btn.setEnabled(False)
        self._stage_lbl.setText("Cancelling…")

    def _set_running(self, on: bool) -> None:
        self._learn_btn.setEnabled(not on and bool(self._paths))
        self._cancel_btn.setVisible(on)
        self._cancel_btn.setEnabled(on)
        self._progress.setVisible(on)
        self._stage_lbl.setVisible(on)
        if on:
            self._progress.setValue(0)
            self._stage_lbl.setText("Starting…")

    def _on_done(self, result) -> None:
        self._set_running(False)
        if isinstance(result, Cancelled):
            self._stage_lbl.setText("")
            return
        if isinstance(result, Exception):
            QtWidgets.QMessageBox.warning(self, "ReinaLook", f"Could not learn the look:\n{result}")
            return
        self._profile = result
        self._summary.setPlainText(recipe_summary(result))
        self._save_btn.setEnabled(True)

    # — save —
    def _save(self, _=None) -> None:
        if self._profile is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Look Profile", "look.json", "Look Profile (*.json)")
        if path:
            from pathlib import Path
            self._profile.name = Path(path).stem
            save_profile(path, self._profile)
            QtWidgets.QMessageBox.information(self, "ReinaLook", f"Saved {path}")
