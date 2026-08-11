"""worker — the one QThread wrapper every heavy GUI computation runs on.

@context  Spec §9: never block the UI thread. One shared thread class instead of a copy per
          tab. The payload gets a ``report(pct)`` callback; a payload may also raise
          Cancelled (cooperatively, when the user hit Cancel) — delivered like any result.
@done     ComputeThread (done/progress signals); Cancelled sentinel exception.
@todo     -
@limits   GUI-only module (imports Qt). The payload runs OFF the UI thread: it must not touch
          widgets — communicate via the signals only.
@affects  Used by main_window.py (legacy compute) + learn_tab.py (staged fit). ADR-0002.
"""

from __future__ import annotations

from PySide6 import QtCore


class Cancelled(Exception):
    """Raised inside a worker payload when the user cancelled — not an error."""


class ComputeThread(QtCore.QThread):
    """Run a heavy payload off the UI thread. ``fn(report)`` → result; exceptions (including
    Cancelled) are delivered through ``done`` instead of crashing the thread."""

    done = QtCore.Signal(object)     # result object or Exception
    progress = QtCore.Signal(int)    # 0..100
    stage = QtCore.Signal(str)       # coarse stage name (staged fits)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(self.progress.emit))
        except Exception as exc:  # noqa: BLE001 — thread boundary: EVERY failure must reach the UI
            self.done.emit(exc)
