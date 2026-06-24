"""run — launch the LookForge desktop app.

@context  Entry point for the GUI (`lutgen-gui` / `python -m lutgen.app`). Imports Qt lazily so
          the core package never depends on PySide6.
@done     main(argv): start QApplication + MainWindow.
@limits   Requires PySide6 (optional 'gui' extra).
@affects  Loads app.main_window. See ADR-0007.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from PySide6 import QtWidgets  # lazy: keep Qt out of the core import path

    from .main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or [])
    window = MainWindow()
    window.resize(1100, 700)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
