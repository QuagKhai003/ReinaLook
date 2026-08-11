"""qt_image — numpy image -> QPixmap with anti-banding dither (shared by all preview panes).

@context  The preview is 8-bit (Qt displays 8-bit), so smooth gradients band on screen even
          though the exported 65-point cube is smooth. A fixed ±0.5-level dither before
          quantizing hides that — cosmetic only, matches how Resolve renders on real footage.
@done     to_pixmap(img) for (H,W,3) float [0,1].
@todo     -
@limits   GUI-only (Qt). Dither buffer covers up to 4K; larger frames convert undithered.
@affects  Used by main_window.py (legacy preview) + apply_tab.py. Split out of main_window
          in ADR-0002 b2.2.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtGui

_DITHER = np.random.default_rng(0).uniform(-0.5, 0.5, (2160, 3840, 1))  # fixed dither field


def to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    """Convert an (H,W,3) float image in [0,1] to a QPixmap (dithered 8-bit)."""
    img = np.clip(img, 0.0, 1.0) * 255.0
    h, w = img.shape[:2]
    if h <= _DITHER.shape[0] and w <= _DITHER.shape[1]:
        img = img + _DITHER[:h, :w]
    arr = np.ascontiguousarray(np.clip(np.round(img), 0, 255).astype(np.uint8))
    qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)
