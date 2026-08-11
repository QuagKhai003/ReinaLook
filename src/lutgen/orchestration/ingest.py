"""ingest — load & normalize reference images (L3 entry, the IO seam).

@context  Turn reference image files into normalized float arrays for stats. The only IO in
          L3; kept behind thin functions so a different decoder can swap in later. Movie
          screenshots often carry LETTERBOX bars; a black bar injects a false black spike
          into every tone statistic and can wreck the Learn fit (ADR-0003), so bars are
          auto-cropped at ingest by default.
@done     load_image (file -> (H,W,3) float64 [0,1], autocrop letterbox); load_references;
          autocrop_letterbox (pure, tested).
@todo     -
@limits   Decodes via Pillow; drops alpha; converts to RGB; optional max-dimension downscale.
          Assumes inputs are already in Rec.709 g2.4 (delivery space) per Plan §1. Autocrop
          is conservative: only clear near-black EDGE runs are cropped (2 px inset), never
          more than 50% of a dimension; dark scenes without bars are untouched.
@affects  Output consumed by orchestration/stats.py + poolstats.py. New dep: Pillow.
          See codemap/INDEX.md + Plan/30_LOOK_FITTER.md §1 + ADR-0004 + ADR-0003.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_LUMA = np.array([0.2126, 0.7152, 0.0722])
_BAR_LUMA = 0.02        # a row/col whose MEAN luma is below this reads as a bar
_BAR_INSET = 2          # extra pixels cropped past the detected bar edge (JPEG halo)
_MAX_CROP_FRAC = 0.5    # never crop more than half of a dimension (safety valve)


def autocrop_letterbox(arr: np.ndarray) -> np.ndarray:
    """Crop black letterbox/pillarbox bars off an ``(H, W, 3)`` [0,1] image (ADR-0003).

    A bar is a run of edge rows/columns whose mean luma is near black. Conservative by
    design: a dark SCENE (low luma but no pure-black edge run) is left untouched, and a
    crop that would remove more than half of a dimension is refused.
    """
    luma = arr @ _LUMA
    h, w = luma.shape

    def _run(means: np.ndarray) -> int:
        bright = means > _BAR_LUMA
        return int(np.argmax(bright)) if bright.any() else 0

    row_means = luma.mean(axis=1)
    col_means = luma.mean(axis=0)
    top, bottom = _run(row_means), _run(row_means[::-1])
    left, right = _run(col_means), _run(col_means[::-1])

    def _bounds(lo: int, hi: int, size: int) -> tuple[int, int]:
        lo = lo + _BAR_INSET if lo else 0            # inset only when a bar was found
        hi = hi + _BAR_INSET if hi else 0
        if lo + hi >= size * _MAX_CROP_FRAC:         # refuse absurd crops
            return 0, size
        return lo, size - hi

    y0, y1 = _bounds(top, bottom, h)
    x0, x1 = _bounds(left, right, w)
    if (y0, y1, x0, x1) == (0, h, 0, w):
        return arr
    return arr[y0:y1, x0:x1]


def load_image(path: str | Path, max_dim: int | None = 1024, *,
               autocrop: bool = True) -> np.ndarray:
    """Load one image as ``(H, W, 3)`` float64 in [0, 1] (RGB, alpha dropped).

    If ``max_dim`` is set, the longer side is downscaled to it (speeds up stats; the global
    color statistics are scale-insensitive). Normalization divides by the dtype max (255 for
    8-bit, 65535 for 16-bit).
    """
    img = Image.open(path)
    img = img.convert("RGB")
    if max_dim is not None and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale)))
        img = img.resize(new_size, Image.BILINEAR)

    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        denom = 65535.0
    else:
        denom = 255.0
    out = arr.astype(np.float64) / denom
    return autocrop_letterbox(out) if autocrop else out


def load_references(paths, max_dim: int | None = 1024, *,
                    autocrop: bool = True) -> list[np.ndarray]:
    """Load multiple reference images (serial — stability over the small decode-parallel win; the
    threaded numpy path was prone to OpenBLAS segfaults). Raises if the list is empty."""
    paths = list(paths)
    if not paths:
        raise ValueError("need at least one reference image")
    return [load_image(p, max_dim=max_dim, autocrop=autocrop) for p in paths]
