"""ingest — load & normalize reference images (L3 entry, the IO seam).

@context  Turn reference image files into normalized float arrays for stats. The only IO in
          L3; kept behind thin functions so a different decoder can swap in later.
@done     load_image (file -> (H,W,3) float64 [0,1]); load_references (many).
@todo     -
@limits   Decodes via Pillow; drops alpha; converts to RGB; optional max-dimension downscale.
          Assumes inputs are already in Rec.709 g2.4 (delivery space) per Plan §1.
@affects  Output consumed by orchestration/stats.py. New dep: Pillow.
          See codemap/INDEX.md + Plan/30_LOOK_FITTER.md §1 + ADR-0004.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, max_dim: int | None = 1024) -> np.ndarray:
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
    return arr.astype(np.float64) / denom


def load_references(paths, max_dim: int | None = 1024) -> list[np.ndarray]:
    """Load multiple reference images (decoded in parallel — Pillow releases the GIL). Order is
    preserved; results are identical to a serial load. Raises if the list is empty."""
    paths = list(paths)
    if not paths:
        raise ValueError("need at least one reference image")
    if len(paths) == 1:
        return [load_image(paths[0], max_dim=max_dim)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as ex:
        return list(ex.map(lambda p: load_image(p, max_dim=max_dim), paths))
