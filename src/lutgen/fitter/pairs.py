"""pairs — learn the exact grade from before/after frame pairs (LUT-from-examples).

@context  The highest-fidelity fitter: given matched frames (before = neutral, after = graded),
          learn the grade as a 3D LUT directly — no content contamination. Splat pairs into the
          grid, fill unsampled nodes, smooth (the mandatory OT regularization). Output is the same
          LookTransform type as Mid/Rich, so it drops into the shared blend/regularize/cube path.
@done     PairsFitter.fit_from_pairs(before, after) -> LookTransform (learned grade cube).
@todo     Per-pixel confidence weighting; gamut-aware extrapolation.
@limits   Pure numeric (no IO). before/after are (H,W,3) [0,1] Rec.709, matched. Out-of-coverage
          colors get the nearest learned grade, then smoothing. Cube ordering = red-fastest.
@affects  Uses engine.apply.apply_cube + engine.grid. Output consumed by pipeline (replace Node 2).
          See ADR-0012 + Plan/30_LOOK_FITTER.md.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter

from lutgen.engine.apply import apply_cube
from lutgen.engine.grid import DEFAULT_SIZE, identity_grid

from .interface import LookTransform


def _learn_grade_cube(before: np.ndarray, after: np.ndarray, size: int,
                      smoothing: float, min_weight: float) -> np.ndarray:
    """Build a (size**3, 3) grade cube mapping `before` colors to `after` colors (red-fastest)."""
    n = size
    b = np.clip(before.reshape(-1, 3), 0.0, 1.0)
    a = after.reshape(-1, 3)
    coords = b * (n - 1)                       # (M,3) in (R,G,B) grid coordinates
    lo = np.floor(coords).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    frac = coords - lo

    nn = n * n
    acc = np.zeros((n ** 3, 3))
    wsum = np.zeros(n ** 3)
    for dr in (0, 1):
        ir = hi[:, 0] if dr else lo[:, 0]
        wr = frac[:, 0] if dr else 1.0 - frac[:, 0]
        for dg in (0, 1):
            ig = hi[:, 1] if dg else lo[:, 1]
            wg = frac[:, 1] if dg else 1.0 - frac[:, 1]
            for db in (0, 1):
                ib = hi[:, 2] if db else lo[:, 2]
                wb = frac[:, 2] if db else 1.0 - frac[:, 2]
                w = wr * wg * wb
                flat = ib * nn + ig * n + ir   # red-fastest flat index, lattice [blue,green,red]
                np.add.at(acc, flat, a * w[:, None])
                np.add.at(wsum, flat, w)

    sampled = wsum > min_weight
    if not sampled.any():
        raise ValueError("no usable pixel pairs (check inputs)")
    node = np.empty((n ** 3, 3))
    node[sampled] = acc[sampled] / wsum[sampled, None]

    grid = identity_grid(n)
    empty = ~sampled
    if empty.any():                            # extrapolate the grade to unsampled colors
        fill = NearestNDInterpolator(grid[sampled], node[sampled])
        node[empty] = fill(grid[empty])

    if smoothing > 0:                          # regularize: smooth in the 3D color volume
        lat = node.reshape(n, n, n, 3)
        for c in range(3):
            lat[..., c] = gaussian_filter(lat[..., c], sigma=smoothing, mode="nearest")
        node = lat.reshape(-1, 3)
    return np.clip(node, 0.0, 1.0)


class _PairsLookTransform:
    """Callable neutral_rgb -> graded_rgb: trilinear sample of the learned grade cube."""

    def __init__(self, grade_cube: np.ndarray, size: int):
        self._cube = grade_cube
        self._size = size

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return apply_cube(rgb, self._cube, self._size)


class PairsFitter:
    """Learn a grade LookTransform from before/after frame pairs (ADR-0012)."""

    def __init__(self, smoothing: float = 0.8, min_weight: float = 1e-3, size: int = DEFAULT_SIZE):
        self._smoothing = float(smoothing)
        self._min_weight = float(min_weight)
        self._size = size

    def fit_from_pairs(self, before_images, after_images) -> LookTransform:
        before_images = list(before_images)
        after_images = list(after_images)
        if len(before_images) != len(after_images) or not before_images:
            raise ValueError("need a matching, non-empty list of before/after images")
        befores, afters = [], []
        for bi, ai in zip(before_images, after_images):
            bi = np.asarray(bi, dtype=np.float64)
            ai = np.asarray(ai, dtype=np.float64)
            if bi.shape != ai.shape:
                raise ValueError(f"pair shape mismatch: {bi.shape} vs {ai.shape}")
            befores.append(bi.reshape(-1, 3))
            afters.append(ai.reshape(-1, 3))
        grade = _learn_grade_cube(
            np.concatenate(befores), np.concatenate(afters),
            self._size, self._smoothing, self._min_weight,
        )
        return _PairsLookTransform(grade, self._size)
