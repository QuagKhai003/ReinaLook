"""_gradecube — fit a smooth 3D grade cube from scattered (before -> after) color samples.

@context  Shared by PairsFitter (ADR-0012, learn from frame pairs) and Rich PDF transfer
          (ADR-0013, fit a continuous cube from IDT-transported points). Trilinear splat into the
          grid, normalize, nearest-fill unsampled colors, Gaussian-smooth (OT regularization).
@done     learn_grade_cube(before, after, size, smoothing, min_weight).
@limits   PURE numeric. Inputs (...,3) [0,1]; output flat (size**3, 3) red-fastest.
@affects  Used by fitter/pairs.py + fitter/rich.py. See ADR-0012/0013.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter

from lutgen.engine.apply import apply_cube
from lutgen.engine.grid import identity_grid


class CubeLookTransform:
    """Callable neutral_rgb -> looked_rgb via trilinear sample of a learned grade cube."""

    def __init__(self, grade_cube: np.ndarray, size: int):
        self._cube = grade_cube
        self._size = size

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return apply_cube(rgb, self._cube, self._size)


def learn_grade_cube(before: np.ndarray, after: np.ndarray, size: int,
                     smoothing: float, min_weight: float) -> np.ndarray:
    """Build a ``(size**3, 3)`` grade cube mapping ``before`` colors to ``after`` (red-fastest)."""
    n = size
    b = np.clip(np.asarray(before, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    a = np.asarray(after, dtype=np.float64).reshape(-1, 3)
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
        raise ValueError("no usable samples (check inputs)")
    node = np.empty((n ** 3, 3))
    node[sampled] = acc[sampled] / wsum[sampled, None]

    grid = identity_grid(n)
    empty = ~sampled
    if empty.any():                            # extrapolate the grade to unsampled colors
        node[empty] = NearestNDInterpolator(grid[sampled], node[sampled])(grid[empty])

    if smoothing > 0:                          # regularize: smooth in the 3D color volume.
        # sigma is in node units → scale with size so the smoothing covers the SAME colour distance
        # at any cube resolution (a 33-tuned 0.8 → ~1.57 at 65; without this, 65 smooths half as
        # much and bands/harshens more than 33 did).
        sigma = smoothing * (n / 33.0)
        lat = node.reshape(n, n, n, 3)
        for c in range(3):
            lat[..., c] = gaussian_filter(lat[..., c], sigma=sigma, mode="nearest")
        node = lat.reshape(-1, 3)
    return np.clip(node, 0.0, 1.0)
