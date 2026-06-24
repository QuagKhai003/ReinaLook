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

from lutgen.engine.grid import DEFAULT_SIZE

from ._gradecube import CubeLookTransform, learn_grade_cube
from .interface import LookTransform


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
        grade = learn_grade_cube(
            np.concatenate(befores), np.concatenate(afters),
            self._size, self._smoothing, self._min_weight,
        )
        return CubeLookTransform(grade, self._size)
