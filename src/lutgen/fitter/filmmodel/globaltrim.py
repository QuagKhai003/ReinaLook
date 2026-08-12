"""globaltrim — Block G of the v2 film model: global exposure trim (spec §3 budget row).

@context  S-curves anchored at (0,0)/(1,1) cannot express a global level shift, so a look
          that is simply darker/brighter than the source world was unreachable (observed:
          tone stage bound-slam on a real pool, ADR-0003/0004). DWG/DI is log-encoded, so a
          constant CODE OFFSET is an exposure change — one parameter restores the missing
          degree of freedom.
@done     GlobalParams(exposure) + apply_global (x + exposure, identity@0 bit-for-bit).
@todo     Black-offset trim if real pools show the need (spec budget allows ~4-7 globals).
@limits   PURE: no IO. Applied FIRST in the pipeline (before crosstalk). Output may leave
          [0,1]; downstream blocks extrapolate C1 and regularize clamps at bake. The fit
          bounds exposure to ±0.3 (≈ ±4 stops in DI).
@affects  Composed first in model.FilmModel; fitted in the tone stage (fit.py); serialized
          under "global" (serialize.py); editable in the recipe editor. ADR-0004 b4.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GlobalParams:
    """Global trims. ``exposure`` is a DWG/DI code offset (log space: +0.07 ≈ +1 stop).
    Neutral = 0 (identity)."""

    exposure: float = 0.0

    def is_identity(self) -> bool:
        return self.exposure == 0.0


def apply_global(rgb: np.ndarray, params: GlobalParams) -> np.ndarray:
    """Apply the global exposure offset to ``rgb`` (...,3) DI code values. New array;
    identity params return the input unchanged (bit-for-bit)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {rgb.shape}")
    if params.is_identity():
        return rgb.copy()
    return rgb + params.exposure
