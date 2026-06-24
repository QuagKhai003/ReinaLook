"""strength — blend a look over the protected base by a strength dial.

@context  The structural guarantee behind the Golden Rule: the strength dial interpolates
          between the base and a looked version of it, so the base can never be broken.
@done     blend(base, look, s) = (1-s)*base + s*look (bit-exact endpoints).
@todo     -
@limits   PURE: no IO. s clamped to [0,1]. Blends in the native Rec.709 g2.4 code space
          (the cube's own space) per ADR-0002. base must be load_base() (never convert.py).
@affects  base from engine/base.py; look is a sampled LookTransform (L2). Output -> regularize
          -> cube_io. See codemap/INDEX.md + Plan/20_COLOR_PIPELINE.md §3 + ADR-0002.
"""

from __future__ import annotations

import numpy as np


def blend(base: np.ndarray, look: np.ndarray, strength: float) -> np.ndarray:
    """Interpolate between ``base`` and ``look`` by ``strength`` in [0, 1].

    Returns ``(1 - s) * base + s * look``, which equals ``base + s * (look - base)`` but is
    bit-exact at the endpoints: ``s=0`` returns ``base`` and ``s=1`` returns ``look`` exactly.
    ``base`` and ``look`` are flat ``(N, 3)`` arrays in the same (g2.4-encoded) space.
    """
    base = np.asarray(base, dtype=np.float64)
    look = np.asarray(look, dtype=np.float64)
    if base.shape != look.shape:
        raise ValueError(f"base {base.shape} and look {look.shape} must match")
    s = float(np.clip(strength, 0.0, 1.0))
    if s == 0.0:
        return base.copy()
    if s == 1.0:
        return look.copy()
    return (1.0 - s) * base + s * look
