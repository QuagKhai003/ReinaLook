"""regularize — clean the final blended cube (clamp, neutral-axis guard, endpoints).

@context  Stage C (Plan §4): keep the output cube valid after a look blend — no out-of-range
          values, no inversions on the neutral axis, endpoints preserved. Runs on the final.
@done     clamp; enforce_neutral_monotonic; regularize() compose.
@todo     Gentle anti-banding smoothing (deferred; matters mainly for the Rich fitter).
@limits   PURE: no IO. Operates on flat (size**3, 3) red-fastest samples. Neutral guard touches
          only the grey diagonal; does not reorder the full 3D lattice.
@affects  Input from engine/strength.py; output to engine/cube_io.py. Uses grid lattice layout.
          See codemap/INDEX.md + Plan/20_COLOR_PIPELINE.md §4 + ADR-0002.
"""

from __future__ import annotations

import numpy as np

from .grid import DEFAULT_SIZE, flatten_lattice, reshape_to_lattice


def clamp(samples: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clip samples into the valid output range [lo, hi]."""
    return np.clip(np.asarray(samples, dtype=np.float64), lo, hi)


def enforce_neutral_monotonic(samples: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Guarantee the neutral (grey) axis is non-decreasing — greys must stay ordered.

    The grey nodes are the lattice diagonal ``[k, k, k]``. A look can invert them (a darker
    grey ending up brighter than a lighter one); this applies a running max along the diagonal
    per channel to remove inversions, leaving everything else untouched.
    """
    samples = np.asarray(samples, dtype=np.float64).copy()
    lat = reshape_to_lattice(samples, size)
    diag = lat[np.arange(size), np.arange(size), np.arange(size)]  # (size, 3)
    fixed = np.maximum.accumulate(diag, axis=0)
    lat[np.arange(size), np.arange(size), np.arange(size)] = fixed
    return flatten_lattice(lat)


def regularize(
    samples: np.ndarray,
    size: int = DEFAULT_SIZE,
    *,
    clamp_range: tuple[float, float] = (0.0, 1.0),
    neutral_monotonic: bool = True,
) -> np.ndarray:
    """Clamp, then (optionally) remove neutral-axis inversions. Endpoints are preserved because
    black (0,0,0) and white (1,1,1) are fixed points of both operations."""
    out = clamp(samples, *clamp_range)
    if neutral_monotonic:
        out = enforce_neutral_monotonic(out, size)
    return out
