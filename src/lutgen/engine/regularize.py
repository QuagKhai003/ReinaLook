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

_LUMA = np.array([0.2126, 0.7152, 0.0722])   # Rec.709


def clamp(samples: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clip samples into the valid output range [lo, hi] (hard, per-channel)."""
    return np.clip(np.asarray(samples, dtype=np.float64), lo, hi)


def gamut_clamp(samples: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Bring out-of-range colors into [lo, hi] by **desaturating toward their own luma**, not by
    clipping each channel independently. Per-channel clipping shifts hue and posterizes saturated
    tones (harsh skin); this preserves hue + brightness and only reduces chroma as much as needed.
    In-gamut samples (incl. the base cube) are returned unchanged → s=0 stays bit-exact."""
    s = np.asarray(samples, dtype=np.float64)
    out = s.copy()
    oog = ((s < lo) | (s > hi)).any(axis=-1)            # only touch out-of-gamut samples
    if not oog.any():
        return out                                      # in-gamut input returned byte-identical
    p = s[oog]
    luma = np.clip(p @ _LUMA, lo, hi)[..., None]        # achromatic anchor, clamped
    chroma = p - (p @ _LUMA)[..., None]                 # color relative to original luma
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(chroma > 1e-12, (hi - luma) / chroma, np.inf)
        t_lo = np.where(chroma < -1e-12, (lo - luma) / chroma, np.inf)
    t = np.clip(np.minimum(t_hi, t_lo).min(axis=-1), 0.0, 1.0)[..., None]
    out[oog] = np.clip(luma + t * chroma, lo, hi)       # desaturate toward luma to fit
    return out


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
    gamut_aware: bool = True,
) -> np.ndarray:
    """Bring into range (gamut-aware soft clamp by default, vs hard per-channel clip), then
    optionally remove neutral-axis inversions. Endpoints preserved; in-gamut input (incl. the base)
    is unchanged, so s=0 stays bit-exact."""
    out = gamut_clamp(samples, *clamp_range) if gamut_aware else clamp(samples, *clamp_range)
    if neutral_monotonic:
        out = enforce_neutral_monotonic(out, size)
    return out
