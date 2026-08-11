"""satluma — Block C of the v2 film model: brightness-dependent saturation (in Oklab).

@context  Film desaturates in deep shadows and bright highlights and holds rich saturation in
          the midtones — saturation is a CURVE over luminance, not a global number (spec §2.3).
          Three control points (shadow/mid/highlight multipliers) joined by a smooth C1 curve.
@done     SatLumaParams (3 multipliers) + apply_sat_luma on Oklab (...,3) arrays.
@todo     -
@limits   PURE: no IO. Operates on Oklab; scales chroma (a,b) only — L untouched. Neutral
          (all multipliers 1) returns input BIT-FOR-BIT. Multiplier curve is C1 (smoothstep-
          eased segments, zero-slope joins) and clamped >= 0 so chroma can never flip sign.
          L outside [0,1] uses the end multiplier (flat extrapolation).
@affects  Applied inside model.FilmModel between Blocks B and D. See ADR-0001 b1.2, spec §3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SatLumaParams:
    """Saturation multipliers at three luminance anchors (L = 0, 0.5, 1). Neutral = all 1.

    The optimizer bounds these in batch 1.4 (e.g. [0.2, 2]); values are clamped >= 0 here so
    no parameter set can produce negative chroma.
    """

    shadow: float = 1.0     # multiplier at L = 0
    mid: float = 1.0        # multiplier at L = 0.5
    high: float = 1.0       # multiplier at L = 1

    def is_identity(self) -> bool:
        return self.shadow == 1.0 and self.mid == 1.0 and self.high == 1.0


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sat_multiplier(luma: np.ndarray, p: SatLumaParams) -> np.ndarray:
    """The C1 multiplier curve over L: smoothstep-eased between the three anchors.

    Zero-slope at every anchor (smoothstep ends flat) -> no kink at L=0.5 and flat
    extrapolation outside [0,1] is automatically C1. Clamped >= 0.
    """
    luma = np.asarray(luma, dtype=np.float64)
    lo = float(max(p.shadow, 0.0))
    mi = float(max(p.mid, 0.0))
    hi = float(max(p.high, 0.0))
    mult = np.where(
        luma <= 0.5,
        lo + (mi - lo) * _smoothstep(0.0, 0.5, luma),
        mi + (hi - mi) * _smoothstep(0.5, 1.0, luma),
    )
    return np.maximum(mult, 0.0)


def apply_sat_luma(lab: np.ndarray, p: SatLumaParams) -> np.ndarray:
    """Scale Oklab chroma by the luminance-dependent multiplier. New array; neutral params
    return the input unchanged (bit-for-bit). L is never modified."""
    lab = np.asarray(lab, dtype=np.float64)
    if lab.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {lab.shape}")
    if p.is_identity():
        return lab.copy()
    out = lab.copy()
    mult = sat_multiplier(lab[..., 0], p)[..., None]
    out[..., 1:] = lab[..., 1:] * mult
    return out
