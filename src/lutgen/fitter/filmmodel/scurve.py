"""scurve — Block B of the v2 film model: per-channel monotonic S-curves.

@context  Film's characteristic curve: shadows compress (toe), midtones respond steeply,
          highlights roll off (shoulder), never hard-clipping. The R/G/B curves differ
          slightly from each other — that is why film shadows drift toward a colour instead of
          staying neutral (spec §2.1). Each channel gets its own SCurveParams.
@done     SCurveParams (toe/shoulder/slope/pivot) + apply_scurve via monotone cubic Hermite.
@todo     -
@limits   PURE: no IO. Curve is C1-smooth and strictly monotonic for the bounded ranges
          (toe,shoulder >= 0; slope in [0.5,2]); endpoints fixed at f(0)=0, f(1)=1 (black/white
          preserved). Identity when toe=shoulder=0 and slope=1 -> input returned essentially
          bit-for-bit. Values outside [0,1] extrapolate linearly (C1) so crosstalk overshoot is
          safe. Final [0,1] clamp is regularize.py's job. Vectorized over (...,3) float64.
@affects  Applied after crosstalk in model.FilmModel. See ADR-0001 batch 1.1, spec §3 Block B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Monotonicity guard: with anchors on the diagonal every segment secant is 1, so the
# Fritsch-Carlson condition reduces to keeping each tangent in [0, 3]. We clamp defensively.
_MAX_TANGENT = 3.0


@dataclass
class SCurveParams:
    """One channel's tone curve. Neutral (toe=0, shoulder=0, slope=1) is the identity line.

    ``toe``/``shoulder`` >= 0 flatten the shadow/highlight ends (compression); ``slope`` is the
    midtone gain at the pivot (>1 steepens = more contrast); ``pivot`` is the fixed crossover
    point the curve passes through on the diagonal. The optimizer bounds these in batch 1.4.
    """

    toe: float = 0.0        # >= 0 : shadow-foot compression
    shoulder: float = 0.0   # >= 0 : highlight roll-off
    slope: float = 1.0      # midtone gain at the pivot (identity = 1)
    pivot: float = 0.5      # in (0,1) : the fixed crossover on the diagonal

    def is_identity(self) -> bool:
        return self.toe == 0.0 and self.shoulder == 0.0 and self.slope == 1.0

    def tangents(self) -> tuple[float, float, float]:
        """Endpoint + midtone tangents (m0 at x=0, mid at pivot, m1 at x=1), clamped to [0, max]."""
        m0 = 1.0 / (1.0 + max(self.toe, 0.0))
        m1 = 1.0 / (1.0 + max(self.shoulder, 0.0))
        mid = float(np.clip(self.slope, 0.0, _MAX_TANGENT))
        return m0, float(np.clip(mid, 0.0, _MAX_TANGENT)), m1


def _hermite(x: np.ndarray, x0: float, x1: float, y0: float, y1: float, m0: float, m1: float) -> np.ndarray:
    """Cubic Hermite on [x0,x1] with values y0,y1 and tangents m0,m1. Vectorized."""
    h = x1 - x0
    t = (x - x0) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def _curve_1d(x: np.ndarray, p: SCurveParams) -> np.ndarray:
    """Evaluate one channel's curve. Two Hermite segments split at the pivot, linear C1
    extrapolation outside [0,1]. Anchors sit on the diagonal: (0,0), (pivot,pivot), (1,1)."""
    m0, mid, m1 = p.tangents()
    pv = float(np.clip(p.pivot, 1e-3, 1.0 - 1e-3))
    y = np.empty_like(x)

    lo = x <= 0.0
    seg1 = (x > 0.0) & (x <= pv)
    seg2 = (x > pv) & (x <= 1.0)
    hi = x > 1.0

    y[lo] = x[lo] * m0                                          # line through 0, slope m0
    y[seg1] = _hermite(x[seg1], 0.0, pv, 0.0, pv, m0, mid)
    y[seg2] = _hermite(x[seg2], pv, 1.0, pv, 1.0, mid, m1)
    y[hi] = 1.0 + (x[hi] - 1.0) * m1                            # line from (1,1), slope m1
    return y


def apply_scurve(rgb: np.ndarray, params: tuple[SCurveParams, SCurveParams, SCurveParams]) -> np.ndarray:
    """Apply per-channel S-curves to ``rgb`` (...,3). ``params`` is (R,G,B). New array; if all
    three channels are identity the input is returned unchanged (bit-for-bit)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {rgb.shape}")
    if len(params) != 3:
        raise ValueError(f"expected 3 channel curves, got {len(params)}")
    if all(p.is_identity() for p in params):
        return rgb.copy()
    out = rgb.copy()
    for c, p in enumerate(params):
        if not p.is_identity():
            out[..., c] = _curve_1d(rgb[..., c], p)
    return out
