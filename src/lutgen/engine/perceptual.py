"""perceptual — Oklab conversion from/to Rec.709 g2.4 AND DWG/DI code values.

@context  Optimal transport works best in a perceptual space (Plan §3). Oklab is defined from
          linear sRGB = linear Rec.709 (shared primaries), so conversion is exact via the
          published Oklab matrices — decode g2.4 -> linear -> Oklab and back. The v2 film model
          (ADR-0001) works in DWG/DI, so a DI bridge decodes DI -> linear DWG -> linear Rec.709
          -> Oklab (and back) with NO clipping, preserving out-of-709 DWG colours.
@done     to_oklab / from_oklab (g2.4); di_to_oklab / oklab_to_di (DWG/DI, round-trip exact).
@limits   PURE: no IO. g2.4 path clamps negatives (display code values); the DI path never
          clips (log working space — round-trip must be lossless for the sacred base).
@affects  Used by fitter/rich.py (Oklab MKL) + stats.py + fitter/filmmodel (Blocks C/D).
          Built on engine/spaces.py. See ADR-0011, ADR-0001.
"""

from __future__ import annotations

import numpy as np

from .spaces import DWG_TO_REC709_MATRIX, di_decode, di_encode

GAMMA = 2.4

_REC709_TO_DWG_MATRIX = np.linalg.inv(DWG_TO_REC709_MATRIX)

# Oklab matrices (Björn Ottosson), linear sRGB/Rec.709 <-> Oklab.
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def to_oklab(rgb_g24: np.ndarray) -> np.ndarray:
    """Rec.709 g2.4 code values -> Oklab (L, a, b)."""
    rgb = np.asarray(rgb_g24, dtype=np.float64)
    linear = np.power(np.clip(rgb, 0.0, None), GAMMA)
    lms = linear @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    """Oklab (L, a, b) -> Rec.709 g2.4 code values."""
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _M2_INV.T
    lms = lms_ ** 3
    linear = lms @ _M1_INV.T
    return np.power(np.clip(linear, 0.0, None), 1.0 / GAMMA)


def di_to_oklab(di_code: np.ndarray) -> np.ndarray:
    """DWG/DI code values -> Oklab (L, a, b). No clipping: DI decode -> linear DWG ->
    linear Rec.709 (may be negative for out-of-709 colours; cbrt handles it) -> Oklab."""
    linear709 = di_decode(di_code) @ DWG_TO_REC709_MATRIX.T
    lms = linear709 @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_di(lab: np.ndarray) -> np.ndarray:
    """Oklab (L, a, b) -> DWG/DI code values (exact inverse of di_to_oklab, no clipping).

    errstate: colour's DI oetf evaluates its log branch on ALL values before np.where selects
    the linear branch for small/negative ones — correct result, spurious warning. Silenced."""
    lab = np.asarray(lab, dtype=np.float64)
    lms = (lab @ _M2_INV.T) ** 3
    linear709 = lms @ _M1_INV.T
    with np.errstate(invalid="ignore"):
        return di_encode(linear709 @ _REC709_TO_DWG_MATRIX.T)
