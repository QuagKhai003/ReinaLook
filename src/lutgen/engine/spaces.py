"""spaces — color spaces & transfer functions for the protected conversion.

@context  Wraps colour-science so convert.py has clean, named building blocks for the
          DWG/DI -> Rec.709 g2.4 base: DI decode, DWG->Rec.709 gamut matrix, g2.4 encode.
@done     DI decode/encode, DWG->Rec.709 linear matrix, pure gamma-2.4 encode/decode.
@limits   PURE: no IO. Output transfer is a PURE 2.4 power (Plan §2 step 4), NOT BT.1886/sRGB.
          Final [0,1] clamping is regularize.py's job; encode guards negatives to avoid NaN.
@affects  Built on colour-science 0.4.7. Consumed by engine/convert.py.
          See codemap/INDEX.md (engine/spaces.py) + Plan/20_COLOR_PIPELINE.md §2.
"""

from __future__ import annotations

import colour
import numpy as np
from colour.models import (
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
)

# Output display gamma (delivery target the user specified: Rec.709 Gamma 2.4).
GAMMA = 2.4

_DWG = colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]
_REC709 = colour.RGB_COLOURSPACES["ITU-R BT.709"]

# 3x3 gamut matrix: linear DWG primaries -> linear Rec.709 primaries.
# Both spaces are D65, so chromatic adaptation is effectively identity; Bradford is harmless
# and mirrors Resolve's "Use White Point Adaptation" CST behaviour.
DWG_TO_REC709_MATRIX = colour.matrix_RGB_to_RGB(
    _DWG, _REC709, chromatic_adaptation_transform="Bradford"
)


def di_decode(code: np.ndarray) -> np.ndarray:
    """DaVinci Intermediate encoded -> scene-linear (per channel)."""
    return np.asarray(oetf_inverse_DaVinciIntermediate(code), dtype=np.float64)


def di_encode(linear: np.ndarray) -> np.ndarray:
    """Scene-linear -> DaVinci Intermediate encoded (inverse of di_decode; for tests/IO)."""
    return np.asarray(oetf_DaVinciIntermediate(linear), dtype=np.float64)


def dwg_to_rec709_linear(linear_dwg: np.ndarray) -> np.ndarray:
    """Linear DWG primaries -> linear Rec.709 primaries via the 3x3 gamut matrix."""
    linear_dwg = np.asarray(linear_dwg, dtype=np.float64)
    return linear_dwg @ DWG_TO_REC709_MATRIX.T


def rec709_g24_encode(linear: np.ndarray) -> np.ndarray:
    """Linear Rec.709 -> Gamma 2.4 encoded. Negatives clamped to 0 to avoid NaN from the
    fractional power; values > 1 are left for regularize.py to clamp."""
    linear = np.asarray(linear, dtype=np.float64)
    return np.power(np.maximum(linear, 0.0), 1.0 / GAMMA)


def rec709_g24_decode(code: np.ndarray) -> np.ndarray:
    """Gamma 2.4 encoded -> linear Rec.709 (inverse of rec709_g24_encode)."""
    code = np.asarray(code, dtype=np.float64)
    return np.power(np.maximum(code, 0.0), GAMMA)
