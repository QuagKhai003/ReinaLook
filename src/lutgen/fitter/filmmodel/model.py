"""model — the v2 film-emulation forward transform (composes the blocks in fixed order).

@context  The single pointwise transform the fitter solves for. Fixed pipeline order (spec §3):
          input (DWG/DI) -> [A] crosstalk -> [B] per-channel S-curves -> ... (C/D added in 1.2).
          Batch 1.1 covers A + B only.
@done     FilmModel(crosstalk, curves) with forward(rgb); identity() constructor; is_identity().
@todo     Blocks C (sat-vs-luma) + D (hue zones) — batch 1.2. Param (de)serialization — batch 1.5.
@limits   PURE: no IO, no network, no AI. Vectorized over (...,3) float64. All-neutral params ->
          input returned BIT-FOR-BIT (identity@0), preserving the sacred strength=0 base.
@affects  Built from crosstalk.py + scurve.py. Consumed by fitter/fit.py (1.4) + pipeline (1.6).
          See ADR-0001, spec §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crosstalk import CrosstalkParams, apply_crosstalk
from .scurve import SCurveParams, apply_scurve


def _identity_curves() -> tuple[SCurveParams, SCurveParams, SCurveParams]:
    return (SCurveParams(), SCurveParams(), SCurveParams())


@dataclass
class FilmModel:
    """The parametric film transform: Block A crosstalk then Block B per-channel S-curves.

    ``curves`` is the (R, G, B) tuple of tone curves — independent per channel, which is what
    lets fitted film shadows drift toward a colour. Default construction is the identity.
    """

    crosstalk: CrosstalkParams = field(default_factory=CrosstalkParams)
    curves: tuple[SCurveParams, SCurveParams, SCurveParams] = field(default_factory=_identity_curves)

    @classmethod
    def identity(cls) -> FilmModel:
        """The neutral model: forward() returns its input bit-for-bit."""
        return cls()

    def is_identity(self) -> bool:
        return self.crosstalk.is_identity() and all(c.is_identity() for c in self.curves)

    def forward(self, rgb: np.ndarray) -> np.ndarray:
        """Apply A then B to ``rgb`` (...,3) in DWG/DI working space. New array; identity model
        returns the input unchanged. Output may exceed [0,1] (regularize.py clamps at bake)."""
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.shape[-1] != 3:
            raise ValueError(f"expected (...,3), got {rgb.shape}")
        if self.is_identity():
            return rgb.copy()
        x = apply_crosstalk(rgb, self.crosstalk)   # Block A
        x = apply_scurve(x, self.curves)           # Block B
        return x
