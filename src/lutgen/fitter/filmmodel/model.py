"""model — the v2 film-emulation forward transform (composes the blocks in fixed order).

@context  The single pointwise transform the fitter solves for. Fixed pipeline order (spec §3):
          input (DWG/DI) -> [A] crosstalk -> [B] per-channel S-curves -> [C] sat-vs-luma
          (Oklab) -> [D] hue-zone trims (Oklab) -> output (DWG/DI). ~33 params total
          (A:6 + B:12 + C:3 + D:12); global trims land with the fit in batch 1.4.
@done     FilmModel(crosstalk, curves, sat_luma, hue_zones).forward; identity(); is_identity().
@todo     Global exposure/black trims (1.4). Param (de)serialization -> Look Profile (1.5).
          Block E hue x luma grid (v2.1, Phase 3).
@limits   PURE: no IO, no network, no AI. Vectorized over (...,3) float64. All-neutral params ->
          input returned BIT-FOR-BIT (identity@0), preserving the sacred strength=0 base. The
          DI->Oklab round-trip runs only when C or D is active, so an A/B-only model adds no
          conversion error.
@affects  Built from crosstalk.py + scurve.py + satluma.py + huezone.py + engine/perceptual.py.
          Consumed by fitter/fit.py (1.4) + pipeline (1.6). See ADR-0001, spec §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lutgen.engine.perceptual import di_to_oklab, oklab_to_di

from .crosstalk import CrosstalkParams, apply_crosstalk
from .huezone import HueZoneParams, apply_hue_zones
from .satluma import SatLumaParams, apply_sat_luma
from .scurve import SCurveParams, apply_scurve


def _identity_curves() -> tuple[SCurveParams, SCurveParams, SCurveParams]:
    return (SCurveParams(), SCurveParams(), SCurveParams())


@dataclass
class FilmModel:
    """The parametric film transform: A crosstalk -> B per-channel S-curves -> C sat-vs-luma
    -> D hue-zone trims. C and D run in Oklab via one shared DI round-trip.

    ``curves`` is the (R, G, B) tuple of tone curves — independent per channel, which is what
    lets fitted film shadows drift toward a colour. Default construction is the identity.
    """

    crosstalk: CrosstalkParams = field(default_factory=CrosstalkParams)
    curves: tuple[SCurveParams, SCurveParams, SCurveParams] = field(default_factory=_identity_curves)
    sat_luma: SatLumaParams = field(default_factory=SatLumaParams)
    hue_zones: HueZoneParams = field(default_factory=HueZoneParams)

    @classmethod
    def identity(cls) -> FilmModel:
        """The neutral model: forward() returns its input bit-for-bit."""
        return cls()

    def is_identity(self) -> bool:
        return (
            self.crosstalk.is_identity()
            and all(c.is_identity() for c in self.curves)
            and self.sat_luma.is_identity()
            and self.hue_zones.is_identity()
        )

    def forward(self, rgb: np.ndarray) -> np.ndarray:
        """Apply A -> B -> C -> D to ``rgb`` (...,3) in DWG/DI working space. New array;
        identity model returns the input unchanged. Output may exceed [0,1] (regularize.py
        clamps at bake). The Oklab round-trip is skipped entirely when C and D are neutral."""
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.shape[-1] != 3:
            raise ValueError(f"expected (...,3), got {rgb.shape}")
        if self.is_identity():
            return rgb.copy()
        x = apply_crosstalk(rgb, self.crosstalk)          # Block A
        x = apply_scurve(x, self.curves)                  # Block B
        if not (self.sat_luma.is_identity() and self.hue_zones.is_identity()):
            lab = di_to_oklab(x)
            lab = apply_sat_luma(lab, self.sat_luma)      # Block C
            lab = apply_hue_zones(lab, self.hue_zones)    # Block D
            x = oklab_to_di(lab)
        return x
