"""filmmodel — the v2 parametric film-emulation model (fittable forward transform).

@context  v2 replaces v1's free-form OT *matching* with a ~40-param film-shaped *emulation*
          model that scipy fits (ADR-0001). This package is the PURE forward transform the
          fitter solves for: A crosstalk -> B per-channel S-curves -> C sat-vs-luma (Oklab)
          -> D hue-zone trims (Oklab), all in DWG/DI working space.
@done     Blocks A (crosstalk), B (scurve), C (satluma), D (huezone); FilmModel (A->B->C->D);
          serialize.py (params <-> dict for the Look Profile).
@todo     Staged fit (1.4); Look Profile (de)serialization (1.5). Wire into LookFitter seam.
@limits   PURE: no IO, no network, no AI. Vectorized NumPy over (...,3) float64. Neutral params
          return the input BIT-FOR-BIT (identity@0) so strength=0 stays the sacred base.
@affects  Consumed later by fitter/fit.py + orchestration/pipeline.py. See ADR-0001, spec §3.
"""

from __future__ import annotations

from .crosstalk import CrosstalkParams, apply_crosstalk, crosstalk_matrix
from .globaltrim import GlobalParams, apply_global
from .huezone import HueZoneParams, apply_hue_zones
from .model import FilmModel
from .satluma import SatLumaParams, apply_sat_luma, sat_multiplier
from .scurve import SCurveParams, apply_scurve

__all__ = [
    "CrosstalkParams",
    "FilmModel",
    "GlobalParams",
    "HueZoneParams",
    "SCurveParams",
    "SatLumaParams",
    "apply_crosstalk",
    "apply_global",
    "apply_hue_zones",
    "apply_sat_luma",
    "apply_scurve",
    "crosstalk_matrix",
    "sat_multiplier",
]
