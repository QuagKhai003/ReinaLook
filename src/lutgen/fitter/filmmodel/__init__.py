"""filmmodel — the v2 parametric film-emulation model (fittable forward transform).

@context  v2 replaces v1's free-form OT *matching* with a ~40-param film-shaped *emulation*
          model that scipy fits (ADR-0001). This package is the PURE forward transform the
          fitter solves for. Batch 1.1 ships Block A (3x3 crosstalk) + Block B (per-channel
          monotonic S-curves); Blocks C/D (Oklab sat-vs-luma, hue zones) land in batch 1.2.
@done     CrosstalkParams/apply_crosstalk (A); SCurveParams/apply_scurve (B); FilmModel (A->B).
@todo     Blocks C+D (batch 1.2); staged fit (1.4). Wire into the LookFitter seam.
@limits   PURE: no IO, no network, no AI. Vectorized NumPy over (...,3) float64. Neutral params
          return the input BIT-FOR-BIT (identity@0) so strength=0 stays the sacred base.
@affects  Consumed later by fitter/fit.py + orchestration/pipeline.py. See ADR-0001, spec §3.
"""

from __future__ import annotations

from .crosstalk import CrosstalkParams, apply_crosstalk, crosstalk_matrix
from .model import FilmModel
from .scurve import SCurveParams, apply_scurve

__all__ = [
    "CrosstalkParams",
    "FilmModel",
    "SCurveParams",
    "apply_crosstalk",
    "apply_scurve",
    "crosstalk_matrix",
]
