"""filmmodel — the v2 parametric film-emulation model (fittable forward transform).

@context  v2 replaces v1's free-form OT *matching* with a ~40-param film-shaped *emulation*
          model that scipy fits (ADR-0001). This package is the PURE forward transform the
          fitter solves for: A crosstalk -> B per-channel S-curves -> C sat-vs-luma (Oklab)
          -> D hue-zone trims (Oklab), all in DWG/DI working space.
@done     Blocks G (globaltrim), A (crosstalk), F (filmsystem: neg→coupling→print, ADR-0008,
          + character.py preset), B (scurve, legacy), C (satluma), D (huezone + fourierhue);
          FilmModel (G->A->F->B->C->D); serialize.py (params <-> dict); scale.py (dials).
@todo     Fit v2 targets F instead of B (ADR-0008 b8.4).
@limits   PURE: no IO, no network, no AI. Vectorized NumPy over (...,3) float64. Neutral params
          return the input BIT-FOR-BIT (identity@0) so strength=0 stays the sacred base.
@affects  Consumed later by fitter/fit.py + orchestration/pipeline.py. See ADR-0001, spec §3.
"""

from __future__ import annotations

from .character import film_print_character
from .crosstalk import CrosstalkParams, apply_crosstalk, crosstalk_matrix
from .filmsystem import (
    CouplingParams,
    FilmSystemParams,
    NegativeParams,
    PrinterLights,
    PrintParams,
    apply_film_system,
)
from .fourierhue import FourierHueParams, apply_fourier_hue
from .globaltrim import GlobalParams, apply_global
from .huezone import HueZoneParams, apply_hue_zones
from .model import FilmModel
from .satluma import SatLumaParams, apply_sat_luma, sat_multiplier
from .scurve import SCurveParams, apply_scurve
from .splittone import SplitToneParams, apply_split_tone

__all__ = [
    "CouplingParams",
    "CrosstalkParams",
    "FilmModel",
    "FilmSystemParams",
    "FourierHueParams",
    "GlobalParams",
    "HueZoneParams",
    "NegativeParams",
    "PrintParams",
    "PrinterLights",
    "SCurveParams",
    "SatLumaParams",
    "SplitToneParams",
    "apply_crosstalk",
    "apply_film_system",
    "apply_fourier_hue",
    "apply_global",
    "apply_hue_zones",
    "apply_sat_luma",
    "apply_scurve",
    "apply_split_tone",
    "crosstalk_matrix",
    "film_print_character",
    "sat_multiplier",
]
