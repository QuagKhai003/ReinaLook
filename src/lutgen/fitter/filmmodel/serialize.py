"""serialize — FilmModel <-> plain dict (the Look Profile's params payload).

@context  A fitted look must be a portable, inspectable, editable recipe (spec §4) — the
          headline advantage over OT, which had no parameters to show. This module turns the
          model into plain floats and back, exactly. File IO lives in orchestration/profile.py;
          this stays pure so the recipe display (Phase 2 UI) can reuse it.
@done     model_to_dict / model_from_dict (round-trip exact, unknown keys ignored); nested
          film_system section (negative/coupling/printer, ADR-0008) — absent in pre-8.2
          profiles, which load neutral and render bit-identically.
@todo     -
@limits   PURE: dicts of floats only, no IO. Unknown keys are IGNORED on read
          (forward-compatible); missing keys fall back to the dataclass neutral defaults, so a
          hand-trimmed profile stays valid. Values are NOT range-clamped here — the model's
          own safety (monotonic curves, clamped multipliers) is what bounds behaviour.
@affects  Used by orchestration/profile.py (save/load) + the Phase-2 recipe editor.
          See ADR-0001 b1.5, spec §4.
"""

from __future__ import annotations

from dataclasses import asdict, fields

from .crosstalk import CrosstalkParams
from .filmsystem import CouplingParams, FilmSystemParams, NegativeParams, PrintParams
from .fourierhue import FourierHueParams
from .globaltrim import GlobalParams
from .huezone import HueZoneParams
from .model import FilmModel
from .satluma import SatLumaParams
from .scurve import SCurveParams


def _from_known(cls, data: dict):
    """Construct a params dataclass from ``data``, keeping only known fields (float-coerced)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: float(v) for k, v in data.items() if k in known})


def model_to_dict(model: FilmModel) -> dict:
    """The model's parameters as a plain nested dict of floats (JSON-ready)."""
    return {
        "crosstalk": asdict(model.crosstalk),
        "curves": {
            "r": asdict(model.curves[0]),
            "g": asdict(model.curves[1]),
            "b": asdict(model.curves[2]),
        },
        "sat_luma": asdict(model.sat_luma),
        "hue_zones": asdict(model.hue_zones),
        "global": asdict(model.global_trim),
        "hue_fourier": asdict(model.hue_fourier),
        "film_system": {
            "negative": asdict(model.film_system.negative),
            "coupling": asdict(model.film_system.coupling),
            "printer": asdict(model.film_system.printer),
        },
    }


def model_from_dict(data: dict) -> FilmModel:
    """Rebuild a FilmModel from :func:`model_to_dict` output (or a hand-edited subset).

    Missing sections/keys default to neutral; unknown keys are ignored.
    """
    if not isinstance(data, dict):
        raise TypeError(f"model params must be a dict, got {type(data).__name__}")
    curves_d = data.get("curves", {})
    fs_d = data.get("film_system", {})
    return FilmModel(
        film_system=FilmSystemParams(
            negative=_from_known(NegativeParams, fs_d.get("negative", {})),
            coupling=_from_known(CouplingParams, fs_d.get("coupling", {})),
            printer=_from_known(PrintParams, fs_d.get("printer", {})),
        ),
        crosstalk=_from_known(CrosstalkParams, data.get("crosstalk", {})),
        curves=(
            _from_known(SCurveParams, curves_d.get("r", {})),
            _from_known(SCurveParams, curves_d.get("g", {})),
            _from_known(SCurveParams, curves_d.get("b", {})),
        ),
        sat_luma=_from_known(SatLumaParams, data.get("sat_luma", {})),
        hue_zones=_from_known(HueZoneParams, data.get("hue_zones", {})),
        global_trim=_from_known(GlobalParams, data.get("global", {})),
        hue_fourier=_from_known(FourierHueParams, data.get("hue_fourier", {})),
    )
