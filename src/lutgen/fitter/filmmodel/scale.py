"""scale — scale a fitted model's look toward neutral, per group (tone vs colour).

@context  A learned look carries the film's absolute tonal mood (exposure/contrast) AND its
          palette. On footage unlike the film's frames the mood can be wrong while the
          palette is right (ADR-0005: dim warm pool -> muddy daylight). Because the model is
          parametric, "less of the look" is exact: interpolate each parameter toward its
          neutral value. Tone and colour scale independently.
@done     scaled_model(model, tone_amount, color_amount).
@todo     -
@limits   PURE: no IO. Amounts clamp to [0, 1] — interpolation toward neutral is convex, so
          every fit-bound and monotonicity guarantee survives scaling (overdrive > 1 would
          break that and is refused). t=c=1 returns an equal model; t=c=0 the identity.
@affects  Used by apply_tab.py (Tone/Color amount dials — bake + export). ADR-0005.
"""

from __future__ import annotations

from dataclasses import asdict

from .crosstalk import CrosstalkParams
from .globaltrim import GlobalParams
from .huezone import HueZoneParams
from .model import FilmModel
from .satluma import SatLumaParams
from .scurve import SCurveParams


def _lerp(neutral: float, value: float, amount: float) -> float:
    return neutral + (value - neutral) * amount


def scaled_model(model: FilmModel, tone_amount: float = 1.0,
                 color_amount: float = 1.0) -> FilmModel:
    """The model with its tone group (G exposure + B curves) scaled by ``tone_amount`` and
    its colour group (A crosstalk + C sat-vs-luma + D hue zones) by ``color_amount``.
    Amounts are clamped to [0, 1]; 1/1 reproduces the model, 0/0 is the identity."""
    t = min(1.0, max(0.0, float(tone_amount)))
    c = min(1.0, max(0.0, float(color_amount)))

    curves = tuple(
        SCurveParams(
            toe=cv.toe * t,
            shoulder=cv.shoulder * t,
            slope=_lerp(1.0, cv.slope, t),
            pivot=_lerp(0.5, cv.pivot, t),
        )
        for cv in model.curves
    )
    crosstalk = CrosstalkParams(**{k: v * c for k, v in asdict(model.crosstalk).items()})
    sat = model.sat_luma
    sat_luma = SatLumaParams(shadow=_lerp(1.0, sat.shadow, c),
                             mid=_lerp(1.0, sat.mid, c),
                             high=_lerp(1.0, sat.high, c))
    hue_zones = HueZoneParams(**{k: v * c for k, v in asdict(model.hue_zones).items()})
    return FilmModel(
        global_trim=GlobalParams(exposure=model.global_trim.exposure * t),
        crosstalk=crosstalk,
        curves=curves,
        sat_luma=sat_luma,
        hue_zones=hue_zones,
    )
