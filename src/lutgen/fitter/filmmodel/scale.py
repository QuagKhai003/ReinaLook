"""scale — scale a fitted model's look toward neutral, per group (tone vs colour).

@context  A learned look carries the film's tonal mood (exposure/contrast) AND its colour
          cast. The cast largely lives in how the three tone curves DIFFER from each other
          (blue crushed vs red lifted = yellow), so the split is by decomposition, not by
          block: TONE = exposure + the channels' SHARED curve shape (their average);
          COLOR = each channel's deviation from that shared shape + crosstalk + sat-vs-luma
          + hue zones. Colour 0 makes all three curves identical — the cast is gone, the
          contrast stays. (First version grouped whole curves under tone; the user's yellow
          cast then sat in the wrong dial and Color did nothing visible.)
@done     scaled_model(model, tone_amount, color_amount) with mean/deviation curve split;
          Block F split (ADR-0008): tone = mean neg gamma + toe + whole print stage,
          color = per-channel gamma deviation + DIR coupling.
@todo     -
@limits   PURE: no IO. Amounts clamp to [0, 1]. Recomposed curve params are clamped to the
          fit bounds (toe/shoulder >= 0, slope [0.5, 2], pivot [0.3, 0.7]) because the
          deviation-only mix (tone 0, colour 1) is not a convex combination. t=c=1 returns
          an equal model; t=c=0 the identity.
@affects  Used by apply_tab.py (Tone/Color amount dials — bake + export). ADR-0005/0007.
"""

from __future__ import annotations

from dataclasses import asdict

from .crosstalk import CrosstalkParams
from .filmsystem import (
    CouplingParams,
    FilmSystemParams,
    NegativeParams,
    PrinterLights,
    PrintParams,
)
from .fourierhue import FourierHueParams
from .globaltrim import GlobalParams
from .huezone import HueZoneParams
from .model import FilmModel
from .satluma import SatLumaParams
from .scurve import SCurveParams
from .splittone import SplitToneParams

# neutral curve params and the fit bounds (recomposition clamps back into them)
_CURVE_NEUTRAL = {"toe": 0.0, "shoulder": 0.0, "slope": 1.0, "pivot": 0.5}
_CURVE_LO = {"toe": 0.0, "shoulder": 0.0, "slope": 0.5, "pivot": 0.3}
_CURVE_HI = {"toe": 2.0, "shoulder": 2.0, "slope": 2.0, "pivot": 0.7}


def _lerp(neutral: float, value: float, amount: float) -> float:
    return neutral + (value - neutral) * amount


def _split_curves(curves, t: float, c: float):
    """TONE scales the channels' shared shape (mean curve); COLOR scales each channel's
    deviation from it: param = neutral + t*(mean - neutral) + c*(channel - mean)."""
    fields = ("toe", "shoulder", "slope", "pivot")
    per_ch = [asdict(cv) for cv in curves]
    mean = {f: sum(d[f] for d in per_ch) / 3.0 for f in fields}
    out = []
    for d in per_ch:
        params = {}
        for f in fields:
            v = _CURVE_NEUTRAL[f] + t * (mean[f] - _CURVE_NEUTRAL[f]) + c * (d[f] - mean[f])
            params[f] = min(_CURVE_HI[f], max(_CURVE_LO[f], v))
        out.append(SCurveParams(**params))
    return tuple(out)


def _split_film_system(fs: FilmSystemParams, t: float, c: float) -> FilmSystemParams:
    """Same decomposition for Block F: TONE = shared negative shape (mean gamma, toe) +
    the whole print stage (contrast/convergence); COLOR = per-channel gamma deviation +
    DIR coupling. Gammas clamp to a sane positive range (the t/c mix is not convex)."""
    g = (fs.negative.g_r, fs.negative.g_g, fs.negative.g_b)
    mean = sum(g) / 3.0
    g_r, g_g, g_b = (min(2.0, max(0.5, 1.0 + t * (mean - 1.0) + c * (v - mean))) for v in g)
    # printer lights: the mean offset is exposure-like (tone), the deviations are the
    # colour timing (color) — same decomposition as the curves
    li = (fs.lights.r, fs.lights.g, fs.lights.b)
    li_mean = sum(li) / 3.0
    l_r, l_g, l_b = (t * li_mean + c * (v - li_mean) for v in li)
    return FilmSystemParams(
        lights=PrinterLights(r=l_r, g=l_g, b=l_b),
        negative=NegativeParams(g_r=g_r, g_g=g_g, g_b=g_b,
                                toe=t * fs.negative.toe, toe_at=fs.negative.toe_at),
        coupling=CouplingParams(**{k: max(0.0, v * c)
                                   for k, v in asdict(fs.coupling).items()}),
        printer=PrintParams(slope=1.0 + t * (fs.printer.slope - 1.0),
                            shoulder=t * fs.printer.shoulder, ptoe=t * fs.printer.ptoe,
                            range_hi=fs.printer.range_hi, range_lo=fs.printer.range_lo),
    )


def scaled_model(model: FilmModel, tone_amount: float = 1.0,
                 color_amount: float = 1.0) -> FilmModel:
    """The model with its TONE (exposure + shared curve shape) scaled by ``tone_amount``
    and its COLOR (per-channel curve deviation + crosstalk + sat-vs-luma + hue zones) by
    ``color_amount``. Amounts clamp to [0, 1]; 1/1 reproduces the model, 0/0 the identity."""
    t = min(1.0, max(0.0, float(tone_amount)))
    c = min(1.0, max(0.0, float(color_amount)))

    curves = _split_curves(model.curves, t, c)
    crosstalk = CrosstalkParams(**{k: v * c for k, v in asdict(model.crosstalk).items()})
    sat = model.sat_luma
    sat_luma = SatLumaParams(shadow=_lerp(1.0, sat.shadow, c),
                             mid=_lerp(1.0, sat.mid, c),
                             high=_lerp(1.0, sat.high, c))
    hue_zones = HueZoneParams(**{k: v * c for k, v in asdict(model.hue_zones).items()})
    hue_fourier = FourierHueParams(**{k: v * c for k, v in asdict(model.hue_fourier).items()})
    return FilmModel(
        global_trim=GlobalParams(exposure=model.global_trim.exposure * t),
        crosstalk=crosstalk,
        curves=curves,
        sat_luma=sat_luma,
        hue_zones=hue_zones,
        hue_fourier=hue_fourier,
        film_system=_split_film_system(model.film_system, t, c),
        split_tone=SplitToneParams(**{k: v * c                    # pure colour: colour dial
                                      for k, v in asdict(model.split_tone).items()}),
    )
