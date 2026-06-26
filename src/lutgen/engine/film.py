"""film — parametric film-stock transfer (ADR-0021).

@context  Reference-matching transfers colour STATISTICS (feels like a coat). Film stock has a
          designed TRANSFER behaviour — a per-channel S-curve, highlights that bleach toward white,
          tone-dependent colour (split-tone). This is a content-independent transfer applied to the
          cube samples, so it reshapes how the DWG output rolls, like a real stock.
@done     FilmStock + apply_film() — filmic S-curve, highlight bleach, split-tone, saturation.
@limits   PURE: no IO. Operates on flat (...,3) Rec.709 g2.4 [0,1]. Neutral params (all 0) return
          the input UNCHANGED (so strength=0 / no-op stays bit-exact). Final clamp is regularize's.
@affects  Composed in orchestration/pipeline after the look + adjustments. See ADR-0021.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722])   # Rec.709
_PIVOT = 0.435                                # ~mid grey in g2.4


@dataclass
class FilmStock:
    """Film-transfer controls. All default 0 (identity). Ranges −1..1 except the 0..1 ones noted."""

    contrast: float = 0.0          # −1 flat … +1 filmic S-curve (per-channel, around mid grey)
    toe: float = 0.0               # 0..1 — lift/soften the shadow foot (matte blacks)
    shoulder: float = 0.0          # 0..1 — roll off the highlights (soft, no hard clip)
    highlight_bleach: float = 0.0  # 0..1 — desaturate highlights toward white (film bleach)
    split_warm: float = 0.0        # −1 cool-highs/warm-shadows … +1 warm-highs/cool-shadows
    saturation: float = 0.0        # −1 grey … +1 vivid

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _scurve(x, amount):
    """Per-channel S-curve around the pivot: contrast in the mids, gentle near the ends."""
    d = x - _PIVOT
    # cubic S: x + k * d * (pivot - |d-span|)… use a smooth odd function bounded in range
    s = d + amount * d * (1.0 - np.abs(d) / max(_PIVOT, 1.0 - _PIVOT))
    return _PIVOT + s


def apply_film(samples: np.ndarray, film: FilmStock) -> np.ndarray:
    """Apply the film transfer to ``samples`` (...,3 in [0,1]). New array; neutral params return the
    input unchanged (bit-for-bit)."""
    if film.is_identity():
        return np.asarray(samples, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64).copy()

    # 1. Per-channel filmic S-curve — the core tonal reshape (also creates highlight colour cross-talk).
    if film.contrast:
        x = _scurve(x, film.contrast)

    # 2. Toe — lift the shadow foot (matte blacks) without touching mids/highs.
    if film.toe:
        m = 1.0 - _smoothstep(0.0, 0.35, x)            # 1 deep in shadow → 0 by ~mid
        x = x + film.toe * 0.12 * m

    # 3. Shoulder — soft highlight roll-off (compress toward white, no hard clip).
    if film.shoulder:
        knee = 0.6
        over = np.maximum(x - knee, 0.0)
        rolled = knee + over / (1.0 + film.shoulder * 3.0 * over / max(1.0 - knee, 1e-6))
        x = np.where(x > knee, rolled, x)

    # 4. Split-tone — warm/cool by luma (film colour science: shadows one way, highlights the other).
    if film.split_warm:
        luma = x @ _LUMA
        hi = _smoothstep(0.5, 1.0, luma)[..., None]
        lo = (1.0 - _smoothstep(0.0, 0.5, luma))[..., None]
        warm = np.array([0.06, 0.0, -0.06])            # toward red/amber
        x = x + film.split_warm * (hi * warm - lo * warm)

    # 5. Highlight bleach — desaturate toward white as luma rises (film highlights wash out).
    if film.highlight_bleach:
        luma = (x @ _LUMA)[..., None]
        t = (film.highlight_bleach * _smoothstep(0.55, 1.0, luma[..., 0]))[..., None]
        x = x + t * (1.0 - x)                          # pull toward white in the brights

    # 6. Saturation — overall chroma around luma.
    if film.saturation:
        luma = (x @ _LUMA)[..., None]
        x = luma + (1.0 + film.saturation) * (x - luma)

    return x
