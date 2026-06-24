"""adjust — deterministic creative adjustments baked onto the cube (ADR-0020).

@context  Beyond reference-matching: hands-on controls to push the look toward what the user wants
          (contrast, saturation, warmth, shadow/highlight, filmic highlight roll-off). Applied to
          the looked cube samples before the strength blend, so they ride on the LUT.
@done     apply_adjustments() with 7 controls; identity at neutral (all 0).
@limits   PURE: no IO. Operates on flat (...,3) Rec.709 g2.4 samples in [0,1]. Neutral params
          (all 0) return the input UNCHANGED (so strength=0 / no-op stays bit-exact). Final clamp
          is regularize's job; this may transiently exceed [0,1].
@affects  Used by orchestration/pipeline. See Plan/30_LOOK_FITTER.md §5 + ADR-0020.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722])   # Rec.709
_PIVOT = 0.435                                # contrast pivot (~middle grey in g2.4)


@dataclass
class Adjustments:
    """Creative grade controls. All default to 0 (identity). Ranges: −1..1 except roll-off 0..1."""

    contrast: float = 0.0        # −1 flat … +1 punchy (S-curve around mid grey)
    saturation: float = 0.0      # −1 grey … +1 vivid
    temperature: float = 0.0     # −1 cool … +1 warm
    tint: float = 0.0            # −1 green … +1 magenta
    shadows: float = 0.0         # −1 crush … +1 lift the low end
    highlights: float = 0.0      # −1 pull down … +1 lift the high end
    highlight_rolloff: float = 0.0  # 0 none … 1 filmic muted/compressed highlights

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_adjustments(samples: np.ndarray, adj: Adjustments) -> np.ndarray:
    """Apply the creative grade to ``samples`` (...,3 in [0,1]). Returns a new array; neutral
    settings return the input unchanged (bit-for-bit)."""
    if adj.is_identity():
        return np.asarray(samples, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64).copy()

    # 1. Temperature / tint — gentle per-channel gain (warm = +R −B; magenta = +R +B −G).
    if adj.temperature or adj.tint:
        gain = np.array([
            1.0 + 0.25 * adj.temperature + 0.12 * adj.tint,
            1.0 - 0.18 * adj.tint,
            1.0 - 0.25 * adj.temperature + 0.12 * adj.tint,
        ])
        x = x * gain

    # 2. Shadows / highlights — lift or pull the ends via luma-band masks (keep mids stable).
    if adj.shadows or adj.highlights:
        luma = x @ _LUMA
        if adj.shadows:
            m = 1.0 - _smoothstep(0.0, 0.5, luma)        # 1 in shadows → 0 by mid
            x = x + (adj.shadows * 0.25 * m)[..., None]
        if adj.highlights:
            m = _smoothstep(0.5, 1.0, luma)              # 0 at mid → 1 in highlights
            x = x + (adj.highlights * 0.25 * m)[..., None]

    # 3. Contrast — S-curve around the mid-grey pivot.
    if adj.contrast:
        slope = 1.0 + adj.contrast
        x = (x - _PIVOT) * slope + _PIVOT

    # 4. Highlight roll-off — soft filmic shoulder above a knee (muted, compressed highlights).
    if adj.highlight_rolloff:
        knee = 0.7
        r = float(np.clip(adj.highlight_rolloff, 0.0, 1.0))
        over = np.maximum(x - knee, 0.0)
        # compress the over-knee range; strength r picks how hard the shoulder is
        comp = over / (1.0 + (r * 2.5) * over / max(1.0 - knee, 1e-6))
        x = np.where(x > knee, knee + comp, x)

    # 5. Saturation — scale chroma around luma (last, so contrast/temp inform it).
    if adj.saturation:
        luma = (x @ _LUMA)[..., None]
        x = luma + (1.0 + adj.saturation) * (x - luma)

    return x
