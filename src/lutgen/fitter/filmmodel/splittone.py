"""splittone — Block S: the luminance-conditional tint curve (split tone, ADR-0008 b8.5).

@context  The user's colour-mapping verdict on real pools: films paint colour BY LEVEL —
          do-revenge measures a warm ARCH (b: 0.005 / 0.026 / 0.032 / 0.027 / 0.019
          across the five luminance bands, with cool ends) — and no other block can
          express a non-monotone-in-level balance: gammas / crosstalk / printer lights
          all drift monotonically. Block S is the first-class control: an Oklab a/b tint
          at five luminance poles (aligned with poolstats' L bands so the fit's per-band
          balance targets map 1:1), interpolated smoothly over L.
@done     SplitToneParams (5 poles x a/b = 10) + apply_split_tone on code-space Oklab.
@limits   PURE: no IO. L untouched (tone monotonicity preserved); a/b get the pole curve
          interpolated at the pixel's own L (linear between pole centres 0.1..0.9, flat
          beyond — offsets are <= 0.05 Oklab so the piecewise-linear kinks are far below
          the delta-E gate). Neutral (all 0) returns input BIT-FOR-BIT. The fit's skin
          corridor rides on top of whatever this block paints.
@affects  Applied inside model.FilmModel's Oklab section (before satluma/zones/fourier).
          Serialized as "split_tone"; colour dial scales it. See ADR-0008 b8.5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Pole centres — aligned with poolstats' five L bands (edges 0.2/0.4/0.6/0.8).
POLE_CENTERS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


@dataclass
class SplitToneParams:
    """Oklab a/b tint at each of the five luminance poles (0 = neutral)."""

    t0_a: float = 0.0
    t0_b: float = 0.0
    t1_a: float = 0.0
    t1_b: float = 0.0
    t2_a: float = 0.0
    t2_b: float = 0.0
    t3_a: float = 0.0
    t3_b: float = 0.0
    t4_a: float = 0.0
    t4_b: float = 0.0

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())

    def poles(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.t0_a, self.t1_a, self.t2_a, self.t3_a, self.t4_a]),
                np.array([self.t0_b, self.t1_b, self.t2_b, self.t3_b, self.t4_b]))


def apply_split_tone(lab: np.ndarray, p: SplitToneParams) -> np.ndarray:
    """Add the tint curve, evaluated at each pixel's own L, to a/b. Neutral = input."""
    if p.is_identity():
        return lab
    out = np.array(lab, dtype=np.float64)
    L = out[..., 0]
    a_poles, b_poles = p.poles()
    out[..., 1] += np.interp(L, POLE_CENTERS, a_poles)
    out[..., 2] += np.interp(L, POLE_CENTERS, b_poles)
    return out
