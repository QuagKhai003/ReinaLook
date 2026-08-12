"""fourierhue — Block D v2: smooth Fourier hue personality (replaces the 6 chunky zones).

@context  A film's hue behaviour is a smooth detailed curve over the hue wheel, not six
          plateaus (spec §3 v2.1: "the single most worthwhile upgrade"). Hue shift and sat
          trim are order-4 Fourier series over hue angle — 9 coefficients each, C-infinity
          smooth and periodic by construction: no zone boundaries exist to break at.
          BLOCK E (b7.2): the hue shift gains a brightness-linear modulation
          shift(θ, L) = F0(θ) + (L − 0.5)·F1(θ), F1 an order-2 series (5 coefs) — the
          proper split-tone generalization ("blues go teal in the mids only", spec §2.4).
@done     FourierHueParams (18 + 5 luma-mod coefficients) + apply_fourier_hue on Oklab;
          eval_shift(hue, L)/eval_trim/eval_lshift (fit, recipe summary, tests).
@todo     -
@limits   PURE: no IO. Operates on Oklab; rotates hue / scales chroma — L untouched; the
          chroma multiplier clamps >= 0; achromatic pixels are fixed points. Neutral (all 0)
          returns the input BIT-FOR-BIT. Coefficient bounds live in the fit (±0.12 rad shift,
          ±0.25 trim per coefficient).
@affects  Applied inside model.FilmModel after the legacy zone trims (old profiles keep
          rendering; new fits leave zones neutral). Serialized as "hue_fourier". ADR-0007.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

ORDER = 4   # Fourier order: coefficients a0 + a1..a4 cos + b1..b4 sin = 9 per attribute


@dataclass
class FourierHueParams:
    """Hue-shift (``s*``, radians) and sat-trim (``t*``, fraction) Fourier coefficients over
    the Oklab hue angle: value(θ) = x0 + Σ_k xc_k·cos(kθ) + xs_k·sin(kθ). All default 0."""

    s0: float = 0.0
    sc1: float = 0.0
    sc2: float = 0.0
    sc3: float = 0.0
    sc4: float = 0.0
    ss1: float = 0.0
    ss2: float = 0.0
    ss3: float = 0.0
    ss4: float = 0.0
    t0: float = 0.0
    tc1: float = 0.0
    tc2: float = 0.0
    tc3: float = 0.0
    tc4: float = 0.0
    ts1: float = 0.0
    ts2: float = 0.0
    ts3: float = 0.0
    ts4: float = 0.0
    # Block E: brightness-linear hue-shift modulation F1(θ), order 2 (ADR-0007 b7.2)
    l0: float = 0.0
    lc1: float = 0.0
    lc2: float = 0.0
    ls1: float = 0.0
    ls2: float = 0.0

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


def _series(hue: np.ndarray, c0: float, cos_c: np.ndarray, sin_c: np.ndarray) -> np.ndarray:
    out = np.full_like(hue, c0)
    for k in range(1, ORDER + 1):
        out += cos_c[k - 1] * np.cos(k * hue) + sin_c[k - 1] * np.sin(k * hue)
    return out


def eval_shift(hue: np.ndarray, p: FourierHueParams,
               luma: np.ndarray | float = 0.5) -> np.ndarray:
    """Hue shift (radians) at hue angle(s) — brightness-modulated (Block E):
    shift(θ, L) = F0(θ) + (L − 0.5)·F1(θ). At L = 0.5 the modulation vanishes."""
    hue = np.asarray(hue, dtype=np.float64)
    base = _series(hue, p.s0,
                   np.array([p.sc1, p.sc2, p.sc3, p.sc4]),
                   np.array([p.ss1, p.ss2, p.ss3, p.ss4]))
    return base + (np.asarray(luma, dtype=np.float64) - 0.5) * eval_lshift(hue, p)


def eval_lshift(hue: np.ndarray, p: FourierHueParams) -> np.ndarray:
    """F1(θ): the per-stop-of-brightness hue-shift modulation (radians per unit L)."""
    hue = np.asarray(hue, dtype=np.float64)
    out = np.full_like(hue, p.l0)
    out += p.lc1 * np.cos(hue) + p.ls1 * np.sin(hue)
    out += p.lc2 * np.cos(2 * hue) + p.ls2 * np.sin(2 * hue)
    return out


def eval_trim(hue: np.ndarray, p: FourierHueParams) -> np.ndarray:
    """Sat trim (fraction; multiplier is 1 + trim) at hue angle(s) ``hue``."""
    return _series(np.asarray(hue, dtype=np.float64), p.t0,
                   np.array([p.tc1, p.tc2, p.tc3, p.tc4]),
                   np.array([p.ts1, p.ts2, p.ts3, p.ts4]))


def apply_fourier_hue(lab: np.ndarray, p: FourierHueParams) -> np.ndarray:
    """Apply the smooth hue personality to Oklab values. New array; neutral params return
    the input unchanged (bit-for-bit). L untouched; chroma-0 pixels unaffected."""
    lab = np.asarray(lab, dtype=np.float64)
    if lab.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {lab.shape}")
    if p.is_identity():
        return lab.copy()

    a, b = lab[..., 1], lab[..., 2]
    hue = np.arctan2(b, a)
    chroma = np.hypot(a, b)
    new_hue = hue + eval_shift(hue, p, lab[..., 0])     # Block E: L-modulated shift
    new_chroma = chroma * np.maximum(1.0 + eval_trim(hue, p), 0.0)
    out = lab.copy()
    out[..., 1] = new_chroma * np.cos(new_hue)
    out[..., 2] = new_chroma * np.sin(new_hue)
    return out
