"""mid — the MVP Look Fitter (baseline tier: per-channel CDF match + saturation).

@context  Turns a ConsensusLook into a LookTransform by per-channel histogram (CDF) matching the
          fixed neutral base toward the references' tonal shape, plus a global saturation match.
          Predictable, smooth, monotone — Plan §0/§2 baseline tier. Rich (OT) replaces it later
          behind the same interface.
@done     MidFitter.fit + _MidLookTransform (per-channel quantile curve + chroma scale +
          tone_strength luma preservation).
@todo     Band-specific hue balance / palette (deferred to Rich).
@limits   PURE numeric (fit reads the fixed base via load_base). Rec.709 g2.4 space. Monotone
          curves (no inversion); final clamp left to regularize. tone_strength<1 keeps the
          references' COLOR character while preserving the input exposure (avoids over-darkening
          when refs are dark/low-key; see BUGS L-003).
@affects  Implements fitter/interface.LookFitter. Consumes ConsensusLook; output sampled by the
          engine + blended by strength. See ADR-0005 + Plan/30_LOOK_FITTER.md §2.
"""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import load_base
from lutgen.orchestration.consensus import ConsensusLook
from lutgen.orchestration.stats import LUMA_WEIGHTS, compute_stats

from .interface import LookTransform

_SAT_SCALE_MAX = 4.0  # guard against blow-up when the base is near-neutral
_DEFAULT_TONE_STRENGTH = 0.6  # how much of the references' tonal/exposure shape to impose


def _strictly_increasing(xp: np.ndarray) -> np.ndarray:
    """Make a quantile vector strictly increasing so np.interp is well-defined (break ties)."""
    inc = np.maximum.accumulate(xp.astype(np.float64))
    return inc + np.arange(inc.size) * 1e-9


class _MidLookTransform:
    """Callable neutral_rgb -> looked_rgb: per-channel CDF curve, partial tonal (luma) shift, and
    chroma (saturation) scale. ``tone_strength`` (0..1) sets how much of the references' exposure
    is imposed: 1 = full tonal match, 0 = keep the input exposure (color cast only)."""

    def __init__(self, source_q, target_q, sat_scale, tone_strength):
        self._src = [_strictly_increasing(source_q[c]) for c in range(3)]
        self._tgt = target_q
        self._sat = float(sat_scale)
        self._tone = float(tone_strength)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        curved = np.empty_like(rgb)
        for c in range(3):
            curved[..., c] = np.interp(rgb[..., c], self._src[c], self._tgt[c])
        luma_in = rgb @ LUMA_WEIGHTS
        luma_curved = curved @ LUMA_WEIGHTS
        # partial tonal shift: keep input exposure, move toward the look's tone by tone_strength
        target_luma = luma_in + self._tone * (luma_curved - luma_in)
        chroma = curved - luma_curved[..., None]          # the look's color cast/character
        return target_luma[..., None] + chroma * self._sat


class MidFitter:
    """Baseline Look Fitter (ADR-0005/0008). `fit(consensus) -> LookTransform`.

    ``tone_strength`` (0..1, default 0.6) controls how much of the references' exposure is
    imposed — lower preserves the input brightness while keeping the color cast (avoids
    over-darkening with dark refs)."""

    def __init__(self, tone_strength: float = _DEFAULT_TONE_STRENGTH):
        self._tone = float(np.clip(tone_strength, 0.0, 1.0))

    def fit(self, consensus: ConsensusLook, source_samples=None) -> LookTransform:
        # source = the neutral distribution the look maps FROM: the protected base by default
        # (replace-Node-2 mode), or the DWG/DI identity grid for the log-space look (ADR-0009).
        base = load_base() if source_samples is None else np.asarray(source_samples, dtype=np.float64)
        source = compute_stats(base)
        src_sat = max(source.saturation_global, 1e-6)
        sat_scale = float(np.clip(consensus.saturation_global / src_sat, 0.0, _SAT_SCALE_MAX))
        return _MidLookTransform(
            source_q=source.channel_quantiles,
            target_q=consensus.channel_quantiles,
            sat_scale=sat_scale,
            tone_strength=self._tone,
        )
