"""stats — per-image color statistics (L3, pure).

@context  Summarize one reference's color character: tonal shape, color balance by luma band,
          saturation, endpoints. These per-image stats are fused across references by
          consensus.py. All in Rec.709 g2.4 code space (Plan §1).
@done     ImageStats + compute_stats(image).
@todo     Palette/hue summary for the Rich fitter (deferred).
@limits   PURE: no IO. Input (H,W,3) float64 [0,1]. Empty luma bands fall back to the global mean.
@affects  Consumed by orchestration/consensus.py. See codemap/INDEX.md + ADR-0004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lutgen.engine.perceptual import to_oklab

# Fixed probe quantiles for the per-channel tonal curve.
QUANTILES = np.linspace(0.0, 1.0, 11)
# Rec.709 luma weights.
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
# Luma band edges (shadow / mid / highlight).
BAND_EDGES = (1.0 / 3.0, 2.0 / 3.0)
# Per-image Oklab subsample size for PDF transfer (ADR-0013).
SAMPLE_CAP = 2000


@dataclass
class ImageStats:
    """Color summary of one reference image (Rec.709 g2.4)."""

    channel_quantiles: np.ndarray   # (3, len(QUANTILES))
    band_balance: np.ndarray        # (3 bands, 3 channels) mean RGB per luma band
    band_saturation: np.ndarray     # (3,) mean saturation per luma band
    saturation_global: float
    black_point: float              # luma p1
    white_point: float              # luma p99
    mean: np.ndarray                # (3,) RGB mean (for OT, ADR-0010)
    covariance: np.ndarray          # (3, 3) RGB covariance (palette, for OT)
    mean_oklab: np.ndarray          # (3,) Oklab mean (perceptual OT, ADR-0011)
    cov_oklab: np.ndarray           # (3, 3) Oklab covariance
    oklab_samples: np.ndarray       # (k, 3) Oklab pixel subsample (PDF transfer target, ADR-0013)


def _saturation(pixels: np.ndarray) -> np.ndarray:
    cmax = pixels.max(axis=1)
    cmin = pixels.min(axis=1)
    return (cmax - cmin) / np.maximum(cmax, 1e-6)


def compute_stats(image: np.ndarray) -> ImageStats:
    """Compute :class:`ImageStats` from an ``(H, W, 3)`` (or ``(N, 3)``) image in [0, 1]."""
    pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3)
    if pixels.shape[0] == 0:
        raise ValueError("empty image")

    channel_quantiles = np.stack(
        [np.quantile(pixels[:, c], QUANTILES) for c in range(3)], axis=0
    )

    luma = pixels @ LUMA_WEIGHTS
    lab = to_oklab(pixels)
    sat = _saturation(pixels)
    global_mean = pixels.mean(axis=0)
    global_sat = float(sat.mean())

    masks = (
        luma < BAND_EDGES[0],
        (luma >= BAND_EDGES[0]) & (luma < BAND_EDGES[1]),
        luma >= BAND_EDGES[1],
    )
    band_balance = np.empty((3, 3))
    band_saturation = np.empty(3)
    for i, m in enumerate(masks):
        if m.any():
            band_balance[i] = pixels[m].mean(axis=0)
            band_saturation[i] = sat[m].mean()
        else:  # empty band → fall back to global so downstream never sees NaN
            band_balance[i] = global_mean
            band_saturation[i] = global_sat

    return ImageStats(
        channel_quantiles=channel_quantiles,
        band_balance=band_balance,
        band_saturation=band_saturation,
        saturation_global=global_sat,
        black_point=float(np.quantile(luma, 0.01)),
        white_point=float(np.quantile(luma, 0.99)),
        mean=global_mean,
        covariance=np.cov(pixels, rowvar=False) if pixels.shape[0] > 1 else np.zeros((3, 3)),
        mean_oklab=lab.mean(axis=0),
        cov_oklab=np.cov(lab, rowvar=False) if pixels.shape[0] > 1 else np.zeros((3, 3)),
        oklab_samples=_subsample(lab, SAMPLE_CAP),
    )


def _subsample(x: np.ndarray, cap: int) -> np.ndarray:
    """Deterministic random subsample of rows (for PDF-transfer targets)."""
    if x.shape[0] <= cap:
        return x.copy()
    idx = np.random.default_rng(0).choice(x.shape[0], cap, replace=False)
    return x[idx]
