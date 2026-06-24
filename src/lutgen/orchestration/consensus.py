"""consensus — fuse N per-image stats into one robust ConsensusLook (L3, pure).

@context  The fitter-agnostic contract between L3 and L2 (Mid + Rich both consume it). Robustly
          combines references: median across images (outlier-safe), with cross-reference
          variance turned into per-trait confidence (low variance = look, high = content).
@done     ConsensusLook + build_consensus(list[ImageStats]).
@todo     Palette/hue summary for Rich (deferred, matches stats.py).
@limits   PURE: no IO. Median fuse; needs >= 1 ImageStats.
@affects  Input from orchestration/stats.py; consumed by L2 fitter (M3). Stable contract —
          change shape only with care. See codemap/INDEX.md + Plan/30_LOOK_FITTER.md §1 + ADR-0004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import ImageStats


@dataclass
class ConsensusLook:
    """The distilled target look across N references — fitter-agnostic (Plan §1)."""

    channel_quantiles: np.ndarray   # (3, Q) median per-channel tonal shape
    band_balance: np.ndarray        # (3, 3) median color cast by luma band
    band_saturation: np.ndarray     # (3,) median saturation per band
    saturation_global: float
    black_point: float
    white_point: float
    mean: np.ndarray                # (3,) target RGB mean (OT, ADR-0010)
    covariance: np.ndarray          # (3, 3) target RGB covariance (palette, OT)
    mean_oklab: np.ndarray          # (3,) target Oklab mean (perceptual OT, ADR-0011)
    cov_oklab: np.ndarray           # (3, 3) target Oklab covariance
    confidence: dict                # per-trait weight in [0,1] from cross-ref variance
    n_refs: int


def _confidence(stack: np.ndarray) -> float:
    """Map cross-reference variance of a trait to a weight in (0, 1] (1 = perfectly consistent)."""
    var = float(np.mean(np.var(stack, axis=0)))
    return 1.0 / (1.0 + var)


def build_consensus(stats: list[ImageStats]) -> ConsensusLook:
    """Fuse per-image :class:`ImageStats` into one :class:`ConsensusLook` via robust median."""
    if not stats:
        raise ValueError("need at least one ImageStats")

    cq = np.stack([s.channel_quantiles for s in stats], axis=0)
    bb = np.stack([s.band_balance for s in stats], axis=0)
    bs = np.stack([s.band_saturation for s in stats], axis=0)
    sg = np.array([s.saturation_global for s in stats])
    bp = np.array([s.black_point for s in stats])
    wp = np.array([s.white_point for s in stats])
    means = np.stack([s.mean for s in stats], axis=0)
    covs = np.stack([s.covariance for s in stats], axis=0)
    cov_agg = covs.mean(axis=0)            # average covariances (stays PSD)
    cov_agg = 0.5 * (cov_agg + cov_agg.T)  # symmetrize
    means_ok = np.stack([s.mean_oklab for s in stats], axis=0)
    covs_ok = np.stack([s.cov_oklab for s in stats], axis=0).mean(axis=0)
    covs_ok = 0.5 * (covs_ok + covs_ok.T)

    confidence = {
        "tone": _confidence(cq),
        "balance": _confidence(bb),
        "saturation": _confidence(np.concatenate([bs, sg[:, None]], axis=1)),
        "endpoints": _confidence(np.stack([bp, wp], axis=1)),
    }

    return ConsensusLook(
        channel_quantiles=np.median(cq, axis=0),
        band_balance=np.median(bb, axis=0),
        band_saturation=np.median(bs, axis=0),
        saturation_global=float(np.median(sg)),
        black_point=float(np.median(bp)),
        white_point=float(np.median(wp)),
        mean=np.median(means, axis=0),
        covariance=cov_agg,
        mean_oklab=np.median(means_ok, axis=0),
        cov_oklab=covs_ok,
        confidence=confidence,
        n_refs=len(stats),
    )
