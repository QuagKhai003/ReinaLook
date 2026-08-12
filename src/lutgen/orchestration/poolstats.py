"""poolstats — pooled robust reference statistics + neutral prior (v2 Learn-mode targets).

@context  Learn mode's key idea (spec §4): scene content is RANDOM across a pool of frames,
          the grade is CONSTANT — so statistics pooled robustly across frames keep the shared
          transform and average the content out. This module computes the per-frame Oklab
          statistics the staged fit (batch 1.4) targets, pools them with per-bin MEDIANS (one
          unusually colourful frame cannot hijack the fit), and provides the documented
          neutral prior the fit relaxes toward where data is thin.
@done     FrameStats + compute_frame_stats (incl. band_mean_ab — per-band conditional
          balance, ADR-0006); PooledTargets + pool_stats (median per bin);
          neutral_prior() with canonical ungraded-world values.
@todo     Optional outlier frame down-weighting (spec §4 "frame weighting") if median pooling
          proves insufficient on real pools — evaluate during 1.4.
@limits   PURE numeric: no IO (ingest loads files). Input frames are (H,W,3)/(N,3) float64
          [0,1] Rec.709 g2.4 (delivery space of graded stills). Zero-pixel bins carry weight 0
          and a neutral value — the fit's per-region regularization keys off the weights.
          The prior is WEAK regularization only, never the core method (spec §4).
@affects  Consumed by fitter/fit.py (batch 1.4). Zone binning matches Block D's ZONE_ANGLES
          (fitter/filmmodel/huezone.py) so targets and model speak the same zones. See ADR-0001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lutgen.engine.perceptual import to_oklab
from lutgen.fitter.filmmodel.huezone import ZONE_ANGLES, ZONE_NAMES

# Tone-curve targets: quantile probe points per RGB channel (denser than v1's 11 — the fit
# matches whole cumulative distributions, spec §5).
QUANTILES = np.linspace(0.0, 1.0, 21)
# Oklab-L band edges for chroma-vs-luma targets (5 bands).
L_BAND_EDGES = np.array([0.2, 0.4, 0.6, 0.8])
N_BANDS = len(L_BAND_EDGES) + 1
N_ZONES = len(ZONE_NAMES)
# Pixels with chroma below this are achromatic — excluded from hue-zone stats (their hue
# angle is numerical noise).
CHROMA_FLOOR = 0.01
# Fine hue statistics for the v2.1 Fourier hue curve (ADR-0007): 12 uniform hue bins.
N_HUE_BINS = 12
HUE_BIN_CENTERS = -np.pi + (np.arange(N_HUE_BINS) + 0.5) * (2.0 * np.pi / N_HUE_BINS)
_TWO_PI = 2.0 * np.pi


@dataclass
class FrameStats:
    """One reference frame's Learn-mode statistics (all Oklab except the RGB tone quantiles)."""

    channel_quantiles: np.ndarray   # (3, len(QUANTILES)) per-RGB-channel tone distribution
    mean_lab: np.ndarray            # (3,) Oklab mean (colour balance)
    chroma_by_band: np.ndarray      # (N_BANDS,) mean chroma per L band
    band_mean_ab: np.ndarray        # (N_BANDS, 2) mean Oklab a/b per L band — the CONDITIONAL
                                    # balance ("shadows cool, highlights warm"), ADR-0006
    band_weight: np.ndarray         # (N_BANDS,) pixel share per L band (sums to 1)
    zone_mean_ab: np.ndarray        # (N_ZONES, 2) mean (a,b) per hue zone (chromatic px only)
    zone_weight: np.ndarray         # (N_ZONES,) chromatic-pixel share per zone (sums to <= 1)
    hue_mean_ab: np.ndarray         # (N_HUE_BINS, 2) mean (a,b) per fine hue bin (ADR-0007)
    hue_weight: np.ndarray          # (N_HUE_BINS,) chromatic-pixel share per bin
    black_point: float              # Oklab L p1
    white_point: float              # Oklab L p99


@dataclass
class PooledTargets:
    """Robust pool of FrameStats — the fit's targets. Same shapes as FrameStats plus counts.

    ``band_weight``/``zone_weight`` are the pool means — the per-region data thickness the
    fit's regularization keys off (thin bin -> relax toward the prior). ``n_frames`` drives
    the frame-count quality guidance (spec §4).
    """

    channel_quantiles: np.ndarray
    mean_lab: np.ndarray
    chroma_by_band: np.ndarray
    band_mean_ab: np.ndarray
    band_weight: np.ndarray
    zone_mean_ab: np.ndarray
    zone_weight: np.ndarray
    hue_mean_ab: np.ndarray
    hue_weight: np.ndarray
    black_point: float
    white_point: float
    n_frames: int


def _zone_index(hue: np.ndarray) -> np.ndarray:
    """Assign each hue angle to its nearest zone centre (angular distance)."""
    diff = np.abs((hue[:, None] - ZONE_ANGLES[None, :] + np.pi) % _TWO_PI - np.pi)
    return np.argmin(diff, axis=1)


def compute_frame_stats(image: np.ndarray) -> FrameStats:
    """Compute :class:`FrameStats` from one ``(H,W,3)`` (or ``(N,3)``) g2.4 image in [0,1]."""
    pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3)
    if pixels.shape[0] == 0:
        raise ValueError("empty image")

    channel_quantiles = np.stack(
        [np.quantile(pixels[:, c], QUANTILES) for c in range(3)], axis=0
    )

    lab = to_oklab(pixels)
    luma = lab[:, 0]
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    n = float(pixels.shape[0])

    band_idx = np.digitize(luma, L_BAND_EDGES)
    chroma_by_band = np.zeros(N_BANDS)
    band_mean_ab = np.zeros((N_BANDS, 2))
    band_weight = np.zeros(N_BANDS)
    for b in range(N_BANDS):
        m = band_idx == b
        band_weight[b] = m.sum() / n
        if m.any():
            chroma_by_band[b] = chroma[m].mean()
            band_mean_ab[b] = lab[m, 1:].mean(axis=0)

    chromatic = chroma >= CHROMA_FLOOR
    zone_mean_ab = np.zeros((N_ZONES, 2))
    zone_weight = np.zeros(N_ZONES)
    hue_mean_ab = np.zeros((N_HUE_BINS, 2))
    hue_weight = np.zeros(N_HUE_BINS)
    if chromatic.any():
        hue = np.arctan2(lab[chromatic, 2], lab[chromatic, 1])
        zidx = _zone_index(hue)
        ab = lab[chromatic, 1:]
        for z in range(N_ZONES):
            m = zidx == z
            zone_weight[z] = m.sum() / n
            if m.any():
                zone_mean_ab[z] = ab[m].mean(axis=0)
        # SOFT (triangular) assignment over the two nearest bin centres: the statistics are
        # then smooth functions of pixel hue, which keeps the fit's finite-difference
        # Jacobian consistent (hard binning made stage 3 unoptimizable — ADR-0007).
        bin_w = _TWO_PI / N_HUE_BINS
        d = np.abs((hue[:, None] - HUE_BIN_CENTERS[None, :] + np.pi) % _TWO_PI - np.pi)
        w = np.maximum(0.0, 1.0 - d / bin_w)            # (n_chromatic, N_HUE_BINS), rows sum ~1
        mass = w.sum(axis=0)
        hue_weight = mass / n
        safe = np.maximum(mass, 1e-9)[:, None]
        hue_mean_ab = (w.T @ ab) / safe
        hue_mean_ab[mass < 1e-9] = 0.0

    return FrameStats(
        channel_quantiles=channel_quantiles,
        mean_lab=lab.mean(axis=0),
        chroma_by_band=chroma_by_band,
        band_mean_ab=band_mean_ab,
        band_weight=band_weight,
        zone_mean_ab=zone_mean_ab,
        zone_weight=zone_weight,
        hue_mean_ab=hue_mean_ab,
        hue_weight=hue_weight,
        black_point=float(np.quantile(luma, 0.01)),
        white_point=float(np.quantile(luma, 0.99)),
    )


def pool_stats(frames: list[FrameStats]) -> PooledTargets:
    """Pool per-frame stats with per-bin MEDIANS (values) and MEANS (weights).

    Median per statistic bin is the robust-pooling lever from spec §4: whatever colour
    behaviour the majority of frames share survives; a single outlier frame's excursions do
    not. Weights average, so a zone only most frames saw still carries proportional thickness.
    """
    if not frames:
        raise ValueError("need at least one frame")
    med = lambda attr: np.median(np.stack([getattr(f, attr) for f in frames]), axis=0)
    mean = lambda attr: np.mean(np.stack([getattr(f, attr) for f in frames]), axis=0)
    return PooledTargets(
        channel_quantiles=med("channel_quantiles"),
        mean_lab=med("mean_lab"),
        chroma_by_band=med("chroma_by_band"),
        band_mean_ab=med("band_mean_ab"),
        band_weight=mean("band_weight"),
        zone_mean_ab=med("zone_mean_ab"),
        zone_weight=mean("zone_weight"),
        hue_mean_ab=med("hue_mean_ab"),
        hue_weight=mean("hue_weight"),
        black_point=float(np.median([f.black_point for f in frames])),
        white_point=float(np.median([f.white_point for f in frames])),
        n_frames=len(frames),
    )


def neutral_prior() -> PooledTargets:
    """The canonical "ungraded world" (spec §4): average colour near gray, moderate saturation,
    blacks near black. The fit pulls toward this wherever pooled evidence is thin — WEAK
    regularization only. Values are documented heuristics, not measurements:

    - tone quantiles: gently midtone-weighted ramp from near-black to near-white,
    - balance: neutral gray at L 0.45 (a=b=0),
    - chroma: moderate in the mids, naturally lower in deep shadow / bright highlight,
    - zones: equal thickness, each zone's mean colour sitting at moderate chroma 0.06
      in its own centre direction (i.e. "no hue twist"),
    - black ~0.02 / white ~0.95.
    """
    ramp = 0.02 + 0.88 * QUANTILES**1.2
    zone_dirs = np.column_stack([np.cos(ZONE_ANGLES), np.sin(ZONE_ANGLES)])
    return PooledTargets(
        channel_quantiles=np.tile(ramp, (3, 1)),
        mean_lab=np.array([0.45, 0.0, 0.0]),
        chroma_by_band=np.array([0.03, 0.05, 0.06, 0.05, 0.035]),
        band_mean_ab=np.zeros((N_BANDS, 2)),   # gray world at every brightness
        band_weight=np.full(N_BANDS, 1.0 / N_BANDS),
        zone_mean_ab=0.06 * zone_dirs,
        zone_weight=np.full(N_ZONES, 1.0 / N_ZONES),
        hue_mean_ab=0.06 * np.column_stack([np.cos(HUE_BIN_CENTERS), np.sin(HUE_BIN_CENTERS)]),
        hue_weight=np.full(N_HUE_BINS, 1.0 / N_HUE_BINS),
        black_point=0.02,
        white_point=0.95,
        n_frames=0,
    )
