"""fit — staged bounded fit of the v3 FilmModel to pooled reference targets.

@context  The heart of the engine (ADR-0001 b1.4, spec §5; rebuilt over Block F in
          ADR-0008 b8.4): find the film-shaped parameters whose transform best explains
          the reference pool. Method: synthesize a deterministic "source world" sample
          cloud from the source targets (neutral prior by default), push it display -> DI
          (inverse base) -> FilmModel -> display (base), measure the same statistics
          poolstats measures, and least-squares them against the reference targets.
          STAGED (non-negotiable): G+F tonal core first (started AT and anchored TO the
          film-print character preset — weak pool ships film, not identity), then
          crosstalk, then coupling+hue curve, then polish. satluma and the legacy display
          S-curves are NOT fitted (v3): sat-vs-luma emerges from the print slope.
@done     FitOptions, FitResult, fit_film_model (4 staged bounded scipy least_squares),
          synth_samples (deterministic source cloud), exposure alignment of the prior world
          (ADR-0003: scene brightness is content — the assumed world's median exposure is
          matched to the pool's so tone curves learn SHAPE; real source pools never rescaled).
          Fit-v2 losses (ADR-0008 b8.3, active in the b8.4 layout): tail-weighted tone
          quantiles, saturation-distribution shape residual (level-free tail/median
          ratios, ref p95 tile-debiased, asymmetric under-sat, polish stage), per-band
          Hunt compensation. b8.4 wiring: 5 stages over Block F — tone (G + neg + print +
          printer lights, preset x0/anchor, confidence-aware knee ridge), crosstalk,
          SYMMETRIC coupling (3 pair params — the asymmetric part is a hue rotation
          invisible to chroma stats), Fourier hue curve, polish.
@todo     b8.5 acceptance on real pools (user eyeball gate).
          Outlier frame down-weighting (poolstats @todo) — evaluate on real pools.
@limits   Conditional colour (ADR-0006): band_mean_ab residuals in every stage teach the
          "shadows cool / highlights warm" behaviour; exposure carries its own strong ridge
          so uniform darkening only wins when nothing conditional explains the data.
          PURE numeric: no file IO beyond the bundled base assets via engine/base (allowed by
          the Golden Rule). Deterministic: seeded cloud + trf least squares. Per-region
          regularization: thin bins (low band/zone weight) get their target residuals damped
          by conf = sqrt(w/(w+w0)) AND every param carries a ridge pull to neutral — unused
          capacity relaxes to identity instead of fitting noise (spec §5). Bounds guarantee
          monotonic curves (tangents <= 2 < 3 = Fritsch limit) and a diagonal-dominant matrix.
@affects  Consumes orchestration/poolstats (targets) + filmmodel (forward) + engine base/apply.
          Consumed by the learn CLI (1.6) and profile save (1.5). See ADR-0001, spec §5.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
from lutgen.engine.grid import reshape_to_lattice
from lutgen.engine.perceptual import from_oklab, to_oklab
from lutgen.fitter.filmmodel import (
    CouplingParams,
    CrosstalkParams,
    FilmModel,
    FilmSystemParams,
    FourierHueParams,
    GlobalParams,
    NegativeParams,
    PrinterLights,
    PrintParams,
    film_print_character,
)
from lutgen.orchestration.poolstats import (
    HUE_BIN_CENTERS,
    L_BAND_EDGES,
    QUANTILES,
    PooledTargets,
    compute_frame_stats,
    neutral_prior,
)

ProgressFn = Callable[[str], None]

# Stage names (also what the progress callback receives, spec §9 "Fitting tone…").
STAGES = ("tone", "crosstalk", "coupling", "huesat", "polish")


@dataclass
class FitOptions:
    """Knobs for the staged fit. Defaults are the production values; tests shrink n_samples."""

    n_samples: int = 3000        # source-cloud size (statistics are heavily over-determined)
    seed: int = 0                # cloud seed — fixed => deterministic fit
    w0: float = 0.02             # thin-data knee: conf = sqrt(w/(w+w0))
    ridge_tone: float = 0.03     # pull-to-anchor strength (v3: anchor = film preset).
                                 # 0.05 held the preset shoulder against a well-measured
                                 # identity world (6% highlight bias); 0.03 lets thick
                                 # evidence win while thin pools still relax to film
    ridge_crosstalk: float = 0.15
    ridge_huesat: float = 0.05  # loosened in ADR-0007 — the §6 validator is the safety gate
    ridge_coupling: float = 0.3  # DIR coupling anchored to the film-print preset (ADR-0008)
    quantile_weight: float = 1.0
    balance_weight: float = 2.0
    band_balance_weight: float = 3.0  # per-band a/b targets (ADR-0006 A) — the conditional
                                      # "shadows cool / highlights warm" signal, every stage
    ridge_exposure: float = 0.4       # ADR-0006 C: plain global darkening must cost more than
                                      # the conditional explanation
    chroma_weight: float = 2.0
    zone_weight: float = 1.5
    tone_anchor_weight: float = 0.3   # stage 2/3: soft anchor keeping stage-1 tone in place
    max_nfev: int | None = None       # per stage; None = scipy default
    diff_step: float = 0.02           # finite-difference secant step: statistics of a
                                      # finite sample cloud carry O(1/n) quantization;
                                      # a wide secant averages over it (ADR-0007)
    # ── fit-v2 losses (ADR-0008 b8.3, ACTIVATED in the b8.4 Block-F stage layout) ──
    tail_weight: float = 2.0          # tone-quantile tail boost: extremes weigh up to this
                                      # factor — film's identity lives at toe/shoulder.
                                      # Primary tone stages only (anchors stay uniform).
    spread_weight: float = 1.5        # saturation-DISTRIBUTION shape residual (Hasler:
                                      # punch = tails vs median, level-free ratios) —
                                      # tone + polish stages, where print slope (the punch
                                      # mechanism) is the parameter being fitted
    hunt_alpha: float = 0.15          # Hunt-effect compensation: perceived colourfulness
                                      # rises with luminance — per-BAND sat is weighted by
                                      # (1 + alpha*(bandL - 0.45)) before shape comparison
    under_sat: float = 2.0            # asymmetry on the spread residual only: losing tail
                                      # saturation costs this x more than gaining it
                                      # (level residuals stay symmetric — vividness contract)
    loss: str = "linear"              # scipy loss. soft_l1 was trialled per ADR-0008 and
                                      # REJECTED as default: its Jacobian reweighting
                                      # crushes the small hue-angle residuals (hue-twist
                                      # recovery corr 0.92 -> 0.38 measured); median
                                      # pooling already provides the outlier robustness
    f_scale: float = 0.3              # knee if a robust loss is chosen explicitly
    exposure_align: bool = True       # prior path only. ON (default): the source world adopts
                                      # the pool's tone distribution, so curves learn the tone
                                      # SHAPE valid at the footage's own level — coherent with
                                      # the ship-without-brightness default (fitting curves
                                      # around a learned exposure and then stripping it left
                                      # them in the wrong domain: dark mids + lifted shadows,
                                      # the user's "wash"). OFF: full-mood fit for users who
                                      # bake film brightness.


@dataclass
class FitResult:
    """The fitted model + per-stage diagnostics (fit-quality metadata for the Look Profile)."""

    model: FilmModel
    stage_cost: dict = field(default_factory=dict)      # stage -> final 0.5*sum(residual^2)
    stage_nfev: dict = field(default_factory=dict)      # stage -> function evaluations
    n_frames: int = 0                                   # frames behind the targets


# ── exposure alignment (ADR-0003 R.2) ─────────────────────────────────

def _exposure_aligned(src: PooledTargets, ref: PooledTargets) -> PooledTargets:
    """Give the assumed source world the POOL'S OWN luma distribution (neutral: identical per
    channel). Scene brightness — including its whole distribution shape, not just a level —
    is content, not grade; a screenshot-only pool cannot reveal the absolute tone reshape
    (spec's hard truth), and pretending it can slams the tone stage into its bounds
    (observed twice on a real pool: slope 2.00/0.70 pinned, then 0.50 after naive scaling).
    What REMAINS learnable — and now is exactly what the tone stage fits — is the per-channel
    DEVIATION from the common luma curve (film's channel-different tone drift), plus every
    colour block on top, all at matched exposure per band."""
    aligned = replace(src)
    luma_q = ref.channel_quantiles.mean(axis=0)      # the pool's tone distribution
    aligned.channel_quantiles = np.tile(np.clip(luma_q, 0.0, 1.0), (3, 1))
    return aligned


# ── source cloud ──────────────────────────────────────────────────────

def synth_samples(targets: PooledTargets, n: int, seed: int = 0) -> np.ndarray:
    """Deterministic display-space sample cloud whose statistics approximate ``targets``.

    Stratified: luma from the pooled tone distribution (inverse CDF), chroma from the per-band
    means with spread, hue from the zone weights around each zone centre; the remaining share
    is achromatic. This is "the source world" the model transforms during fitting.
    """
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.uniform(0.0, 1.0, n)) / n            # stratified uniform

    # luma: inverse-CDF through the channel-mean quantile curve, mapped to Oklab L via the
    # gray axis (display value v -> L of (v,v,v)).
    value_q = targets.channel_quantiles.mean(axis=0)
    v = np.interp(u, QUANTILES, value_q)
    gray_axis = np.linspace(0.0, 1.0, 64)
    gray_l = to_oklab(np.tile(gray_axis[:, None], (1, 3)))[:, 0]
    luma = np.interp(v, gray_axis, gray_l)

    band = np.digitize(luma, L_BAND_EDGES)
    chroma = targets.chroma_by_band[band] * rng.uniform(0.4, 1.6, n)

    # hue from the pool's fine 12-bin distribution + in-bin jitter: CONTINUOUS wheel
    # coverage (the old 6-zone-cluster sampling left half the hue bins empty, making the
    # v2.1 Fourier hue curve unidentifiable — ADR-0007)
    hue_w = targets.hue_weight.copy()
    chromatic_share = float(hue_w.sum())
    bin_width = 2.0 * np.pi / len(hue_w)
    if chromatic_share > 0:
        hue_p = hue_w / chromatic_share
        bins = rng.choice(len(hue_p), size=n, p=hue_p)
        hue = HUE_BIN_CENTERS[bins] + rng.uniform(-bin_width / 2, bin_width / 2, n)
        achromatic = rng.uniform(0.0, 1.0, n) >= min(1.0, chromatic_share)
        chroma = np.where(achromatic, chroma * 0.05, chroma)
    else:                                   # fully achromatic source
        hue = rng.uniform(-np.pi, np.pi, n)
        chroma = chroma * 0.05

    lab = np.column_stack([luma, chroma * np.cos(hue), chroma * np.sin(hue)])
    return np.clip(from_oklab(lab), 0.0, 1.0)


# ── base round-trip (built once per fit) ──────────────────────────────

def _cube_fn(samples: np.ndarray, size: int):
    axis = np.linspace(0.0, 1.0, size)
    lattice = reshape_to_lattice(samples, size)
    interp = RegularGridInterpolator((axis, axis, axis), lattice, method="linear",
                                     bounds_error=False, fill_value=None)
    return lambda x: interp(np.clip(x, 0.0, 1.0)[:, ::-1])


# ── parameter vectors per stage (v3: Block F is the tonal core, ADR-0008) ──

_PRESET = film_print_character()

# tone stage vector: Block G exposure + negative (g_r, g_g, g_b, toe) + print (slope,
# shoulder, ptoe). x0 AND the ridge anchor are the film-print character preset — a weak
# pool relaxes toward FILM, not toward identity (ADR-0008). toe_at / range_hi / range_lo
# stay at the preset's datasheet positions (fitting knee POSITIONS from marginal
# statistics is under-determined; strengths are what the data can support).
_FTONE_X0 = np.array([0.0,
                      _PRESET.negative.g_r, _PRESET.negative.g_g, _PRESET.negative.g_b,
                      _PRESET.negative.toe,
                      _PRESET.printer.slope, _PRESET.printer.shoulder, _PRESET.printer.ptoe,
                      0.0, 0.0, 0.0])                   # printer lights r/g/b (stops)
# physics bounds (research briefs): relative gammas near the LAD window, system contrast
# 0.9-1.8, convergence strengths in [0, 1) where the softplus knees stay monotone,
# printer lights within ±0.3 stop per channel (colour-timing corrections are modest;
# ±0.5 let a blue light of +0.38 paint magenta contamination on the first real pool)
_FTONE_LO = np.array([-0.3, 0.70, 0.70, 0.70, 0.0, 0.90, 0.0, 0.0, -0.3, -0.3, -0.3])
_FTONE_HI = np.array([0.3, 1.35, 1.35, 1.35, 0.8, 1.80, 0.95, 0.95, 0.3, 0.3, 0.3])

_CT_NEUTRAL = np.zeros(6)
_CT_LO, _CT_HI = np.full(6, -0.25), np.full(6, 0.25)

# hue stage vector: DIR coupling (6 suppression amounts >= 0 — saturation-non-decreasing
# by construction) + the Fourier hue curve (23 coefs). satluma is RETIRED from the fit
# (ADR-0008): sat-vs-luma behaviour EMERGES from the print curve's slope profile; the
# block stays in the model for old profiles only.
# Spectral decay: order-k coefficients are bounded by base/k, keeping the curve's
# DERIVATIVE bounded (a steep hue curve compresses hues into delta-E spikes).
_HARM = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])   # k per coef (a0, a1..a4, b1..b4)
_HARM_L = np.array([1.0, 1.0, 2.0, 1.0, 2.0])                     # Block E: l0, lc1, lc2, ls1, ls2
# Coupling is fitted SYMMETRIC (3 pair strengths): its asymmetric part is a hue ROTATION,
# invisible to the chroma statistics this stage fits — left free it floats on noise and
# corrupts the hue curve fitted after it (twist recovery corr -0.20 measured). Symmetric
# suppression is pure separation, zero net rotation; hue personality belongs to the curve.
_COUP_X0 = np.array([(_PRESET.coupling.rg + _PRESET.coupling.gr) / 2,
                     (_PRESET.coupling.rb + _PRESET.coupling.br) / 2,
                     (_PRESET.coupling.gb + _PRESET.coupling.bg) / 2])
_COUP_LO, _COUP_HI = np.zeros(3), np.full(3, 0.15)
_HUE_X0 = np.zeros(23)
_HUE_LO = np.concatenate([-0.12 / _HARM, -0.25 / _HARM, -0.2 / _HARM_L])
_HUE_HI = np.concatenate([0.12 / _HARM, 0.25 / _HARM, 0.2 / _HARM_L])


def _ftone_from_vec(v: np.ndarray, coupling: CouplingParams) -> tuple[GlobalParams, FilmSystemParams]:
    negative = NegativeParams(g_r=v[1], g_g=v[2], g_b=v[3], toe=v[4],
                              toe_at=_PRESET.negative.toe_at)
    printer = PrintParams(slope=v[5], shoulder=v[6], ptoe=v[7],
                          range_hi=_PRESET.printer.range_hi,
                          range_lo=_PRESET.printer.range_lo)
    lights = PrinterLights(r=v[8], g=v[9], b=v[10])
    return GlobalParams(exposure=v[0]), FilmSystemParams(negative, coupling, printer, lights)


def _ct_from_vec(v: np.ndarray) -> CrosstalkParams:
    return CrosstalkParams(rg=v[0], rb=v[1], gr=v[2], gb=v[3], br=v[4], bg=v[5])


def _coup_from_vec(v: np.ndarray) -> CouplingParams:
    return CouplingParams(rg=v[0], gr=v[0], rb=v[1], br=v[1], gb=v[2], bg=v[2])


def _hue_from_vec(v: np.ndarray) -> FourierHueParams:
    # Saturation LEVEL is never learned (ADR-0007): the hue-trim curve's DC term (t0) is
    # zeroed, so hue-to-hue differences survive while the level stays the footage's own.
    coefs = dict(zip(FourierHueParams.field_names(), v, strict=True))
    coefs["t0"] = 0.0                                    # sat level is never learned
    return FourierHueParams(**coefs)


# ── residuals ─────────────────────────────────────────────────────────

def _conf(w: np.ndarray, w0: float) -> np.ndarray:
    return np.sqrt(w / (w + w0))


def _residuals(out_display: np.ndarray, ref: PooledTargets, opt: FitOptions,
               *, tone: bool, balance: bool, chroma: bool, zones: bool,
               hue_luma: bool = False, spread: bool = False,
               tone_weight: float | None = None) -> np.ndarray:
    s = compute_frame_stats(out_display)
    parts = []
    if tone:
        tw = opt.quantile_weight if tone_weight is None else tone_weight
        # tail-weighted quantile residuals (ADR-0008): the toe/shoulder ends of the
        # distribution carry the film's tonal identity — weight rises quadratically
        # from 1 at the median to `tail_weight` at the extremes. PRIMARY tone stages
        # only: in the later stages tone is a soft anchor, and tail-boosting the anchor
        # measurably degraded Block E recovery (b8.3 ablation).
        wq = 1.0 if tone_weight is not None else (
            1.0 + (opt.tail_weight - 1.0) * (2.0 * np.abs(QUANTILES - 0.5)) ** 2)
        parts.append((tw * wq * (s.channel_quantiles - ref.channel_quantiles)).ravel())
    if balance:
        parts.append(opt.balance_weight * (s.mean_lab - ref.mean_lab))
        # conditional balance (ADR-0006 A): colour per luminance band, conf-damped for thin bands
        cb = _conf(ref.band_weight, opt.w0)[:, None]
        parts.append((opt.band_balance_weight * cb *
                      (s.band_mean_ab - ref.band_mean_ab)).ravel())
    if chroma:
        # Saturation SHAPE only (ADR-0007): both sides normalized by their weighted mean —
        # the LEVEL is never learned (web stills under-measure vividness). An asymmetric
        # level residual was trialled and REVERTED: it degraded the verified hue-luminance
        # result (greens 1.145→0.98 vs ref 1.208) without closing the sat gap.
        c = _conf(ref.band_weight, opt.w0)
        w = ref.band_weight / max(ref.band_weight.sum(), 1e-9)
        # Hunt compensation (ADR-0008): perceived colourfulness rises with luminance, so
        # each band's measured saturation is weighted up by its brightness BEFORE the
        # shape normalization — mismatches in bright bands (where perception amplifies
        # them) then cost more. A global Hunt factor would cancel in the normalization;
        # the per-band factor is where the effect is actually expressible.
        band_l = np.concatenate([[L_BAND_EDGES[0] / 2],
                                 (L_BAND_EDGES[:-1] + L_BAND_EDGES[1:]) / 2,
                                 [(L_BAND_EDGES[-1] + 1.0) / 2]])
        hunt = 1.0 + opt.hunt_alpha * (band_l - 0.45)
        rs_h = ref.sat_by_band * hunt
        ss_h = s.sat_by_band * hunt
        rs = rs_h / max(float(w @ rs_h), 1e-9)
        ss = ss_h / max(float(w @ ss_h), 1e-9)
        parts.append(opt.chroma_weight * c * (ss - rs))
    if spread:
        # saturation-DISTRIBUTION shape (ADR-0008, Hasler): tails relative to the median —
        # the punch statistic the per-band means miss. Level-free ratios (vividness
        # contract intact). NO tile debias here: on real frames the max-over-tiles p95
        # reads the single most colourful patch (0.40 vs global 0.23 on the do-revenge
        # pool) and demanding the whole tail reach it drove the fit 2-4x oversaturated
        # with every tone param at its rail — the statistic stays for diagnostics only.
        # Fitted where the punch levers live (tone + polish stages); in stage 3 it
        # drowned the hue personality signal (corr 0.78 -> 0.46, b8.3 ablation).
        r_q = ref.sat_quantiles
        r_shape = r_q / max(r_q[1], 1e-9)                # index 1 = the median
        s_shape = s.sat_quantiles / max(s.sat_quantiles[1], 1e-9)
        d = np.delete(s_shape - r_shape, 1)              # median entry is identically 0
        # asymmetric (ADR-0008): losing tail saturation costs more than gaining it
        d = np.where(d < 0.0, d * np.sqrt(opt.under_sat), d)
        parts.append(opt.spread_weight * d)
    if hue_luma:
        # per-hue LUMINANCE, relative to each side's overall mean L (level-free): how bright
        # the film renders each colour family — the lush-vs-olive greens axis. Fitted in
        # stages 1-2: curves/crosstalk are the only blocks that can move a hue's brightness
        # (stage 3's sat/hue blocks preserve L by construction).
        ch = _conf(ref.hue_weight, opt.w0)
        rl = ref.hue_mean_l / max(ref.mean_lab[0], 0.05)
        sl_ = s.hue_mean_l / max(s.mean_lab[0], 0.05)
        parts.append(1.0 * ch * (sl_ - rl))
    if zones:
        # fine 12-bin hue targets (ADR-0007), compared in HUE-ANGLE and CHROMA-RATIO units —
        # raw a/b differences scale with chroma (~0.05) and drown under the ridge; angle and
        # ratio are unit-compatible with the shift/trim parameters themselves.
        c = _conf(ref.hue_weight, opt.w0)
        # magnitudes normalized by each side's mean L: brightness-invariant per-hue vividness
        mag_r = np.hypot(ref.hue_mean_ab[:, 0], ref.hue_mean_ab[:, 1]) / max(ref.mean_lab[0], 0.05)
        mag_s = np.hypot(s.hue_mean_ab[:, 0], s.hue_mean_ab[:, 1]) / max(s.mean_lab[0], 0.05)
        c = c * (mag_r / (mag_r + 0.01))                # near-achromatic bins: angle is noise
        ang_r = np.arctan2(ref.hue_mean_ab[:, 1], ref.hue_mean_ab[:, 0])
        ang_s = np.arctan2(s.hue_mean_ab[:, 1], s.hue_mean_ab[:, 0])
        d_ang = (ang_s - ang_r + np.pi) % (2.0 * np.pi) - np.pi
        d_mag = (mag_s - mag_r) / np.maximum(mag_r, 0.02)
        cw = c / max(c.sum(), 1e-9)
        d_mag = d_mag - float(cw @ d_mag)                 # hue-RELATIVE vividness only
        parts.append(opt.zone_weight * c * d_ang)
        parts.append(opt.zone_weight * c * d_mag)
        # Block E signal: the same comparison per dark/bright half (half weight each)
        for h in (0, 1):
            c2 = _conf(ref.hue2_weight[h], opt.w0)
            m_r = np.hypot(ref.hue2_mean_ab[h, :, 0], ref.hue2_mean_ab[h, :, 1])
            c2 = c2 * (m_r / (m_r + 0.01))
            a_r = np.arctan2(ref.hue2_mean_ab[h, :, 1], ref.hue2_mean_ab[h, :, 0])
            a_s = np.arctan2(s.hue2_mean_ab[h, :, 1], s.hue2_mean_ab[h, :, 0])
            d2 = (a_s - a_r + np.pi) % (2.0 * np.pi) - np.pi
            parts.append(opt.zone_weight * c2 * d2)
    return np.concatenate(parts)


# ── the staged fit ────────────────────────────────────────────────────

def fit_film_model(ref: PooledTargets, source: PooledTargets | None = None,
                   options: FitOptions | None = None,
                   progress: ProgressFn | None = None) -> FitResult:
    """Fit a FilmModel so that (base ∘ model ∘ inverse-base) applied to the source world
    reproduces the reference pool's statistics. Staged: tone -> crosstalk -> hue/sat detail,
    each stage bounded and ridge-pulled toward neutral (per-region regularization)."""
    opt = options or FitOptions()
    if source is not None:
        src_targets = source                      # a real measured pool is never rescaled
    else:
        src_targets = neutral_prior()
        # the assumed world adopts the POOL's hue-mass structure: constants of the hue curve
        # (s0, l0 …) are invisible in marginal statistics under a uniform wheel — peaks are
        # what make them identifiable. Safe now that no mass residual exists to fight the
        # shift (that combination was tried and removed — see decisions/LOG).
        src_targets.hue_weight = ref.hue_weight.copy()
        if opt.exposure_align:
            src_targets = _exposure_aligned(src_targets, ref)

    # Coherence (ADR-0007/0008): with alignment ON, brightness is CONTENT — exposure is
    # PINNED at 0, not merely ridged. The b8.4 Hunt/spread residuals gave exposure fresh
    # gradients to exploit (a brighter render measures more colourful), and on the first
    # real-pool run the fit brightened the whole world +0.18 DI to fake punch; the shipped
    # look (exposure stripped) then washed — the recurring disease, new variant.
    pin_exposure = opt.exposure_align and source is None
    ftone_lo, ftone_hi = _FTONE_LO.copy(), _FTONE_HI.copy()
    if pin_exposure:
        ftone_lo[0], ftone_hi[0] = -1e-9, 1e-9           # scipy needs lo < hi strictly

    cloud = synth_samples(src_targets, opt.n_samples, opt.seed)
    inv = _cube_fn(load_base_inverse(), INVERSE_SIZE)
    fwd = _cube_fn(load_base(), DEFAULT_SIZE)
    di_cloud = inv(cloud)                                        # constant across evaluations

    result = FitResult(model=FilmModel.identity(), n_frames=ref.n_frames)

    def run_stage(name, x0, lo, hi, ridge, neutral, model_of, res_kw):
        if progress:
            progress(name)
        ridge_v = np.sqrt(np.broadcast_to(np.asarray(ridge, dtype=np.float64), x0.shape))

        def f(v):
            out = fwd(model_of(v).forward(di_cloud))
            data = _residuals(out, ref, opt, **res_kw)
            reg = ridge_v * (v - neutral)
            return np.concatenate([data, reg])

        sol = least_squares(f, x0, bounds=(lo, hi), method="trf",
                            loss=opt.loss, f_scale=opt.f_scale,
                            diff_step=opt.diff_step, max_nfev=opt.max_nfev)
        result.stage_cost[name] = float(sol.cost)
        result.stage_nfev[name] = int(sol.nfev)
        return sol.x

    # Stage 1 — exposure + film system tonal core (Blocks G+F: negative gammas/toe, print
    # slope/shoulder/ptoe). Starts AT the film-print character and is ridge-anchored there:
    # a weak pool ships film, not identity. The spread residual is active here — print
    # slope IS the punch mechanism (sat follows slope), so it finally has its lever.
    tone_ridge = np.full(_FTONE_X0.shape, opt.ridge_tone)
    tone_ridge[0] = opt.ridge_exposure                 # ADR-0006 C: the lazy global darkening
    # CONFIDENCE-AWARE preset anchoring (ADR-0008): the knee strengths (neg toe, shoulder,
    # print toe) act where data is often thin — anchor them to the film preset ONLY to the
    # extent the pool's own tail bands are thin. A pool that HAS measured highlights gets
    # its own shoulder; a night pool keeps film's. This is what makes "weak pool => film"
    # per-REGION instead of a global bias.
    c_lo = float(_conf(np.array([ref.band_weight[0]]), opt.w0)[0])
    c_hi = float(_conf(np.array([ref.band_weight[-1]]), opt.w0)[0])
    tone_ridge[4] *= 1.0 - 0.9 * c_lo                  # negative toe
    tone_ridge[6] *= 1.0 - 0.9 * c_hi                  # print shoulder
    tone_ridge[7] *= 1.0 - 0.9 * c_lo                  # print black convergence
    tone_ridge[8:] = opt.ridge_crosstalk               # printer lights: cast params, same
                                                       # caution as the crosstalk mixer
    # Coupling stays NEUTRAL until its own stage: the preset's asymmetric suppression
    # rotates the hue field, and fitting tone/crosstalk on a pre-rotated world sends the
    # balance signal into the wrong blocks (crosstalk chased a pure hue twist with -0.08
    # entries — measured).
    tone_v = run_stage(
        "tone", _FTONE_X0, ftone_lo, ftone_hi, tone_ridge, _FTONE_X0,
        lambda v: FilmModel(global_trim=_ftone_from_vec(v, CouplingParams())[0],
                            film_system=_ftone_from_vec(v, CouplingParams())[1]),
        {"tone": True, "balance": True, "chroma": True, "zones": False, "spread": True},
    )

    def _fs_of(v, coupling):
        return _ftone_from_vec(v, coupling)

    global_trim, film_system = _fs_of(tone_v, CouplingParams())

    # Stage 2 — crosstalk (Block A: the rendering-primaries mixer in front of the film
    # system — the hue crossovers ARE the look). Tone frozen; quantiles stay a soft anchor.
    ct_v = run_stage(
        "crosstalk", _CT_NEUTRAL, _CT_LO, _CT_HI, opt.ridge_crosstalk, _CT_NEUTRAL,
        lambda v: FilmModel(global_trim=global_trim, crosstalk=_ct_from_vec(v),
                            film_system=film_system),
        {"tone": True, "balance": True, "chroma": False, "zones": False,
         "tone_weight": opt.tone_anchor_weight},  # hue detail belongs to the Fourier stage
    )
    crosstalk = _ct_from_vec(ct_v)

    # Stage 3 — DIR coupling (F.coupling): the density-domain saturation machinery, fitted
    # against the chroma statistics ONLY. Separate from the hue curve — co-fitting the two
    # let coupling corrupt the hue field (twist recovery corr -0.30 measured); one signal,
    # one block is this codebase's hard-won staging rule.
    coup_v = run_stage(
        "coupling", _COUP_X0, _COUP_LO, _COUP_HI, opt.ridge_coupling, _COUP_X0,
        # chroma statistics ONLY (plus the tone anchor): asymmetric coupling rotates the
        # hue field as a side effect, so letting it chase balance targets corrupts the
        # hue curve fitted after it — the mean-colour signal belongs to crosstalk.
        lambda v: FilmModel(global_trim=global_trim, crosstalk=crosstalk,
                            film_system=_fs_of(tone_v, _coup_from_vec(v))[1]),
        {"tone": True, "balance": False, "chroma": True, "zones": False,
         "tone_weight": opt.tone_anchor_weight},
    )
    coupling = _coup_from_vec(coup_v)
    film_system = _fs_of(tone_v, coupling)[1]

    # Stage 4 — hue personality (the Fourier curve). Everything tonal/saturation frozen;
    # the trim coefficients ridge stiffer than shift; higher harmonics ridge ~k^2
    # (curvature prior — the classic smoothness penalty).
    hue_ridge = np.concatenate([opt.ridge_huesat * _HARM ** 2,
                                opt.ridge_huesat * 10.0 * _HARM ** 2,   # trims: see below
                                opt.ridge_huesat * 0.5 * _HARM_L ** 2])  # Block E
    # trims ridge 10x (was 4x): when the pool sits at a different level than the source
    # world, Block F's knees leave per-hue vividness residue that the trims chase as a
    # phantom (tc1 hit 0.24 on the darkened-world guard) — hue-relative sat personality
    # must be cheap only when the signal is strong
    hue_v = run_stage(
        "huesat", _HUE_X0, _HUE_LO, _HUE_HI, hue_ridge, _HUE_X0,
        # zones (per-hue angle/vividness) + balance only: the band-chroma shape belongs
        # to the coupling/tone stages — with satluma retired, a chroma residual here can
        # only be (mis)expressed through the trim coefficients (bound-slams measured)
        lambda v: FilmModel(global_trim=global_trim, crosstalk=crosstalk,
                            film_system=film_system, hue_fourier=_hue_from_vec(v)),
        {"tone": True, "balance": True, "chroma": False, "zones": True,
         "tone_weight": opt.tone_anchor_weight},
    )
    hue_fourier = _hue_from_vec(hue_v)

    # Stage 5 — polish (per-hue LUMINANCE + punch): with the hue field settled, re-fit
    # G+F tone + crosstalk against the hue-luma targets ("lush vs olive greens") and the
    # spread residual. Anchored at the stage-1/2 solution: refinement, not re-opening.
    p0 = np.concatenate([tone_v, ct_v])
    p_lo = np.concatenate([ftone_lo, _CT_LO])           # exposure stays pinned on the
    p_hi = np.concatenate([ftone_hi, _CT_HI])           # aligned path through polish
    p_ridge = np.concatenate([np.full(_FTONE_X0.size, 0.3), np.full(6, 0.3)])

    def _polish_model(v):
        gt, fs = _fs_of(v[:_FTONE_X0.size], coupling)
        return FilmModel(global_trim=gt, crosstalk=_ct_from_vec(v[_FTONE_X0.size:]),
                         film_system=fs, hue_fourier=hue_fourier)

    pol_v = run_stage(
        "polish", p0, p_lo, p_hi, p_ridge, p0,
        _polish_model,
        {"tone": True, "balance": True, "chroma": False, "zones": False, "hue_luma": True,
         "spread": True, "tone_weight": opt.tone_anchor_weight},
    )
    global_trim, film_system = _fs_of(pol_v[:_FTONE_X0.size], coupling)
    crosstalk = _ct_from_vec(pol_v[_FTONE_X0.size:])

    result.model = FilmModel(global_trim=global_trim, crosstalk=crosstalk,
                             film_system=film_system, hue_fourier=hue_fourier)
    if progress:
        progress("done")
    return result
