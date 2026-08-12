"""fit — staged bounded fit of the v2 FilmModel to pooled reference targets.

@context  The heart of v2 (ADR-0001 b1.4, spec §5): find the ~33 film-shaped parameters whose
          transform best explains the reference pool. Method: synthesize a deterministic
          "source world" sample cloud from the source targets (neutral prior by default),
          push it display -> DI (inverse base) -> FilmModel -> display (base), measure the
          same statistics poolstats measures, and least-squares them against the reference
          targets. STAGED (non-negotiable): tone curves first, then crosstalk, then C/D —
          each stage freezes the previous ones. Fitting all params at once is not stable.
@done     FitOptions, FitResult, fit_film_model (3 staged bounded scipy least_squares),
          synth_samples (deterministic source cloud), exposure alignment of the prior world
          (ADR-0003: scene brightness is content — the assumed world's median exposure is
          matched to the pool's so tone curves learn SHAPE; real source pools never rescaled).
@todo     Global exposure/black trims (spec budget) if acceptance shows they're needed.
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
    CrosstalkParams,
    FilmModel,
    FourierHueParams,
    GlobalParams,
    SatLumaParams,
    SCurveParams,
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
STAGES = ("tone", "crosstalk", "huesat", "polish")


@dataclass
class FitOptions:
    """Knobs for the staged fit. Defaults are the production values; tests shrink n_samples."""

    n_samples: int = 3000        # source-cloud size (statistics are heavily over-determined)
    seed: int = 0                # cloud seed — fixed => deterministic fit
    w0: float = 0.02             # thin-data knee: conf = sqrt(w/(w+w0))
    ridge_tone: float = 0.05     # pull-to-neutral strength per stage (per-region reg.)
    ridge_crosstalk: float = 0.15
    ridge_huesat: float = 0.05  # loosened in ADR-0007 — the §6 validator is the safety gate
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


# ── parameter vectors per stage ───────────────────────────────────────

# tone stage vector: global exposure (Block G) + (toe, shoulder, slope, pivot) x RGB
_TONE_NEUTRAL = np.array([0.0] + [0.0, 0.0, 1.0, 0.5] * 3)
_TONE_LO = np.array([-0.3] + [0.0, 0.0, 0.5, 0.3] * 3)
_TONE_HI = np.array([0.3] + [2.0, 2.0, 2.0, 0.7] * 3)
_CT_NEUTRAL = np.zeros(6)
_CT_LO, _CT_HI = np.full(6, -0.25), np.full(6, 0.25)
# satluma x3, then the Fourier hue curve: 9 shift coefs (rad) + 9 trim coefs (ADR-0007;
# the legacy 6-zone params are no longer fitted — they remain for old profiles only).
# Spectral decay: order-k coefficients are bounded by base/k, keeping the curve's DERIVATIVE
# bounded (a steep hue curve compresses hues into delta-E spikes and tone reversals).
_HARM = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])   # k per coef (a0, a1..a4, b1..b4)
_HARM_L = np.array([1.0, 1.0, 2.0, 1.0, 2.0])                     # Block E: l0, lc1, lc2, ls1, ls2
_CD_NEUTRAL = np.array([1.0, 1.0, 1.0] + [0.0] * 23)
# sat-vs-luma bounded to the physically-plausible film range: multipliers near 2x create
# chroma gradients steep enough to reverse channels on saturated ramps (found by the gate)
# sat-vs-luma is a gentle RELATIVE shape only (±30%); level is never learned
_CD_LO = np.concatenate([[0.7, 0.7, 0.7], -0.12 / _HARM, -0.25 / _HARM, -0.2 / _HARM_L])
_CD_HI = np.concatenate([[1.3, 1.3, 1.3], 0.12 / _HARM, 0.25 / _HARM, 0.2 / _HARM_L])


def _tone_from_vec(v: np.ndarray) -> tuple[GlobalParams, tuple[SCurveParams, SCurveParams, SCurveParams]]:
    curves = tuple(
        SCurveParams(toe=v[i], shoulder=v[i + 1], slope=v[i + 2], pivot=v[i + 3])
        for i in (1, 5, 9)
    )
    return GlobalParams(exposure=v[0]), curves


def _ct_from_vec(v: np.ndarray) -> CrosstalkParams:
    return CrosstalkParams(rg=v[0], rb=v[1], gr=v[2], gb=v[3], br=v[4], bg=v[5])


def _cd_from_vec(v: np.ndarray) -> tuple[SatLumaParams, FourierHueParams]:
    # Saturation LEVEL is never learned (ADR-0007): web-still statistics under-measure a
    # film's perceived vividness (compression + dark grades), so the fit chased dull numbers
    # and the user's footage lost its colour. Only the RELATIVE saturation behaviour is
    # learned: sat-vs-luma is normalized to mean 1, and the hue-trim curve's DC term (t0)
    # is zeroed — hue-to-hue differences survive, the overall level stays the footage's own.
    sl = SatLumaParams(shadow=v[0], mid=v[1], high=v[2])
    coefs = dict(zip(FourierHueParams.field_names(), v[3:], strict=True))
    coefs["t0"] = 0.0                                    # sat level is never learned
    fh = FourierHueParams(**coefs)
    return sl, fh


# ── residuals ─────────────────────────────────────────────────────────

def _conf(w: np.ndarray, w0: float) -> np.ndarray:
    return np.sqrt(w / (w + w0))


def _residuals(out_display: np.ndarray, ref: PooledTargets, opt: FitOptions,
               *, tone: bool, balance: bool, chroma: bool, zones: bool,
               hue_luma: bool = False, tone_weight: float | None = None) -> np.ndarray:
    s = compute_frame_stats(out_display)
    parts = []
    if tone:
        tw = opt.quantile_weight if tone_weight is None else tone_weight
        parts.append(tw * (s.channel_quantiles - ref.channel_quantiles).ravel())
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
        rs = ref.sat_by_band / max(float(w @ ref.sat_by_band), 1e-9)
        ss = s.sat_by_band / max(float(w @ s.sat_by_band), 1e-9)
        parts.append(opt.chroma_weight * c * (ss - rs))
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
                            diff_step=opt.diff_step, max_nfev=opt.max_nfev)
        result.stage_cost[name] = float(sol.cost)
        result.stage_nfev[name] = int(sol.nfev)
        return sol.x

    # Stage 1 — global exposure + tone curves (Blocks G+B). The most abundant statistic.
    tone_ridge = np.full(_TONE_NEUTRAL.shape, opt.ridge_tone)
    tone_ridge[0] = opt.ridge_exposure                 # ADR-0006 C: the lazy global darkening
    tone_v = run_stage(
        "tone", _TONE_NEUTRAL, _TONE_LO, _TONE_HI, tone_ridge, _TONE_NEUTRAL,
        lambda v: FilmModel(global_trim=_tone_from_vec(v)[0], curves=_tone_from_vec(v)[1]),
        {"tone": True, "balance": True, "chroma": False, "zones": False},
    )
    global_trim, curves = _tone_from_vec(tone_v)

    # Stage 2 — crosstalk (Block A). Tone frozen; quantiles stay as a soft anchor.
    ct_v = run_stage(
        "crosstalk", _CT_NEUTRAL, _CT_LO, _CT_HI, opt.ridge_crosstalk, _CT_NEUTRAL,
        lambda v: FilmModel(global_trim=global_trim, crosstalk=_ct_from_vec(v), curves=curves),
        {"tone": True, "balance": True, "chroma": False, "zones": False,
         "tone_weight": opt.tone_anchor_weight},  # hue detail belongs to the Fourier stage

    )
    crosstalk = _ct_from_vec(ct_v)

    # Stage 3 — saturation & hue detail (Blocks C/D). G, A and B frozen.
    # trim coefficients ridge stiffer than shift (spurious sat wiggle bands on gradients);
    # higher harmonics ridge ~k^2 — a curvature penalty, the classic smoothness prior
    cd_ridge = np.concatenate([np.full(3, opt.ridge_huesat * 4.0),  # sat shape: gentle
                               opt.ridge_huesat * _HARM ** 2,
                               opt.ridge_huesat * 4.0 * _HARM ** 2,
                               opt.ridge_huesat * 0.5 * _HARM_L ** 2])  # Block E
    cd_v = run_stage(
        "huesat", _CD_NEUTRAL, _CD_LO, _CD_HI, cd_ridge, _CD_NEUTRAL,
        lambda v: FilmModel(global_trim=global_trim, crosstalk=crosstalk, curves=curves,
                            sat_luma=_cd_from_vec(v)[0], hue_fourier=_cd_from_vec(v)[1]),
        {"tone": True, "balance": True, "chroma": True, "zones": True,
         "tone_weight": opt.tone_anchor_weight},
    )
    sat_luma, hue_fourier = _cd_from_vec(cd_v)

    # Stage 4 — polish (per-hue LUMINANCE): with the hue field settled, re-fit tone+crosstalk
    # against the hue-luma targets ("lush vs olive greens") — the only blocks that can move a
    # hue family's brightness. Anchored at the stage-1/2 solution, so this refines rather
    # than re-opens the tone fit.
    p0 = np.concatenate([tone_v, ct_v])
    p_lo = np.concatenate([_TONE_LO, _CT_LO])
    p_hi = np.concatenate([_TONE_HI, _CT_HI])
    p_ridge = np.concatenate([np.full(13, 0.3), np.full(6, 0.3)])

    def _polish_model(v):
        gt, cv = _tone_from_vec(v[:13])
        return FilmModel(global_trim=gt, crosstalk=_ct_from_vec(v[13:]), curves=cv,
                         sat_luma=sat_luma, hue_fourier=hue_fourier)

    pol_v = run_stage(
        "polish", p0, p_lo, p_hi, p_ridge, p0,
        _polish_model,
        {"tone": True, "balance": True, "chroma": False, "zones": False, "hue_luma": True,
         "tone_weight": opt.tone_anchor_weight},
    )
    global_trim, curves = _tone_from_vec(pol_v[:13])
    crosstalk = _ct_from_vec(pol_v[13:])

    result.model = FilmModel(global_trim=global_trim, crosstalk=crosstalk, curves=curves,
                             sat_luma=sat_luma, hue_fourier=hue_fourier)
    if progress:
        progress("done")
    return result
