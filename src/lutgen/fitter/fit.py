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
@limits   PURE numeric: no file IO beyond the bundled base assets via engine/base (allowed by
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
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
)
from lutgen.fitter.filmmodel.huezone import ZONE_ANGLES
from lutgen.orchestration.poolstats import (
    L_BAND_EDGES,
    QUANTILES,
    PooledTargets,
    compute_frame_stats,
    neutral_prior,
)

ProgressFn = Callable[[str], None]

# Stage names (also what the progress callback receives, spec §9 "Fitting tone…").
STAGES = ("tone", "crosstalk", "huesat")


@dataclass
class FitOptions:
    """Knobs for the staged fit. Defaults are the production values; tests shrink n_samples."""

    n_samples: int = 3000        # source-cloud size (statistics are heavily over-determined)
    seed: int = 0                # cloud seed — fixed => deterministic fit
    w0: float = 0.02             # thin-data knee: conf = sqrt(w/(w+w0))
    ridge_tone: float = 0.05     # pull-to-neutral strength per stage (per-region reg.)
    ridge_crosstalk: float = 0.15
    ridge_huesat: float = 0.15
    quantile_weight: float = 1.0
    balance_weight: float = 2.0
    chroma_weight: float = 2.0
    zone_weight: float = 1.5
    tone_anchor_weight: float = 0.3   # stage 2/3: soft anchor keeping stage-1 tone in place
    max_nfev: int | None = None       # per stage; None = scipy default
    diff_step: float = 1e-3           # finite-difference step (quantile stats are piecewise)
    exposure_align: bool = True       # prior path only: match the assumed world's exposure to
                                      # the pool's (scene brightness is content, not grade —
                                      # ADR-0003; a real user source pool is never rescaled)


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

    zone_w = targets.zone_weight.copy()
    chromatic_share = float(zone_w.sum())
    if chromatic_share > 0:
        zone_p = zone_w / chromatic_share
        zones = rng.choice(len(zone_p), size=n, p=zone_p)
        hue = ZONE_ANGLES[zones] + rng.uniform(-0.5, 0.5, n)
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

_TONE_NEUTRAL = np.array([0.0, 0.0, 1.0, 0.5] * 3)               # (toe, shoulder, slope, pivot) x RGB
_TONE_LO = np.array([0.0, 0.0, 0.5, 0.3] * 3)
_TONE_HI = np.array([2.0, 2.0, 2.0, 0.7] * 3)
_CT_NEUTRAL = np.zeros(6)
_CT_LO, _CT_HI = np.full(6, -0.25), np.full(6, 0.25)
_CD_NEUTRAL = np.array([1.0, 1.0, 1.0] + [0.0] * 12)             # satluma x3, then 6 x (shift, trim)
_CD_LO = np.array([0.2, 0.2, 0.2] + [-0.35, -0.5] * 6)
_CD_HI = np.array([2.0, 2.0, 2.0] + [0.35, 0.5] * 6)


def _tone_from_vec(v: np.ndarray) -> tuple[SCurveParams, SCurveParams, SCurveParams]:
    return tuple(
        SCurveParams(toe=v[i], shoulder=v[i + 1], slope=v[i + 2], pivot=v[i + 3])
        for i in (0, 4, 8)
    )


def _ct_from_vec(v: np.ndarray) -> CrosstalkParams:
    return CrosstalkParams(rg=v[0], rb=v[1], gr=v[2], gb=v[3], br=v[4], bg=v[5])


def _cd_from_vec(v: np.ndarray) -> tuple[SatLumaParams, HueZoneParams]:
    sl = SatLumaParams(shadow=v[0], mid=v[1], high=v[2])
    z = v[3:]
    hz = HueZoneParams(
        r_shift=z[0], r_trim=z[1], y_shift=z[2], y_trim=z[3], g_shift=z[4], g_trim=z[5],
        c_shift=z[6], c_trim=z[7], b_shift=z[8], b_trim=z[9], m_shift=z[10], m_trim=z[11],
    )
    return sl, hz


# ── residuals ─────────────────────────────────────────────────────────

def _conf(w: np.ndarray, w0: float) -> np.ndarray:
    return np.sqrt(w / (w + w0))


def _residuals(out_display: np.ndarray, ref: PooledTargets, opt: FitOptions,
               *, tone: bool, balance: bool, chroma: bool, zones: bool,
               tone_weight: float | None = None) -> np.ndarray:
    s = compute_frame_stats(out_display)
    parts = []
    if tone:
        tw = opt.quantile_weight if tone_weight is None else tone_weight
        parts.append(tw * (s.channel_quantiles - ref.channel_quantiles).ravel())
    if balance:
        parts.append(opt.balance_weight * (s.mean_lab - ref.mean_lab))
    if chroma:
        c = _conf(ref.band_weight, opt.w0)
        parts.append(opt.chroma_weight * c * (s.chroma_by_band - ref.chroma_by_band))
    if zones:
        c = _conf(ref.zone_weight, opt.w0)[:, None]
        parts.append((opt.zone_weight * c * (s.zone_mean_ab - ref.zone_mean_ab)).ravel())
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

        def f(v):
            out = fwd(model_of(v).forward(di_cloud))
            data = _residuals(out, ref, opt, **res_kw)
            reg = np.sqrt(ridge) * (v - neutral)
            return np.concatenate([data, reg])

        sol = least_squares(f, x0, bounds=(lo, hi), method="trf",
                            diff_step=opt.diff_step, max_nfev=opt.max_nfev)
        result.stage_cost[name] = float(sol.cost)
        result.stage_nfev[name] = int(sol.nfev)
        return sol.x

    # Stage 1 — tone curves (Block B). Determined by the most abundant statistic.
    tone_v = run_stage(
        "tone", _TONE_NEUTRAL, _TONE_LO, _TONE_HI, opt.ridge_tone, _TONE_NEUTRAL,
        lambda v: FilmModel(curves=_tone_from_vec(v)),
        {"tone": True, "balance": True, "chroma": False, "zones": False},
    )
    curves = _tone_from_vec(tone_v)

    # Stage 2 — crosstalk (Block A). Tone frozen; quantiles stay as a soft anchor.
    ct_v = run_stage(
        "crosstalk", _CT_NEUTRAL, _CT_LO, _CT_HI, opt.ridge_crosstalk, _CT_NEUTRAL,
        lambda v: FilmModel(crosstalk=_ct_from_vec(v), curves=curves),
        {"tone": True, "balance": True, "chroma": False, "zones": True,
         "tone_weight": opt.tone_anchor_weight},
    )
    crosstalk = _ct_from_vec(ct_v)

    # Stage 3 — saturation & hue detail (Blocks C/D). A and B frozen.
    cd_v = run_stage(
        "huesat", _CD_NEUTRAL, _CD_LO, _CD_HI, opt.ridge_huesat, _CD_NEUTRAL,
        lambda v: FilmModel(crosstalk=crosstalk, curves=curves,
                            sat_luma=_cd_from_vec(v)[0], hue_zones=_cd_from_vec(v)[1]),
        {"tone": True, "balance": True, "chroma": True, "zones": True,
         "tone_weight": opt.tone_anchor_weight},
    )
    sat_luma, hue_zones = _cd_from_vec(cd_v)

    result.model = FilmModel(crosstalk=crosstalk, curves=curves,
                             sat_luma=sat_luma, hue_zones=hue_zones)
    if progress:
        progress("done")
    return result
