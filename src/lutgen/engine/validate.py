"""validate — stress-validation of a finished cube before export (the §6 gate).

@context  A fitted look must never export a broken LUT (spec §6): tone reversals band,
          hue breaks posterize, ΔE spikes between adjacent nodes show as contouring on real
          footage. The 65³ lattice is denser than any test chart, so the checks run on the
          cube itself: per-axis monotonicity (gradient ramps), the grey diagonal (tone ramp),
          adjacent-node ΔE in Oklab (smoothness), a hue-wheel sweep (hue continuity), and
          black/white endpoint sanity.
@done     Violation + ValidationReport; validate_cube(samples, size, reference=...).
@todo     -
@limits   PURE: no IO. Thresholds are RELATIVE to a reference cube (the base for "node2",
          the identity grid for "between") — the base's own curvature is not a violation.
          Tolerances leave room for a strong look but catch the failure modes that band.
@affects  Used by orchestration/learn.py (diagnose_model, block attribution) and the CLI
          apply export gate. See ADR-0001 b1.7, spec §6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import reshape_to_lattice
from .perceptual import to_oklab

# Monotonicity: grey diagonal is strict (float noise only). The per-axis check works on the
# SLOPE scale (diff x (size-1); identity slope = 1) per CHANNEL, flagged above 5% of steps:
# legitimate looks measure <= ~3% (hue curves wiggle individual channels; the base's own
# saturation compression adds corner wiggle even under plain mild crosstalk), while broken
# blocks measure 19..73% (channel-swap / hue-tear regimes). A display-LUMA variant was tried
# and rejected: the base itself dips display luma ~2% at gamut corners under legit mixing.
_MONO_TOL = 1e-6
_AXIS_SLOPE = -0.25      # a step steeper-down than this counts as reversed
_AXIS_BAD_FRAC = 0.05    # per-channel: >5% of steps reversed = violation
# Smoothness: max adjacent-node Oklab ΔE may exceed the reference's by this factor + floor.
_DE_FACTOR = 3.0
_DE_FLOOR = 0.05
# Hue sweep: max hue jump (radians) between adjacent sweep samples beyond the reference's.
_HUE_FACTOR = 3.0
_HUE_FLOOR = 0.12
# Endpoints: output black/white may drift from the reference endpoints by at most this (per ch).
_ENDPOINT_TOL = 0.10


@dataclass
class Violation:
    check: str          # "monotonic-tone" | "delta-e" | "hue-break" | "endpoints" | "range"
    detail: str         # human-readable: what, where, how bad

    def __str__(self) -> str:
        return f"[{self.check}] {self.detail}"


@dataclass
class ValidationReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.ok:
            return "validation OK"
        return "\n".join(str(v) for v in self.violations)


def _axis_monotonic(lat: np.ndarray) -> list[Violation]:
    """Each output channel must ramp up along its own input axis. Flags SYSTEMIC reversal
    (share of reversed steps above _AXIS_BAD_FRAC), not the isolated gamut-edge wiggle a
    legitimate hue/sat trim leaves. Lattice axes are (blue, green, red): channel c's input
    axis is axis 2-c."""
    size = lat.shape[0]
    out = []
    for c, name in enumerate(("red", "green", "blue")):
        slope = np.diff(lat[..., c], axis=2 - c) * (size - 1)
        frac = float((slope < _AXIS_SLOPE).mean())
        if frac > _AXIS_BAD_FRAC:
            out.append(Violation("monotonic-tone",
                                 f"{name} channel reverses along its ramp on {frac:.2%} of "
                                 f"steps (worst slope {float(slope.min()):.2f}; identity = 1)"))
    return out


def _gray_diag_monotonic(lat: np.ndarray, size: int) -> list[Violation]:
    """The grey ramp is strict: every output channel AND luminance must be ordered."""
    diag = lat[np.arange(size), np.arange(size), np.arange(size)]
    out = []
    d_ch = np.diff(diag, axis=0)
    worst_ch = float(d_ch.min())
    if worst_ch < -_MONO_TOL:
        out.append(Violation("monotonic-tone",
                             f"grey ramp channel value reverses (worst {worst_ch:.4f})"))
    luma = to_oklab(np.clip(diag, 0.0, 1.0))[:, 0]
    d = np.diff(luma)
    worst = float(d.min())
    if worst < -_MONO_TOL:
        where = int(np.argmin(d))
        out.append(Violation("monotonic-tone",
                             f"grey ramp brightness reverses near node {where}/{size - 1} "
                             f"(worst dL {worst:.4f})"))
    return out


def _max_neighbor_de(lat: np.ndarray) -> float:
    lab = to_oklab(np.clip(lat, 0.0, 1.0))
    worst = 0.0
    for ax in range(3):
        d = np.diff(lab, axis=ax)
        worst = max(worst, float(np.sqrt((d ** 2).sum(axis=-1)).max()))
    return worst


def _hue_sweep(samples_fn, n: int = 360) -> np.ndarray:
    """Output hue angles for a mid-grey-centred hue ring pushed through the cube."""
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    ring = 0.5 + 0.25 * np.column_stack([np.cos(theta),
                                         np.cos(theta - 2.0 * np.pi / 3.0),
                                         np.cos(theta + 2.0 * np.pi / 3.0)])
    out = to_oklab(np.clip(samples_fn(ring), 0.0, 1.0))
    return np.arctan2(out[:, 2], out[:, 1])


def _max_hue_jump(hues: np.ndarray) -> float:
    d = np.diff(np.concatenate([hues, hues[:1]]))          # close the ring
    d = np.abs((d + np.pi) % (2.0 * np.pi) - np.pi)        # wrapped difference
    return float(d.max())


def validate_cube(samples: np.ndarray, size: int, reference: np.ndarray,
                  *, interp=None, reference_interp=None) -> ValidationReport:
    """Run the §6 stress checks on final cube ``samples`` against a ``reference`` cube
    (the base for "node2", the identity grid for "between") whose own behaviour sets the
    thresholds. ``interp``/``reference_interp``: optional ``(N,3)->(N,3)`` cube-sampling
    functions for the hue sweep (skipped when not given)."""
    report = ValidationReport()
    lat = reshape_to_lattice(np.asarray(samples, dtype=np.float64), size)
    ref_lat = reshape_to_lattice(np.asarray(reference, dtype=np.float64), size)

    # range sanity
    if samples.min() < -1e-9 or samples.max() > 1.0 + 1e-9:
        report.violations.append(Violation(
            "range", f"samples outside [0,1]: min {samples.min():.4f}, max {samples.max():.4f}"))

    # tone: per-axis ramps + grey diagonal
    report.violations += _axis_monotonic(lat)
    report.violations += _gray_diag_monotonic(lat, size)

    # smoothness: adjacent-node ΔE vs the reference's own
    de = _max_neighbor_de(lat)
    de_ref = _max_neighbor_de(ref_lat)
    de_limit = _DE_FACTOR * de_ref + _DE_FLOOR
    if de > de_limit:
        report.violations.append(Violation(
            "delta-e", f"adjacent-node ΔE {de:.3f} exceeds limit {de_limit:.3f} "
                       f"(reference max {de_ref:.3f}) — will band on gradients"))

    # hue continuity: mid-tone hue wheel through the cube
    if interp is not None and reference_interp is not None:
        jump = _max_hue_jump(_hue_sweep(interp))
        jump_ref = _max_hue_jump(_hue_sweep(reference_interp))
        limit = _HUE_FACTOR * jump_ref + _HUE_FLOOR
        if jump > limit:
            report.violations.append(Violation(
                "hue-break", f"hue-wheel sweep jumps {jump:.3f} rad between adjacent hues "
                             f"(limit {limit:.3f}) — hue tears across a zone boundary"))

    # endpoints: black stays black-ish, white stays white-ish (vs the reference's endpoints)
    for idx, name in ((0, "black"), (-1, "white")):
        node = lat[idx, idx, idx] if idx == 0 else lat[-1, -1, -1]
        ref_node = ref_lat[idx, idx, idx] if idx == 0 else ref_lat[-1, -1, -1]
        drift = float(np.abs(node - ref_node).max())
        if drift > _ENDPOINT_TOL:
            report.violations.append(Violation(
                "endpoints", f"{name} point drifts {drift:.3f} from the conversion's "
                             f"{name} (tol {_ENDPOINT_TOL})"))
    return report
