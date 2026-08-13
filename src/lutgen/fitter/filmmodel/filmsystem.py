"""filmsystem — Block F: the physical negative→print tonal core (ADR-0008).

@context  Research verdict (docs/research/2026-08-12-*): film's look IS a structure —
          a near-linear negative in log exposure, saturation-INCREASING channel coupling in
          the density domain (DIR couplers: negative off-diagonals), and a steep print
          S-curve whose slope profile CREATES the saturation-vs-exposure behaviour (mid
          punch, ends converging neutral). Every successful emulation product implements
          this two-stage model; free statistical maps cannot reproduce it. This block is
          that system, parameterized as DEVIATIONS so neutral params = identity bit-for-bit
          (all existing contracts — sacred strength-0, dials, validator — survive).
@done     NegativeParams / CouplingParams / PrintParams / PrinterLights / FilmSystemParams +
          apply_film_system on DWG/DI code values; eval helpers for tests/recipe. Printer
          lights (b8.4): per-channel print-stage exposure offsets — the physical colour-
          timing/cast mechanism; grey stays fixed for every sub-stage EXCEPT lights.
@todo     -
@limits   PURE: no IO. Works in STOPS relative to 18% grey (le = log2(lin/0.18)), so a DI
          code offset (Block G / printer lights) composes naturally and every threshold
          parameter (toe_at, range_hi/lo) reads directly in stops. Monotone by
          construction: neg is an affine-plus-soft-knee map with positive slope, coupling
          rows sum to 0 with off-diagonals <= 0 (bounded so diagonal dominance holds),
          print is a monotone soft-saturating S. Neutral (all zeros / ones) returns input
          BIT-FOR-BIT. Out-of-range DI handled by the same smooth maps (no branches).
@affects  Composed by model.FilmModel (b8.2) replacing the display-space S-curves as the
          tonal core. See ADR-0008 + Kodak US7327382 skeleton.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from lutgen.engine.spaces import di_decode, di_encode

_GREY = 0.18                 # log-exposure anchor (18% grey)
_EPS = 1e-6


@dataclass
class NegativeParams:
    """Per-channel negative behaviour, relative to a neutral 1:1 log response.

    ``g_*``: relative gamma per channel (1 = neutral; film: R slightly LOW — Vision3's red
    layer runs ~15% flatter, one mechanism of warm shadow drift). ``toe``: soft shadow-foot
    compression amount (0 = off); ``toe_at``: where the foot starts, in stops below grey.
    """

    g_r: float = 1.0
    g_g: float = 1.0
    g_b: float = 1.0
    toe: float = 0.0
    toe_at: float = -3.0

    def is_identity(self) -> bool:
        return self.g_r == 1.0 and self.g_g == 1.0 and self.g_b == 1.0 and self.toe == 0.0


@dataclass
class CouplingParams:
    """Density-domain inter-layer coupling (DIR couplers). ``xy`` = how strongly density in
    layer x SUPPRESSES layer y (>= 0 here; applied with a negative sign). Rows sum to zero
    internally, so neutrals are exactly preserved; saturation is non-decreasing by
    construction — the sign film actually has (research brief #1/#3).
    """

    rg: float = 0.0
    rb: float = 0.0
    gr: float = 0.0
    gb: float = 0.0
    br: float = 0.0
    bg: float = 0.0

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())

    def matrix(self) -> np.ndarray:
        """I + C with C off-diagonals negative (suppression) and rows summing to 1."""
        r, g, b = 0, 1, 2
        c = np.zeros((3, 3))
        c[g, r] = -self.rg
        c[b, r] = -self.rb
        c[r, g] = -self.gr
        c[b, g] = -self.gb
        c[r, b] = -self.br
        c[g, b] = -self.bg
        np.fill_diagonal(c, -c.sum(axis=1))          # diagonal boosts what others suppress
        return np.eye(3) + c


@dataclass
class PrintParams:
    """The print stage — the LOOK. ``slope``: system contrast around grey (1 = neutral;
    film systems run ≈1.4–1.6). ``shoulder``/``ptoe``: smooth convergence strengths at the
    white / black ends (0 = off) — these CREATE highlight desaturation toward paper-white
    and dense neutral blacks, because all channels share the limits. ``range_hi``/
    ``range_lo``: where the limits sit, in stops above/below grey.
    """

    slope: float = 1.0
    shoulder: float = 0.0
    ptoe: float = 0.0
    range_hi: float = 2.6    # paper-white ~2.6 stops over grey (print input window ~6 stops)
    range_lo: float = -3.4

    def is_identity(self) -> bool:
        return self.slope == 1.0 and self.shoulder == 0.0 and self.ptoe == 0.0


@dataclass
class PrinterLights:
    """Per-channel printer-light offsets, in STOPS of exposure at the print stage — the
    physical colour-cast/timing mechanism (Kodak flow: the timer dials R/G/B lights when
    printing the negative). 0 = neutral. These move grey CHROMATICALLY by design — they
    are colour, not brightness (overall brightness stays Block G's job)."""

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0

    def is_identity(self) -> bool:
        return self.r == 0.0 and self.g == 0.0 and self.b == 0.0


@dataclass
class FilmSystemParams:
    negative: NegativeParams
    coupling: CouplingParams
    printer: PrintParams
    lights: PrinterLights = field(default_factory=PrinterLights)

    @classmethod
    def neutral(cls) -> FilmSystemParams:
        return cls(NegativeParams(), CouplingParams(), PrintParams())

    def is_identity(self) -> bool:
        return (self.negative.is_identity() and self.coupling.is_identity()
                and self.printer.is_identity() and self.lights.is_identity())

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        names = []
        for section, klass in (("negative", NegativeParams), ("coupling", CouplingParams),
                               ("printer", PrintParams), ("lights", PrinterLights)):
            names += [f"{section}.{f.name}" for f in fields(klass)]
        return tuple(names)


def _soft_knee_low(x: np.ndarray, at: float, amount: float) -> np.ndarray:
    """Smoothly compress ``x`` below ``at`` by ``amount`` (0..1). C1; identity at amount=0.
    softplus-based: keeps monotonicity for amount < 1."""
    if amount == 0.0:
        return x
    k = 1.5                                          # knee softness (1/stops)
    return x + amount * (np.logaddexp(0.0, -k * (x - at)) / k) - amount * (
        np.logaddexp(0.0, -k * (0.0 - at)) / k)      # re-anchored so grey (x=0) is fixed


def _soft_knee_high(x: np.ndarray, at: float, amount: float) -> np.ndarray:
    """Smoothly compress ``x`` above ``at`` (the shoulder). C1; identity at amount=0."""
    if amount == 0.0:
        return x
    k = 1.5
    return x - amount * (np.logaddexp(0.0, k * (x - at)) / k) + amount * (
        np.logaddexp(0.0, k * (0.0 - at)) / k)


def eval_log_exposure(di_code: np.ndarray) -> np.ndarray:
    """DI code -> exposure in STOPS relative to 18% grey (log2). Vectorized (...,3).
    Stops are the domain of every threshold in this block (toe_at, range_hi/lo, knee k)."""
    lin = np.maximum(di_decode(np.asarray(di_code, dtype=np.float64)), _EPS)
    return np.log2(lin / _GREY)


def _from_log_exposure(le: np.ndarray) -> np.ndarray:
    return di_encode(_GREY * np.power(2.0, le))


def apply_film_system(di_code: np.ndarray, p: FilmSystemParams) -> np.ndarray:
    """The negative→print system on DWG/DI code values. New array; neutral params return the
    input BIT-FOR-BIT. Grey (18%) is a fixed point of every sub-stage EXCEPT printer lights
    (which move grey chromatically by design — colour timing); overall brightness stays
    Block G's job."""
    di_code = np.asarray(di_code, dtype=np.float64)
    if di_code.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {di_code.shape}")
    if p.is_identity():
        return di_code.copy()

    le = eval_log_exposure(di_code)

    # — negative: per-channel relative gamma + shadow foot (density ∝ g·le above the toe) —
    g = np.array([p.negative.g_r, p.negative.g_g, p.negative.g_b])
    d = le * g                                        # "density" relative to grey, per channel
    d = _soft_knee_low(d, p.negative.toe_at, p.negative.toe)

    # — coupling: DIR suppression in the density domain (saturation-non-decreasing) —
    if not p.coupling.is_identity():
        d = d @ p.coupling.matrix().T

    # — printer lights: per-channel exposure offsets at the print stage (colour timing) —
    if not p.lights.is_identity():
        d = d + np.array([p.lights.r, p.lights.g, p.lights.b])

    # — print: system contrast + shared convergence limits (the look) —
    d = d * p.printer.slope
    d = _soft_knee_high(d, p.printer.range_hi, p.printer.shoulder)
    d = _soft_knee_low(d, p.printer.range_lo, p.printer.ptoe)

    return _from_log_exposure(d)
