"""character — the film-print character preset (Block F starting vector, ADR-0008 b8.2).

@context  The research verdict: a weak/ambiguous pool must fall back to FILM, not to
          identity — the fit INITIALIZES here and is prior-anchored here (b8.4). Numbers
          come from published Kodak data (docs/research/2026-08-12-film-physics.md), not
          from taste: Digital LAD print-through aim gammas R 0.966 / G 1.063 / B 1.082
          (the measured composed neg→print neutral tracking — R runs ~9% flatter, B ~2%
          steeper, normalized to G), system gamma 0.55×2.6 ≈ 1.4, print input window
          ~6 stops (paper-white ≈ +2.6, D-max ≈ −3.4), neg toe near −3 stops, mid-density
          DIR interimage boost.
@done     film_print_character() -> FilmSystemParams.
@todo     Fit-v2 uses this as x0 + ridge anchor (b8.4).
@limits   PURE constants. Honest naming per spec §7: this is a "film-print character" —
          a plausible colour-negative→print system — NOT a certified emulation of any
          named stock. Values are relative deviations composed on top of the sacred base
          conversion (neutral params = identity), so the preset is a LOOK, not a
          colorimetric replacement.
@affects  Used by fitter/fit.py (b8.4) + app recipe display. See ADR-0008.
"""

from __future__ import annotations

from .filmsystem import CouplingParams, FilmSystemParams, NegativeParams, PrintParams


def film_print_character() -> FilmSystemParams:
    """The datasheet-derived film-print starting character (see module brief for sources)."""
    return FilmSystemParams(
        negative=NegativeParams(
            g_r=0.909,           # LAD print-through 0.966/1.063 — red layer runs flatter
            g_g=1.0,
            g_b=1.018,           # 1.082/1.063 — blue slightly steeper (cool-shadow feel)
            toe=0.20,            # gentle neg foot
            toe_at=-3.0,         # ~3 stops under grey (datasheet toe region)
        ),
        coupling=CouplingParams( # modest DIR interimage (mid-density sat boost)
            rg=0.05, rb=0.02, gr=0.04, gb=0.04, br=0.02, bg=0.03,
        ),
        printer=PrintParams(
            slope=1.40,          # system gamma ≈ 0.55 × 2.6 (dark-surround intent)
            shoulder=0.65,       # paper-white convergence
            ptoe=0.55,           # D-max neutral black convergence
            range_hi=2.6,        # ~6-stop print input window, asymmetric around grey
            range_lo=-3.4,
        ),
    )
