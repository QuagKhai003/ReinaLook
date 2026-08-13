"""Deterministic tests for Block F — the physical negative→print system (ADR-0008 b8.1).

The EMERGENT-behaviour tests are the point: with film-like parameters the system must
produce, without any explicit saturation control, the signatures the research briefs
identify — mid-tone saturation punch, highlight desaturation toward a shared white,
neutral-converging blacks, coupling that INCREASES separation, and filmic push behaviour.
"""

from __future__ import annotations

import numpy as np

from lutgen.engine.perceptual import to_oklab
from lutgen.fitter.filmmodel.filmsystem import (
    CouplingParams,
    FilmSystemParams,
    NegativeParams,
    PrintParams,
    apply_film_system,
    eval_log_exposure,
)


def _filmish() -> FilmSystemParams:
    """A plausible film-print character (magnitudes from the research briefs)."""
    return FilmSystemParams(
        negative=NegativeParams(g_r=0.92, g_g=1.0, g_b=1.02, toe=0.25, toe_at=-3.0),
        coupling=CouplingParams(rg=0.06, gr=0.05, gb=0.05, bg=0.04),
        printer=PrintParams(slope=1.35, shoulder=0.6, ptoe=0.5),
    )


def _ramp(n=257):
    x = np.linspace(0.01, 0.99, n)
    return np.column_stack([x, x, x])


def _chroma(di):
    lab = to_oklab(np.clip(di, 0, 1))
    return float(np.hypot(lab[..., 1], lab[..., 2]).max())


# ── contracts ─────────────────────────────────────────────────────────

def test_neutral_is_identity_bit_for_bit():
    rgb = np.random.default_rng(0).uniform(0, 1, (256, 3))
    np.testing.assert_array_equal(apply_film_system(rgb, FilmSystemParams.neutral()), rgb)
    assert FilmSystemParams.neutral().is_identity()


def test_grey_is_a_fixed_point():
    p = _filmish()
    from lutgen.engine.spaces import di_encode
    g = float(di_encode(np.array([0.18]))[0])
    out = apply_film_system(np.full((4, 3), g), p)
    np.testing.assert_allclose(out, g, atol=1e-9)      # exposure belongs to Block G


def test_monotone_on_gray_axis_and_per_channel():
    p = _filmish()
    out = apply_film_system(_ramp(), p)
    assert np.all(np.diff(out, axis=0) > -1e-9)
    # per-channel monotone along each channel's own ramp with others fixed
    x = np.linspace(0.05, 0.95, 129)
    for c in range(3):
        probe = np.full((129, 3), 0.4)
        probe[:, c] = x
        o = apply_film_system(probe, p)
        assert np.all(np.diff(o[:, c]) > -1e-9)


def test_smooth_c1_no_kinks():
    out = apply_film_system(_ramp(1025), _filmish())
    d2 = np.diff(out[:, 1], 2)
    assert np.max(np.abs(d2)) < 5e-4


# ── emergent film behaviour (the research signatures) ─────────────────

def _saturated_probe(le_stops, sep=0.15):
    """A colour at a given exposure level: channels separated by ±sep in log exposure."""
    from lutgen.engine.spaces import di_encode
    le = np.array([le_stops * 0.301 + sep, le_stops * 0.301, le_stops * 0.301 - sep])
    return di_encode(0.18 * np.power(10.0, le)).reshape(1, 3)


def test_saturation_follows_curve_slope():
    """Mid-tones separate MORE than the input (slope>1 punch); highlights separate LESS
    (shoulder) — with no explicit saturation parameter anywhere."""
    p = _filmish()
    mid_in, hi_in = _saturated_probe(0.0), _saturated_probe(2.4)
    mid_out, hi_out = apply_film_system(mid_in, p), apply_film_system(hi_in, p)
    spread = lambda a: float(eval_log_exposure(a).max() - eval_log_exposure(a).min())
    assert spread(mid_out) > spread(mid_in) * 1.15     # mid punch (slope 1.35 − coupling)
    assert spread(hi_out) < spread(hi_in) * 0.9        # highlight convergence


def test_highlights_desaturate_toward_shared_white():
    p = _filmish()
    hi = _saturated_probe(2.8)
    assert float(_chroma(apply_film_system(hi, p))) < float(_chroma(hi)) * 0.75


def test_blacks_converge_neutral():
    p = _filmish()
    lo = _saturated_probe(-4.0)
    assert float(_chroma(apply_film_system(lo, p))) < float(_chroma(lo)) * 0.9


def test_coupling_increases_separation():
    """DIR coupling alone (negative off-diagonals) must INCREASE colour separation — the
    sign our old fitted crosstalk got wrong."""
    p = FilmSystemParams(NegativeParams(), CouplingParams(rg=0.08, gr=0.08, gb=0.06, bg=0.06),
                         PrintParams())
    mid = _saturated_probe(0.0)
    assert float(_chroma(apply_film_system(mid, p))) > float(_chroma(mid)) * 1.05


def test_coupling_preserves_neutrals_exactly():
    p = FilmSystemParams(NegativeParams(), CouplingParams(rg=0.1, gr=0.07, br=0.05),
                         PrintParams())
    gray = np.tile(np.linspace(0.05, 0.95, 16)[:, None], (1, 3))
    np.testing.assert_allclose(apply_film_system(gray, p), gray, atol=1e-9)


def test_push_behaviour_is_filmic():
    """+2 stops on the INPUT must not be a plain gain: contrast around the new level and
    colour separation must change (the negative/print structure responds to level) —
    Filmbox's 'behaves as though the negative saw more light' property."""
    p = _filmish()
    base = _saturated_probe(0.0)
    pushed = _saturated_probe(2.0)                     # same colour, 2 stops brighter
    r_base = apply_film_system(base, p)
    r_push = apply_film_system(pushed, p)
    # separation at +2 stops is compressed vs at 0 stops (shoulder engagement)
    spread = lambda a: float(eval_log_exposure(a).max() - eval_log_exposure(a).min())
    assert spread(r_push) < spread(r_base) * 0.85


def test_red_gamma_deficit_warms_shadows():
    """Vision3's flatter red layer: shadows drift warm (R falls less than G/B below grey)."""
    p = FilmSystemParams(NegativeParams(g_r=0.90, g_g=1.0, g_b=1.03), CouplingParams(),
                         PrintParams(slope=1.3))
    shadow = np.full((1, 3), 0.25)                     # a dark neutral
    out = apply_film_system(shadow, p)
    assert out[0, 0] > out[0, 1] > out[0, 2]           # R > G > B: warm shadow


def test_field_names_and_sections():
    names = FilmSystemParams.field_names()
    assert len(names) == 5 + 6 + 5 + 3                       # + printer lights
    assert "negative.g_r" in names and "coupling.rg" in names and "printer.slope" in names
