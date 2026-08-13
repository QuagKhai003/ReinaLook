"""b8.2 tests — character preset, FilmModel v3 integration, serialization, dials, recipe."""

from __future__ import annotations

import numpy as np

from lutgen.engine.perceptual import to_oklab
from lutgen.fitter.filmmodel import FilmModel, film_print_character
from lutgen.fitter.filmmodel.filmsystem import FilmSystemParams, apply_film_system
from lutgen.fitter.filmmodel.scale import scaled_model
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict


def _chroma(di):
    lab = to_oklab(np.clip(di, 0, 1))
    return np.hypot(lab[..., 1], lab[..., 2])


# ── the preset itself ─────────────────────────────────────────────────

def test_preset_is_not_identity_and_reads_as_film():
    fs = film_print_character()
    assert not fs.is_identity()
    # red flatter than green, blue steeper (LAD print-through ordering)
    assert fs.negative.g_r < fs.negative.g_g < fs.negative.g_b
    # positive system contrast + both convergence ends on
    assert fs.printer.slope > 1.2 and fs.printer.shoulder > 0 and fs.printer.ptoe > 0


def test_preset_has_film_signatures():
    """The acceptance line 'at preset, output already reads as film': mid punch + highlight
    desaturation, straight from the preset numbers."""
    fs = film_print_character()
    rng = np.random.default_rng(1)
    mid = np.clip(0.38 + rng.normal(0, 0.03, (64, 3)), 0, 1)      # colours near grey
    hi = np.clip(0.78 + rng.normal(0, 0.02, (64, 3)), 0, 1)       # colours near white
    assert _chroma(apply_film_system(mid, fs)).mean() > _chroma(mid).mean() * 1.05
    assert _chroma(apply_film_system(hi, fs)).mean() < _chroma(hi).mean() * 0.9


# ── model integration ─────────────────────────────────────────────────

def test_model_identity_still_bit_for_bit():
    rgb = np.random.default_rng(2).uniform(0, 1, (128, 3))
    m = FilmModel.identity()
    assert m.is_identity()
    np.testing.assert_array_equal(m.forward(rgb), rgb)


def test_model_with_film_system_only():
    m = FilmModel(film_system=film_print_character())
    assert not m.is_identity()
    rgb = np.random.default_rng(3).uniform(0.1, 0.9, (64, 3))
    out = m.forward(rgb)
    np.testing.assert_allclose(out, apply_film_system(rgb, m.film_system))


def test_old_profile_dict_stays_bit_identical():
    """A pre-8.2 profile dict has no film_system section — it must load neutral and render
    exactly as before (legacy blocks path)."""
    legacy = FilmModel(film_system=FilmSystemParams.neutral())
    d = model_to_dict(legacy)
    del d["film_system"]
    m = model_from_dict(d)
    assert m.film_system.is_identity()
    rgb = np.random.default_rng(4).uniform(0, 1, (64, 3))
    np.testing.assert_array_equal(m.forward(rgb), legacy.forward(rgb))


def test_serialize_round_trip_exact():
    m = FilmModel(film_system=film_print_character())
    m2 = model_from_dict(model_to_dict(m))
    assert m2.film_system == m.film_system
    rgb = np.random.default_rng(5).uniform(0, 1, (32, 3))
    np.testing.assert_array_equal(m.forward(rgb), m2.forward(rgb))


# ── dials ─────────────────────────────────────────────────────────────

def test_dials_zero_neutralize_film_system():
    m = FilmModel(film_system=film_print_character())
    s = scaled_model(m, 0.0, 0.0)
    assert s.film_system.is_identity()
    assert s.is_identity()


def test_dials_full_reproduce_film_system():
    m = FilmModel(film_system=film_print_character())
    assert scaled_model(m, 1.0, 1.0).film_system == m.film_system


def test_tone_dial_keeps_contrast_drops_color():
    m = FilmModel(film_system=film_print_character())
    s = scaled_model(m, 1.0, 0.0).film_system
    assert s.printer.slope == m.film_system.printer.slope       # contrast stays
    assert s.coupling.is_identity()                              # coupling is colour
    assert abs(s.negative.g_r - s.negative.g_b) < 1e-12          # channel deviations gone


def test_color_dial_keeps_deviations_drops_contrast():
    m = FilmModel(film_system=film_print_character())
    s = scaled_model(m, 0.0, 1.0).film_system
    assert s.printer.slope == 1.0 and s.printer.shoulder == 0.0
    assert not s.coupling.is_identity()
    assert s.negative.g_r < s.negative.g_b                       # deviation ordering kept


# ── recipe display ────────────────────────────────────────────────────

def test_recipe_shows_film_system():
    from lutgen.app.recipe import recipe_summary
    from lutgen.orchestration.profile import LookProfile
    p = LookProfile(name="t", model=FilmModel(film_system=film_print_character()))
    text = recipe_summary(p)
    assert "Film system" in text
    assert "contrast" in text and "coupling" in text and "negative" in text
