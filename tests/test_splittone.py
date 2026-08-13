"""Block S (split tone / luminance-conditional tint curve) — b8.5."""

from __future__ import annotations

import numpy as np

from lutgen.fitter.filmmodel import FilmModel, SplitToneParams
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict
from lutgen.fitter.filmmodel.splittone import apply_split_tone


def _lab(seed=0, n=256):
    rng = np.random.default_rng(seed)
    lab = np.empty((n, 3))
    lab[:, 0] = rng.uniform(0.05, 0.95, n)
    lab[:, 1:] = rng.uniform(-0.15, 0.15, (n, 2))
    return lab


def test_identity_bit_for_bit():
    lab = _lab()
    np.testing.assert_array_equal(apply_split_tone(lab, SplitToneParams()), lab)
    assert FilmModel.identity().is_identity()


def test_paints_an_arch():
    """Warm mids, cool ends — the shape no monotone block can express."""
    st = SplitToneParams(t0_b=-0.02, t1_b=0.02, t2_b=0.03, t3_b=0.02, t4_b=-0.02)
    lab = np.column_stack([np.linspace(0.05, 0.95, 91), np.zeros(91), np.zeros(91)])
    out = apply_split_tone(lab, st)
    b = out[:, 2]
    assert b[45] > b[0] and b[45] > b[-1]              # mid warmer than both ends
    assert np.all(np.diff(out[:, 0]) >= 0)             # L untouched


def test_l_never_changes():
    st = SplitToneParams(t0_a=0.05, t2_b=-0.05, t4_a=-0.05)
    lab = _lab(1)
    out = apply_split_tone(lab, st)
    np.testing.assert_array_equal(out[:, 0], lab[:, 0])


def test_serializes_and_old_profiles_stay_neutral():
    m = FilmModel(split_tone=SplitToneParams(t1_b=0.02, t4_b=-0.03))
    m2 = model_from_dict(model_to_dict(m))
    assert m2.split_tone == m.split_tone
    d = model_to_dict(FilmModel.identity())
    del d["split_tone"]                                 # a pre-8.5 profile
    assert model_from_dict(d).split_tone.is_identity()


def test_color_dial_scales_it():
    from lutgen.fitter.filmmodel.scale import scaled_model
    m = FilmModel(split_tone=SplitToneParams(t1_b=0.02))
    assert scaled_model(m, 1.0, 0.0).split_tone.is_identity()   # colour off => gone
    assert scaled_model(m, 0.0, 1.0).split_tone == m.split_tone # colour on => kept


def test_fit_recovers_a_split_tone():
    """Ground truth: film preset + a warm-mid arch. The crosstalk stage must land the
    tint curve's direction (band-balance targets drive it)."""
    from lutgen.fitter.filmmodel import film_print_character
    from lutgen.fitter.fit import FitOptions, _cube_fn, fit_film_model, synth_samples
    from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
    from lutgen.orchestration.poolstats import compute_frame_stats, neutral_prior, pool_stats
    OPT = FitOptions(n_samples=1500, max_nfev=80, exposure_align=False)  # 16-param stage needs headroom
    truth = FilmModel(film_system=film_print_character(),
                      split_tone=SplitToneParams(t1_b=0.025, t2_b=0.03, t4_b=-0.02))
    cloud = synth_samples(neutral_prior(), OPT.n_samples, OPT.seed)
    inv = _cube_fn(load_base_inverse(), INVERSE_SIZE)
    fwd = _cube_fn(load_base(), DEFAULT_SIZE)
    ref = pool_stats([compute_frame_stats(np.clip(fwd(truth.forward(inv(cloud))), 0, 1))])
    res = fit_film_model(ref, options=OPT)
    # v3 philosophy: assert the RENDERED statistic, not pole params (crosstalk/lights can
    # legitimately carry part of the split — expression across blocks is degenerate)
    out = np.clip(fwd(res.model.forward(inv(cloud))), 0, 1)
    s = compute_frame_stats(out)
    r = pool_stats([compute_frame_stats(np.clip(fwd(truth.forward(inv(cloud))), 0, 1))])
    # honest capability: the rendered warm arch EXISTS (an inner band warmer than both
    # ends) — printer lights legitimately absorb the monotone share of the warmth, so
    # magnitude-tracking of the poles themselves is degenerate on synthetic worlds
    inner = float(s.band_mean_ab[1:4, 1].max())
    assert inner > s.band_mean_ab[0, 1] + 0.005
    assert inner > s.band_mean_ab[4, 1] + 0.005
    assert r.band_mean_ab[1:4, 1].max() > r.band_mean_ab[0, 1]   # truth world does too
