"""Deterministic tests for Block G — global exposure trim (ADR-0004 b4.1)."""

from __future__ import annotations

import numpy as np
import pytest

from lutgen.fitter.filmmodel import (
    FilmModel,
    GlobalParams,
    SCurveParams,
    apply_global,
)
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict


def test_identity_is_bit_for_bit():
    rgb = np.random.default_rng(0).uniform(0, 1, (128, 3))
    np.testing.assert_array_equal(apply_global(rgb, GlobalParams()), rgb)
    assert FilmModel.identity().is_identity()          # G joins the identity contract


def test_exposure_is_code_offset():
    rgb = np.random.default_rng(1).uniform(0.2, 0.8, (64, 3))
    out = apply_global(rgb, GlobalParams(exposure=0.07))
    np.testing.assert_allclose(out, rgb + 0.07, atol=1e-15)


def test_model_applies_g_first():
    rgb = np.random.default_rng(2).uniform(0.1, 0.9, (64, 3))
    curves = (SCurveParams(slope=1.4), SCurveParams(), SCurveParams())
    m = FilmModel(global_trim=GlobalParams(exposure=-0.1), curves=curves)
    expected = FilmModel(curves=curves).forward(rgb - 0.1)
    np.testing.assert_allclose(m.forward(rgb), expected, atol=1e-12)


def test_nonidentity_model_with_only_exposure():
    rgb = np.random.default_rng(3).uniform(0.2, 0.8, (32, 3))
    m = FilmModel(global_trim=GlobalParams(exposure=0.05))
    assert not m.is_identity()
    np.testing.assert_allclose(m.forward(rgb), rgb + 0.05, atol=1e-15)


def test_serialize_roundtrip_and_backcompat():
    m = FilmModel(global_trim=GlobalParams(exposure=-0.12))
    assert model_from_dict(model_to_dict(m)) == m
    # profiles saved before Block G have no "global" section -> neutral default
    d = model_to_dict(m)
    del d["global"]
    assert model_from_dict(d).global_trim.is_identity()


def test_fit_recovers_pure_exposure_shift():
    """A look that is ONLY a global darkening must land in the exposure param (not curves)."""
    from lutgen.engine.base import (
        DEFAULT_SIZE,
        INVERSE_SIZE,
        load_base,
        load_base_inverse,
    )
    from lutgen.fitter.fit import FitOptions, _cube_fn, fit_film_model, synth_samples
    from lutgen.orchestration.poolstats import (
        compute_frame_stats,
        neutral_prior,
        pool_stats,
    )

    prior = neutral_prior()
    cloud = synth_samples(prior, 1200, 0)
    inv = _cube_fn(load_base_inverse(), INVERSE_SIZE)
    fwd = _cube_fn(load_base(), DEFAULT_SIZE)
    truth = FilmModel(global_trim=GlobalParams(exposure=-0.15))
    out = np.clip(fwd(truth.forward(inv(cloud))), 0, 1)
    ref = pool_stats([compute_frame_stats(out)])

    res = fit_film_model(ref, options=FitOptions(n_samples=1200, max_nfev=30,
                                                 exposure_align=False))
    assert res.model.global_trim.exposure == pytest.approx(-0.15, abs=0.05)
    for c in res.model.curves:                        # curves stay near neutral
        assert abs(c.slope - 1.0) < 0.25
