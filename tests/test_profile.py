"""Deterministic tests for the Look Profile — serialize.py (pure params <-> dict) and
profile.py (versioned JSON file IO) — ADR-0001 b1.5."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lutgen.fitter.filmmodel import (
    CrosstalkParams,
    FilmModel,
    HueZoneParams,
    SatLumaParams,
    SCurveParams,
)
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict
from lutgen.orchestration.profile import (
    PROFILE_FORMAT,
    LookProfile,
    load_profile,
    save_profile,
)


def _rich_model() -> FilmModel:
    return FilmModel(
        crosstalk=CrosstalkParams(rg=0.04, rb=-0.02, gr=0.08, gb=0.01, br=0.03, bg=-0.05),
        curves=(SCurveParams(toe=0.4, shoulder=0.2, slope=1.25, pivot=0.45),
                SCurveParams(slope=1.1),
                SCurveParams(shoulder=0.5, slope=0.9, pivot=0.55)),
        sat_luma=SatLumaParams(shadow=0.7, mid=1.25, high=0.85),
        hue_zones=HueZoneParams(r_shift=0.12, r_trim=-0.1, b_trim=0.2, m_shift=-0.05),
    )


# ── serialize: pure round-trip ────────────────────────────────────────

def test_dict_roundtrip_exact():
    m = _rich_model()
    assert model_from_dict(model_to_dict(m)) == m


def test_identity_roundtrip_stays_identity():
    m = model_from_dict(model_to_dict(FilmModel.identity()))
    assert m.is_identity()


def test_empty_dict_gives_identity():
    assert model_from_dict({}).is_identity()


def test_partial_dict_neutral_defaults():
    # hand-trimmed profile: only one curve's slope given — everything else neutral
    m = model_from_dict({"curves": {"g": {"slope": 1.3}}})
    assert m.curves[1].slope == 1.3
    assert m.curves[0].is_identity() and m.curves[2].is_identity()
    assert m.crosstalk.is_identity() and m.sat_luma.is_identity() and m.hue_zones.is_identity()


def test_unknown_keys_ignored():
    d = model_to_dict(_rich_model())
    d["crosstalk"]["future_field"] = 9.9
    d["curves"]["r"]["banana"] = 1.0
    assert model_from_dict(d) == _rich_model()


def test_non_dict_raises():
    with pytest.raises(TypeError):
        model_from_dict([1, 2, 3])


def test_dict_is_json_ready_floats():
    d = model_to_dict(_rich_model())
    json.dumps(d)                                     # must not raise
    assert d["curves"]["r"]["toe"] == 0.4
    assert d["hue_zones"]["b_trim"] == 0.2


# ── profile: file round-trip ──────────────────────────────────────────

def test_file_roundtrip(tmp_path):
    p = LookProfile(model=_rich_model(), name="kodak-ish", n_frames=7,
                    stage_cost={"tone": 0.01, "crosstalk": 0.005, "huesat": 0.002},
                    stage_nfev={"tone": 30, "crosstalk": 12, "huesat": 25})
    f = tmp_path / "look.json"
    save_profile(f, p)
    q = load_profile(f)
    assert q.model == p.model
    assert q.name == "kodak-ish" and q.n_frames == 7
    assert q.stage_cost == p.stage_cost and q.stage_nfev == p.stage_nfev


def test_saved_file_is_human_readable(tmp_path):
    f = tmp_path / "look.json"
    save_profile(f, LookProfile(model=_rich_model(), name="x"))
    text = f.read_text(encoding="utf-8")
    assert text.startswith("{\n")                     # indented JSON
    data = json.loads(text)
    assert data["format"] == PROFILE_FORMAT and data["version"] == 1
    assert "crosstalk" in data["model"]


def test_load_rejects_wrong_format(tmp_path):
    f = tmp_path / "notlook.json"
    f.write_text(json.dumps({"format": "something-else", "version": 1, "model": {}}))
    with pytest.raises(ValueError, match="not a ReinaLook look profile"):
        load_profile(f)


def test_load_rejects_wrong_version(tmp_path):
    f = tmp_path / "v99.json"
    f.write_text(json.dumps({"format": PROFILE_FORMAT, "version": 99, "model": {}}))
    with pytest.raises(ValueError, match="unsupported profile version"):
        load_profile(f)


def test_load_rejects_missing_model(tmp_path):
    f = tmp_path / "nomodel.json"
    f.write_text(json.dumps({"format": PROFILE_FORMAT, "version": 1}))
    with pytest.raises(ValueError, match="no 'model' section"):
        load_profile(f)


def test_load_rejects_invalid_json(tmp_path):
    f = tmp_path / "broken.json"
    f.write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_profile(f)


def test_v1_preset_is_rejected(tmp_path):
    # a v1 refs-preset must not masquerade as a look profile
    f = tmp_path / "preset.json"
    f.write_text(json.dumps({"version": 1, "refs": ["a.png"], "strength": 0.8}))
    with pytest.raises(ValueError, match="not a ReinaLook look profile"):
        load_profile(f)


def test_from_fit_result_and_apply(tmp_path):
    # end-to-end shape: FitResult -> profile -> file -> load -> forward() works
    from lutgen.fitter.fit import FitResult
    res = FitResult(model=_rich_model(), stage_cost={"tone": 0.02}, stage_nfev={"tone": 9},
                    n_frames=5)
    prof = LookProfile.from_fit_result(res, name="from-fit")
    f = tmp_path / "ff.json"
    save_profile(f, prof)
    loaded = load_profile(f)
    rgb = np.random.default_rng(0).uniform(0, 1, (64, 3))
    np.testing.assert_array_equal(loaded.model.forward(rgb), _rich_model().forward(rgb))
