"""Deterministic tests for orchestration/learn.py + the learn/apply CLI (ADR-0001 b1.6).

Bake tests prove the Golden Rule end-to-end for the v2 path: strength 0 = base (node2) /
identity (between) bit-for-bit. CLI tests run the real subcommands on tiny synthetic pools."""

from __future__ import annotations

import numpy as np
from PIL import Image

from lutgen.cli import main
from lutgen.engine.base import DEFAULT_SIZE, load_base
from lutgen.engine.cube_io import read_cube
from lutgen.engine.grid import identity_grid
from lutgen.fitter.filmmodel import CrosstalkParams, FilmModel, SCurveParams
from lutgen.fitter.fit import FitOptions
from lutgen.orchestration.learn import (
    frame_count_hint,
    learn_profile,
    render_cube_from_profile,
)
from lutgen.orchestration.profile import LookProfile, load_profile

FAST = FitOptions(n_samples=800, max_nfev=8)


def _write_frames(tmp_path, n, tint=(0.05, 0.01, -0.03)):
    """Tiny synthetic 'graded' frames: random scenes sharing one tint."""
    paths = []
    for i in range(n):
        rng = np.random.default_rng(i)
        # spatially smooth frames (real photos correlate spatially; per-pixel noise has a
        # pathological chroma distribution that trips the stress validator by design)
        coarse = rng.uniform(0.1, 0.9, (4, 4, 3))
        smooth = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize(
            (24, 24), Image.BILINEAR), dtype=np.float64) / 255.0
        img = np.clip(smooth + np.asarray(tint), 0, 1)
        p = tmp_path / f"ref{i}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    return paths


def _model() -> FilmModel:
    return FilmModel(crosstalk=CrosstalkParams(gr=0.06),
                     curves=(SCurveParams(toe=0.3, slope=1.2), SCurveParams(),
                             SCurveParams(shoulder=0.4)))


# ── bake: render_cube_from_profile ────────────────────────────────────

def test_apply_strength0_node2_is_base_bit_for_bit():
    cube = render_cube_from_profile(_model(), 0.0, placement="node2")
    np.testing.assert_array_equal(cube.samples, load_base(DEFAULT_SIZE))


def test_apply_strength0_between_is_identity_bit_for_bit():
    cube = render_cube_from_profile(_model(), 0.0, placement="between")
    np.testing.assert_array_equal(cube.samples, identity_grid(DEFAULT_SIZE))


def test_apply_identity_model_full_strength_node2_is_base():
    cube = render_cube_from_profile(FilmModel.identity(), 1.0, placement="node2")
    np.testing.assert_allclose(cube.samples, load_base(DEFAULT_SIZE), atol=1e-12)


def test_apply_nonidentity_changes_output_but_stays_sane():
    cube = render_cube_from_profile(_model(), 1.0, placement="node2")
    base = load_base(DEFAULT_SIZE)
    assert not np.allclose(cube.samples, base)
    assert np.all(cube.samples >= 0.0) and np.all(cube.samples <= 1.0)


def test_apply_accepts_profile_wrapper_and_bad_placement_raises():
    prof = LookProfile(model=_model(), name="x")
    a = render_cube_from_profile(prof, 0.5)
    b = render_cube_from_profile(_model(), 0.5)
    np.testing.assert_array_equal(a.samples, b.samples)
    import pytest
    with pytest.raises(ValueError, match="unknown placement"):
        render_cube_from_profile(prof, 1.0, placement="node5")


def test_apply_strength_interpolates():
    base = load_base(DEFAULT_SIZE)
    half = render_cube_from_profile(_model(), 0.5, placement="node2").samples
    full = render_cube_from_profile(_model(), 1.0, placement="node2").samples
    np.testing.assert_allclose(half, 0.5 * (base + full), atol=1e-12)


# ── learn_profile end-to-end ──────────────────────────────────────────

def test_learn_profile_from_synthetic_pool(tmp_path):
    stages = []
    prof = learn_profile(_write_frames(tmp_path, 6), name="warm", options=FAST,
                         progress=stages.append)
    assert prof.n_frames == 6 and prof.name == "warm"
    assert stages == ["tone", "crosstalk", "coupling", "huesat", "polish", "huesat", "done"]
    assert set(prof.stage_cost) == {"tone", "crosstalk", "coupling", "huesat", "polish"}
    # warm-tinted pool must produce a non-identity recipe
    assert not prof.model.is_identity()


# ── CLI subcommands ───────────────────────────────────────────────────

def test_cli_learn_then_apply(tmp_path, capsys):
    refs = _write_frames(tmp_path, 5)
    prof_path = tmp_path / "look.json"
    cube_path = tmp_path / "look.cube"

    rc = main(["learn", "--refs", *refs, "--out", str(prof_path), "--fast"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "5 frames" in out and "fitting: tone" in out and "wrote" in out
    prof = load_profile(prof_path)
    assert prof.n_frames == 5 and prof.name == "look"

    # --force: a 5-frame draft fit on synthetic noise is legitimately gate-flagged; this
    # test exercises the CLI plumbing, not fit quality (gate behaviour has its own tests)
    rc = main(["apply", "--profile", str(prof_path), "--out", str(cube_path),
               "--strength", "0.8", "--force"])
    assert rc == 0
    cube = read_cube(cube_path)
    assert cube.size == DEFAULT_SIZE
    assert np.all(cube.samples >= 0.0) and np.all(cube.samples <= 1.0)


def test_cli_learn_single_frame_warns(tmp_path, capsys):
    refs = _write_frames(tmp_path, 1)
    rc = main(["learn", "--refs", *refs, "--out", str(tmp_path / "one.json"), "--fast"])
    assert rc == 0
    assert "absorb" in capsys.readouterr().out    # the single-image wall, surfaced


def test_frame_count_hint_tiers():
    assert "absorb" in frame_count_hint(1)
    assert "5+" in frame_count_hint(3)
    assert "good" in frame_count_hint(7)
    assert "excellent" in frame_count_hint(12)


# ── ADR-0007: automatic lighting grouping ─────────────────────────────

def _mood_frames(tmp_path, n, level, tag):
    paths = []
    for i in range(n):
        rng = np.random.default_rng(hash(tag) % 1000 + i)
        coarse = rng.uniform(max(0.02, level - 0.1), min(0.98, level + 0.1), (4, 4, 3))
        img = np.asarray(Image.fromarray((coarse * 255).astype(np.uint8)).resize(
            (24, 24), Image.BILINEAR), dtype=np.float64) / 255.0
        p = tmp_path / f"{tag}{i}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    return paths


def test_grouping_splits_mixed_pool_prefers_brighter(tmp_path):
    from lutgen.orchestration.learn import group_pool_by_lighting
    day = _mood_frames(tmp_path, 5, 0.6, "day")
    night = _mood_frames(tmp_path, 7, 0.12, "night")
    kept, dropped, note = group_pool_by_lighting(day + night)
    assert sorted(kept) == sorted(day)                  # brighter group wins despite minority
    assert sorted(dropped) == sorted(night)
    assert "mixed lighting" in note and "6" not in note.split("frames")[0]


def test_grouping_keeps_coherent_pool_whole(tmp_path):
    from lutgen.orchestration.learn import group_pool_by_lighting
    day = _mood_frames(tmp_path, 6, 0.5, "coh")
    kept, dropped, note = group_pool_by_lighting(day)
    assert kept == day and dropped == [] and note == ""


def test_learn_profile_carries_grouping_note(tmp_path):
    day = _mood_frames(tmp_path, 4, 0.55, "d2")
    night = _mood_frames(tmp_path, 4, 0.1, "n2")
    prof = learn_profile(day + night, options=FAST)
    assert "mixed lighting" in prof.grouping_note


# ── adaptive mode + memory-colour physical fallback (ADR-0008 b8.5 r7) ─

def test_adaptive_single_profile_on_mixed_pool(tmp_path):
    import numpy as np
    from PIL import Image
    from lutgen.orchestration.learn import learn_profile_adaptive
    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):                                   # bright group
        img = (np.clip(0.62 + rng.normal(0, 0.08, (48, 64, 3)), 0, 1) * 255).astype("uint8")
        p = tmp_path / f"b{i}.png"; Image.fromarray(img).save(p); paths.append(str(p))
    for i in range(4):                                   # dark group
        img = (np.clip(0.18 + rng.normal(0, 0.05, (48, 64, 3)), 0, 1) * 255).astype("uint8")
        p = tmp_path / f"d{i}.png"; Image.fromarray(img).save(p); paths.append(str(p))
    from lutgen.fitter.fit import FitOptions
    prof = learn_profile_adaptive(paths, options=FitOptions(n_samples=300, max_nfev=6))
    assert "adaptive" in prof.grouping_note
    assert prof.n_frames == 8
    out = prof.model.forward(np.random.default_rng(1).uniform(0, 1, (16, 3)))
    assert np.all(np.isfinite(out))


def test_memory_guard_physical_fallback_no_nan():
    import numpy as np
    from lutgen.fitter.fit import FitOptions, _memory_residual, _cube_fn, _memory_probes
    from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
    from lutgen.engine.perceptual import to_oklab
    from lutgen.fitter.filmmodel import FilmModel, film_print_character
    inv = _cube_fn(load_base_inverse(), INVERSE_SIZE)
    fwd = _cube_fn(load_base(), DEFAULT_SIZE)
    probes = _memory_probes(-105.0, np.array([0.6, 0.8]), np.array([0.05]))
    lab0 = to_oklab(probes)
    c0 = np.hypot(lab0[:, 1], lab0[:, 2])
    m = FilmModel(film_system=film_print_character())
    r = _memory_residual(m, inv(probes), np.full(len(probes), np.nan),
                         np.full(len(probes), np.radians(-125.0)), c0, fwd, FitOptions())
    assert np.all(np.isfinite(r))
    # the physical fallback tracks the film system itself => corridor cost ~0 for a
    # model whose statistical blocks are neutral
    assert float(np.abs(r).sum()) < 1e-6
