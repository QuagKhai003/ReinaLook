"""learn — the v2 Learn/Apply workflow: reference pool -> Look Profile -> baked .cube.

@context  The two-phase product shape (spec §4): LEARN a film-shaped recipe from 5-15 graded
          frames once, then APPLY any saved profile to bake a .cube. This module wires
          ingest -> poolstats -> fit (learn) and profile -> FilmModel -> grid bake (apply).
@done     learn_profile(ref_paths, source_paths=None, targets=...) -> LookProfile — refs-only
          is THE workflow (ADR-0004: full look vs the normal-world prior, Block G absorbs
          level); source_paths is an optional power path (real neutral pool = measured source
          world); pool_targets (the cacheable ingest+stats half); render_cube_from_profile;
          frame_count_hint (the single-image wall, surfaced per spec §4); validate_baked_cube
          + diagnose_model (§6 stress gate with per-block attribution — the CLI export gate).
@todo     Source-adaptive white-balance trim in Apply (Phase 4).
@limits   L3: file IO via ingest only. Bake: the model is natively DWG/DI -> DWG/DI, so
          "between" is a direct forward pass and "node2" composes the base AFTER the model.
          strength=0 -> base ("node2") / identity ("between") bit-for-bit — the Golden Rule
          blend runs in the cube's own output space (ADR-0002).
@affects  Uses ingest + poolstats (L3), fitter/fit (L2), engine grid/base/apply/strength/
          regularize (L1), profile.py. Called by cli.py learn/apply. See ADR-0001 b1.6.
"""

from __future__ import annotations

import numpy as np

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import DEFAULT_SIZE, load_base
from lutgen.engine.cube_io import Cube
from lutgen.engine.grid import identity_grid
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.engine.validate import ValidationReport, validate_cube
from lutgen.fitter.filmmodel import FilmModel
from lutgen.fitter.fit import FitOptions, ProgressFn, fit_film_model

from .ingest import load_image, load_references
from .poolstats import compute_frame_stats, pool_stats
from .profile import LookProfile


def frame_count_hint(n: int) -> str:
    """The honest quality guidance for a pool of ``n`` frames (spec §4 — surface, don't hide)."""
    if n <= 1:
        return ("1 frame: the look will absorb this scene's colors (scene and grade are "
                "indistinguishable from one image). Add 4+ varied frames for a clean profile.")
    if n < 5:
        return f"{n} frames: usable, but 5+ varied frames give a genuinely good profile."
    if n < 10:
        return f"{n} frames: good pool."
    return f"{n} frames: excellent — about as close as physics allows without the negative."


def pool_targets(ref_paths, *, max_dim: int | None = 1024):
    """Load a reference pool and compute its pooled Learn targets (the ingest+stats half of
    learn_profile). Split out so callers can CACHE it — the spec §9 rule: pooled statistics
    are recomputed only when the pool changes, never per fit/dial (a re-Learn with an
    unchanged pool skips straight to the fit)."""
    images = load_references(ref_paths, max_dim=max_dim)
    return pool_stats([compute_frame_stats(img) for img in images])


# a pool whose frames span this much median luma mixes lighting moods (day + night grades
# blend into neither — proven on a real pool, ADR-0007); the dominant group is used
_GROUP_SPREAD = 0.18
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def group_pool_by_lighting(ref_paths, *, max_dim: int | None = 1024):
    """Split a reference pool into lighting groups when it mixes moods (ADR-0007).

    Returns ``(kept_paths, dropped_paths, note)``. Frames are grouped by median luma with a
    2-means threshold; when the spread is small the pool is one mood and everything is kept.
    The larger group wins (ties: the brighter one); the note is surfaced in the UI/CLI so the
    exclusion is never silent, and the dropped frames make the film's OTHER look."""
    paths = list(ref_paths)
    meds = []
    for p in paths:
        img = load_image(p, max_dim=max_dim)
        meds.append(float(np.median(img.reshape(-1, 3) @ _LUMA)))
    meds = np.array(meds)
    if meds.max() - meds.min() <= _GROUP_SPREAD or len(paths) < 4:
        return paths, [], ""
    thr = (meds.max() + meds.min()) / 2.0
    for _ in range(8):                                   # 1-D 2-means
        lo, hi = meds[meds <= thr], meds[thr < meds]
        if not len(lo) or not len(hi):
            break
        new = (lo.mean() + hi.mean()) / 2.0
        if abs(new - thr) < 1e-6:
            break
        thr = new
    bright = meds > thr + 0.04                           # stragglers near the threshold are
    ambiguous_lo = np.abs(meds - thr) <= 0.04            # neither mood: exclude them
    # prefer the BRIGHTER group when it can carry a fit (>= 3 frames): looks are usually
    # applied to normally-exposed footage, and dim web frames are the less trustworthy half
    dark = (~bright) & (~ambiguous_lo)
    pick_bright = bright.sum() >= 3 or bright.sum() >= dark.sum()
    chosen = bright if pick_bright else dark
    keep = [p for p, k in zip(paths, chosen) if k]
    drop = [p for p, k in zip(paths, chosen) if not k]
    mood = "brighter" if pick_bright else "darker"
    note = (f"mixed lighting detected: learned from the {mood} {len(keep)} of "
            f"{len(paths)} frames; the {len(drop)} excluded frames are the film's other "
            f"look — learn them as a separate profile")
    return keep, drop, note


def learn_profile(
    ref_paths,
    source_paths=None,
    *,
    name: str = "untitled",
    max_dim: int | None = 1024,
    options: FitOptions | None = None,
    progress: ProgressFn | None = None,
    targets=None,
    source_targets=None,
) -> LookProfile:
    """LEARN: reference frames -> fitted Look Profile (spec §4 Learn mode).

    With ``source_paths`` (a pool of the USER'S OWN neutral frames — ADR-0004), their pooled
    statistics become the fit's measured source world: the absolute tone reshape from that
    footage to the graded look becomes real, learnable signal (this is the unpaired
    Neutral+Graded mode on the parametric fitter). Without a source pool, the fit runs
    against the tone-aligned assumed world (colour-science-only recipe — subtler by design).
    ``targets``/``source_targets`` accept precomputed :func:`pool_targets` (caching seam).
    """
    grouping_note = ""
    if targets is None:
        kept, _dropped, grouping_note = group_pool_by_lighting(ref_paths, max_dim=max_dim)
        targets = pool_targets(kept, max_dim=max_dim)
    if source_targets is None and source_paths:
        source_targets = pool_targets(source_paths, max_dim=max_dim)
    result = fit_film_model(targets, source=source_targets, options=options,
                            progress=progress)
    profile = LookProfile.from_fit_result(result, name=name)
    profile.grouping_note = grouping_note
    return profile


def render_cube_from_profile(
    profile: LookProfile | FilmModel,
    strength: float = 1.0,
    *,
    title: str | None = None,
    placement: str = "node2",
    size: int = DEFAULT_SIZE,
) -> Cube:
    """APPLY: bake a saved profile (or bare model) into a finished ``.cube``.

    The model transforms DWG/DI -> DWG/DI, so:
    - ``"node2"``  (replace Node 2): grid -> model -> base lookup = DWG/DI -> Rec.709 + look.
      ``strength=0`` -> the base, bit-for-bit.
    - ``"between"`` (between Node 1 & 2): grid -> model, still DWG/DI. ``strength=0`` -> identity.
    """
    model = profile.model if isinstance(profile, LookProfile) else profile
    grid = identity_grid(size)
    looked_di = np.clip(model.forward(grid), 0.0, 1.0)   # model may overshoot; cube domain is [0,1]

    if placement == "between":
        final = regularize(blend(grid, looked_di, strength), size)
    elif placement == "node2":
        base = load_base(size)
        looked = apply_cube(looked_di, base, size)        # base conversion AFTER the film model
        final = regularize(blend(base, looked, strength), size)
    else:
        raise ValueError(f"unknown placement {placement!r} (expected 'node2' or 'between')")
    return Cube(size=size, samples=final, title=title)


# ── §6 stress-validation gate (mandatory before export) ──────────────

def _reference_for(placement: str, size: int) -> np.ndarray:
    return identity_grid(size) if placement == "between" else load_base(size)


def validate_baked_cube(cube: Cube, placement: str = "node2") -> ValidationReport:
    """Run the spec §6 stress checks on a baked cube (tone reversals, ΔE smoothness,
    hue-wheel continuity, endpoints) against the placement's reference conversion."""
    ref = _reference_for(placement, cube.size)
    return validate_cube(
        cube.samples, cube.size, ref,
        interp=lambda x: apply_cube(x, cube.samples, cube.size),
        reference_interp=lambda x: apply_cube(x, ref, cube.size),
    )


_BLOCK_STACKS = (
    ("crosstalk", lambda m: FilmModel(crosstalk=m.crosstalk)),
    ("tone curves", lambda m: FilmModel(crosstalk=m.crosstalk, curves=m.curves)),
    ("sat-vs-luma", lambda m: FilmModel(crosstalk=m.crosstalk, curves=m.curves,
                                        sat_luma=m.sat_luma)),
    ("hue zones", lambda m: FilmModel(crosstalk=m.crosstalk, curves=m.curves,
                                      sat_luma=m.sat_luma, hue_zones=m.hue_zones)),
)


def diagnose_model(model: FilmModel, strength: float = 1.0, *,
                   placement: str = "node2", size: int = DEFAULT_SIZE) -> dict[str, list[str]]:
    """Attribute §6 violations to the model block that introduces them.

    Bakes the model with blocks enabled progressively (A -> A+B -> A+B+C -> full); a check
    that first fails when block X joins is X's fault. Returns {block name: [violations]} —
    empty dict = clean."""
    blamed: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name, stack in _BLOCK_STACKS:
        cube = render_cube_from_profile(stack(model), strength, placement=placement, size=size)
        report = validate_baked_cube(cube, placement)
        fresh = [str(v) for v in report.violations if str(v) not in seen]
        if fresh:
            blamed[name] = fresh
            seen.update(fresh)
    return blamed
