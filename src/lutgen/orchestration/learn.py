"""learn — the v2 Learn/Apply workflow: reference pool -> Look Profile -> baked .cube.

@context  The two-phase product shape (spec §4): LEARN a film-shaped recipe from 5-15 graded
          frames once, then APPLY any saved profile to bake a .cube. This module wires
          ingest -> poolstats -> fit (learn) and profile -> FilmModel -> grid bake (apply).
@done     learn_profile(ref_paths) -> LookProfile; render_cube_from_profile(profile) -> Cube;
          frame_count_hint (the single-image wall, surfaced per spec §4).
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
from lutgen.fitter.filmmodel import FilmModel
from lutgen.fitter.fit import FitOptions, ProgressFn, fit_film_model

from .ingest import load_references
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


def learn_profile(
    ref_paths,
    *,
    name: str = "untitled",
    max_dim: int | None = 1024,
    options: FitOptions | None = None,
    progress: ProgressFn | None = None,
) -> LookProfile:
    """LEARN: reference frames -> fitted Look Profile (spec §4 Learn mode).

    Loads the pool, computes robust pooled statistics, runs the staged fit against the
    neutral prior, and wraps the result as a savable profile.
    """
    images = load_references(ref_paths, max_dim=max_dim)
    targets = pool_stats([compute_frame_stats(img) for img in images])
    result = fit_film_model(targets, options=options, progress=progress)
    return LookProfile.from_fit_result(result, name=name)


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
