"""pipeline — wire the whole stack: references -> .cube (L3 orchestration).

@context  The single end-to-end function the CLI/GUI call. Ties L3 (ingest/consensus) → L2
          (fitter) → L1 (base/strength/regularize) → a Cube. Fitter is injectable so Rich swaps
          in with no caller change.
@done     render_cube(ref_paths, strength, *, title, fitter, max_dim) -> Cube.
@todo     -
@limits   Base stays protected: strength=0 -> base bit-for-bit. Output already regularized/clamped.
@affects  Uses ingest, stats, consensus (L3), fitter (L2), base/strength/regularize/cube_io (L1).
          Called by cli.py. See codemap/INDEX.md + ADR-0006.
"""

from __future__ import annotations

import numpy as np

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import DEFAULT_SIZE, load_base, load_base_inverse
from lutgen.engine.cube_io import Cube
from lutgen.engine.grid import identity_grid
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.interface import LookFitter
from lutgen.fitter.mid import MidFitter

from .consensus import build_consensus
from .ingest import load_references
from .stats import compute_stats


def render_cube(
    ref_paths,
    strength: float = 1.0,
    *,
    title: str | None = None,
    fitter: LookFitter | None = None,
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
) -> Cube:
    """Build a finished `.cube` from reference images at the given strength.

    Steps: load refs → per-image stats → consensus → fit a LookTransform → sample it on the
    protected base → blend by strength → regularize. ``strength = 0`` returns the base exactly.
    """
    fitter = fitter or MidFitter()
    base = load_base(size)

    images = load_references(ref_paths, max_dim=max_dim)
    consensus = build_consensus([compute_stats(img) for img in images])
    look = fitter.fit(consensus)

    look_samples = look(base)
    final = regularize(blend(base, look_samples, strength), size)
    return Cube(size=size, samples=final, title=title)


def render_look_cube(
    ref_paths,
    strength: float = 1.0,
    *,
    tone_strength: float = 0.0,
    title: str | None = None,
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
) -> Cube:
    """Build a LOOK-ONLY cube in DWG/DI space, applied BETWEEN Node 1 and Node 2 (ADR-0009).

    Node 1 and Node 2 stay unchanged; this cube (DWG/DI -> DWG/DI) carries only the creative look,
    so Node 2 still does the technical conversion and brightness/contrast/saturation are preserved.
    References (Rec.709) are pulled into DWG/DI via the inverse base cube. ``strength = 0`` returns
    the identity grid (pass-through). ``tone_strength`` defaults to 0 (exposure preserved; Node 2
    owns tone).
    """
    inverse = load_base_inverse(size)
    grid = identity_grid(size)

    images = load_references(ref_paths, max_dim=max_dim)
    refs_dwg = [apply_cube(img, inverse, size) for img in images]  # Rec.709 -> DWG/DI
    consensus = build_consensus([compute_stats(r) for r in refs_dwg])

    look = MidFitter(tone_strength=tone_strength).fit(consensus, source_samples=grid)
    look_samples = look(grid)
    final = regularize(blend(grid, look_samples, strength), size)
    return Cube(size=size, samples=final, title=title)


def render_cube_dual(
    source_paths,
    target_paths,
    strength: float = 1.0,
    *,
    fitter: LookFitter | None = None,
    title: str | None = None,
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
    sample_cap: int = 200_000,
) -> Cube:
    """REPLACE Node 2 by transporting a NEUTRAL pool toward a GRADED pool (ADR-0016).

    `source_paths` = neutral images (your ungraded footage), `target_paths` = graded images (the
    look). Unpaired, any counts. The fitter (Mid/Rich, mkl/pdf) transports the source colour
    distribution onto the target's, calibrated on real neutral footage rather than a uniform source.
    Learned transform is applied to the protected base, then blended. `strength = 0` returns the base.
    """
    fitter = fitter or MidFitter()
    base = load_base(size)

    targets = load_references(target_paths, max_dim=max_dim)
    consensus = build_consensus([compute_stats(img) for img in targets])

    sources = load_references(source_paths, max_dim=max_dim)
    source_pixels = np.concatenate([img.reshape(-1, 3) for img in sources])
    if source_pixels.shape[0] > sample_cap:
        idx = np.random.default_rng(0).choice(source_pixels.shape[0], sample_cap, replace=False)
        source_pixels = source_pixels[idx]

    look = fitter.fit(consensus, source_samples=source_pixels)
    final = regularize(blend(base, look(base), strength), size)
    return Cube(size=size, samples=final, title=title)


def render_cube_from_pairs(
    before_paths,
    after_paths,
    strength: float = 1.0,
    *,
    smoothing: float = 0.8,
    title: str | None = None,
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
) -> Cube:
    """Build a cube that REPLACES Node 2 by learning the grade from before/after frame pairs
    (ADR-0012). `before` = neutral (Node 1+2), `after` = graded; same frames. The learned grade is
    applied to the protected base, then blended by strength. `strength = 0` returns the base."""
    from lutgen.fitter.pairs import PairsFitter

    base = load_base(size)
    befores = load_references(before_paths, max_dim=max_dim)
    afters = load_references(after_paths, max_dim=max_dim)
    look = PairsFitter(smoothing=smoothing, size=size).fit_from_pairs(befores, afters)
    final = regularize(blend(base, look(base), strength), size)
    return Cube(size=size, samples=final, title=title)
