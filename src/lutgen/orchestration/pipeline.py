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
from lutgen.engine.base import DEFAULT_SIZE, INVERSE_SIZE, load_base, load_base_inverse
from lutgen.engine.cube_io import Cube
from lutgen.engine.grid import identity_grid
from lutgen.engine.regularize import regularize
from lutgen.engine.strength import blend
from lutgen.fitter.interface import LookFitter
from lutgen.fitter.mid import MidFitter

from .consensus import build_consensus
from .ingest import load_references
from .stats import compute_stats


def _assemble(looked_full: np.ndarray, strength: float, placement: str, size: int) -> np.ndarray:
    """Turn a full-strength looked-base (Rec.709) into the final cube samples for the chosen
    placement, blended by ``strength``.

    - ``"node2"`` (replace Node 2): DWG/DI → Rec.709 + look. ``strength=0`` → base, bit-for-bit.
    - ``"between"`` (between Node 1 & 2): DWG/DI → DWG/DI look only; Node 2 still converts after, so
      brightness/contrast/saturation are preserved. The full look is mapped back to DWG/DI via the
      inverse base cube, then blended with the identity grid. ``strength=0`` → identity (pass-through).
    """
    base = load_base(size)
    if placement == "between":
        # map the look back to DWG/DI via the higher-res inverse (applied at its own size)
        dwgdi_full = apply_cube(regularize(looked_full, size), load_base_inverse(), INVERSE_SIZE)
        return regularize(blend(identity_grid(size), dwgdi_full, strength), size)
    return regularize(blend(base, looked_full, strength), size)


def render_cube(
    ref_paths,
    strength: float = 1.0,
    *,
    title: str | None = None,
    fitter: LookFitter | None = None,
    placement: str = "node2",
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
) -> Cube:
    """Build a finished `.cube` from reference images at the given strength.

    Steps: load refs → per-image stats → consensus → fit a LookTransform → sample on the protected
    base → assemble for ``placement`` ("node2" replace, or "between" Node 1 & 2) → blend by strength.
    """
    fitter = fitter or MidFitter()
    base = load_base(size)

    images = load_references(ref_paths, max_dim=max_dim)
    consensus = build_consensus([compute_stats(img) for img in images])
    look = fitter.fit(consensus)

    final = _assemble(look(base), strength, placement, size)
    return Cube(size=size, samples=final, title=title)


def render_cube_dual(
    source_paths,
    target_paths,
    strength: float = 1.0,
    *,
    fitter: LookFitter | None = None,
    title: str | None = None,
    placement: str = "node2",
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
    sample_cap: int = 200_000,
) -> Cube:
    """Transport a NEUTRAL pool toward a GRADED pool (ADR-0016); ``placement`` node2/between (ADR-0017).

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
    final = _assemble(look(base), strength, placement, size)
    return Cube(size=size, samples=final, title=title)


def render_cube_from_pairs(
    before_paths,
    after_paths,
    strength: float = 1.0,
    *,
    smoothing: float = 0.8,
    title: str | None = None,
    placement: str = "node2",
    max_dim: int | None = 1024,
    size: int = DEFAULT_SIZE,
) -> Cube:
    """Learn the grade from before/after frame pairs (ADR-0012); ``placement`` node2/between (ADR-0017).
    `before` = neutral (Node 1+2), `after` = graded; same frames. `strength = 0` → base/identity."""
    from lutgen.fitter.pairs import PairsFitter

    base = load_base(size)
    befores = load_references(before_paths, max_dim=max_dim)
    afters = load_references(after_paths, max_dim=max_dim)
    look = PairsFitter(smoothing=smoothing, size=size).fit_from_pairs(befores, afters)
    final = _assemble(look(base), strength, placement, size)
    return Cube(size=size, samples=final, title=title)
