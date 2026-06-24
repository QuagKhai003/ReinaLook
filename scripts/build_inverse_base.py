"""build_inverse_base — generate the inverse of the base cube (Rec.709 -> DWG/DI).

@context  The "between Node 1 & 2" placement (ADR-0017) maps the look back to DWG/DI via the
          inverse of the Node-2 base. The base's DaVinci tone map compresses highlights ~100:1, so
          the inverse is steep/ill-conditioned — a coarse cube interpolates it poorly. We build a
          HIGHER-RESOLUTION inverse (65-point by default) from a DENSER forward sampling of the base
          to reduce the error (ADR-0018). Slow; run once, bundle the asset.
@usage    python scripts/build_inverse_base.py [inverse_size] [forward_density]
@limits   Out-of-DWG/DI-gamut Rec.709 nodes fall back to nearest. Minutes to build.
@affects  Writes src/lutgen/engine/data/base_inverse_rec709_to_dwg_di.cube. See ADR-0017/0018.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import load_base
from lutgen.engine.cube_io import write_cube
from lutgen.engine.grid import identity_grid

OUT = Path(__file__).resolve().parent.parent / "src/lutgen/engine/data/base_inverse_rec709_to_dwg_di.cube"


def main(inverse_size: int = 65, forward_density: int = 49) -> int:
    base = load_base()                          # 33-point base (DWG/DI -> Rec.709)
    # Denser forward sampling of the base for a better-conditioned Delaunay source.
    fwd_grid = identity_grid(forward_density)   # DWG/DI coords
    fwd_rec709 = apply_cube(fwd_grid, base)     # their Rec.709 values
    print(f"building Delaunay inverse from {forward_density**3} forward samples (slow)…")
    lin = LinearNDInterpolator(fwd_rec709, fwd_grid)
    near = NearestNDInterpolator(fwd_rec709, fwd_grid)

    query = identity_grid(inverse_size)         # Rec.709 coords at the N-cube nodes
    inv = lin(query)
    mask = np.isnan(inv).any(axis=1)
    inv[mask] = near(query[mask])
    inv = np.clip(inv, 0.0, 1.0)

    write_cube(OUT, inv, size=inverse_size, title="LookForge base inverse Rec709->DWG/DI", clamp=True)
    print(f"wrote {OUT}  (size {inverse_size}, {mask.sum()} nodes via nearest fallback)")
    return 0


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:3]]
    raise SystemExit(main(*args))
