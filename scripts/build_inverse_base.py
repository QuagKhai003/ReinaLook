"""build_inverse_base — generate the inverse of the base cube (Rec.709 -> DWG/DI).

@context  The log-space look pipeline (ADR-0009) needs to pull Rec.709 reference images back
          into DWG/DI. That's the inverse of the Node-2 base cube. Inverting a scattered 3D LUT
          is slow (Delaunay over 35,937 points), so we precompute it ONCE here and bundle the
          result as a 33-point asset that loads fast via trilinear apply.
@usage    python scripts/build_inverse_base.py  (writes the bundled asset; run once)
@limits   Out-of-DWG/DI-gamut Rec.709 nodes fall back to nearest. ~3-4 min to build.
@affects  Writes src/lutgen/engine/data/base_inverse_rec709_to_dwg_di.cube. See ADR-0009.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from lutgen.engine.base import load_base
from lutgen.engine.cube_io import write_cube
from lutgen.engine.grid import identity_grid

OUT = Path(__file__).resolve().parent.parent / "src/lutgen/engine/data/base_inverse_rec709_to_dwg_di.cube"


def main() -> int:
    base = load_base()          # DWG/DI grid coords -> Rec.709 values
    grid = identity_grid()      # the DWG/DI coords
    print("building Delaunay inverse (slow)…")
    lin = LinearNDInterpolator(base, grid)
    near = NearestNDInterpolator(base, grid)

    query = identity_grid()     # Rec.709 coords at the 33-cube nodes
    inv = lin(query)
    mask = np.isnan(inv).any(axis=1)
    inv[mask] = near(query[mask])
    inv = np.clip(inv, 0.0, 1.0)

    write_cube(OUT, inv, title="LookForge base inverse Rec709->DWG/DI", clamp=True)
    print(f"wrote {OUT}  ({mask.sum()} nodes via nearest fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
