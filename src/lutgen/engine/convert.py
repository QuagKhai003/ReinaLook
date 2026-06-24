"""convert — EXPERIMENTAL synthesized DWG/DI -> Rec.709 g2.4 conversion (NOT the base layer).

@context  A pure colour-science conversion (DI decode -> DWG->Rec.709 -> g2.4). It does NOT
          match Resolve's real CST, which adds DaVinci proprietary tone mapping + saturation
          compression (mean dE ~40 vs the verified cube). Superseded as the protected base by
          base.py (loaded reference cube) per ADR-0003 / BUGS S-001. Kept for research and as a
          building block the Rich phase may reuse.
@done     convert_base(grid): DI decode -> DWG->Rec.709 linear -> pure g2.4 encode.
@todo     -
@limits   PURE: no IO. Output is raw stage-A (may exceed 1.0). NOT the Golden Rule base.
@affects  Depends on engine/grid.py, engine/spaces.py. Used by scripts/validate_cube.py.
          See codemap/INDEX.md + BUGS.md S-001.
"""

from __future__ import annotations

import numpy as np

from .spaces import di_decode, dwg_to_rec709_linear, rec709_g24_encode


def convert_base(grid: np.ndarray) -> np.ndarray:
    """Apply the protected DWG/DI -> Rec.709 g2.4 conversion to grid samples.

    ``grid`` is DaVinci-Intermediate-encoded DWG color (e.g. from ``identity_grid``), shape
    ``(..., 3)`` in [0, 1]. Returns Rec.709 g2.4-encoded color of the same shape. Output is
    unclamped stage-A (may exceed 1.0 near full code); regularization happens downstream.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape[-1] != 3:
        raise ValueError(f"expected trailing dim 3, got {grid.shape}")
    linear_dwg = di_decode(grid)                 # DI decode -> scene-linear (DWG primaries)
    linear_709 = dwg_to_rec709_linear(linear_dwg)  # gamut convert -> linear Rec.709
    return rec709_g24_encode(linear_709)          # encode -> Rec.709 Gamma 2.4
