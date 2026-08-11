"""crosstalk — Block A of the v2 film model: a 3x3 dye-crosstalk mixing matrix.

@context  Film dye layers contaminate each other (red exposure slightly stains the green dye,
          etc). A small mixing matrix reproduces the characteristic film hue twist a digital
          grade can't (a film orange != a digital orange) — spec §2.2. This is the first block
          in the fixed pipeline (A crosstalk -> B tone -> C/D in Oklab).
@done     CrosstalkParams (6 off-diagonal), crosstalk_matrix (rows sum to 1), apply_crosstalk.
@todo     -
@limits   PURE: no IO. Matrix is dest-row (row i = how output channel i is mixed from inputs),
          applied as ``rgb @ M.T`` to match engine/spaces.py. Rows sum to 1, so energy per output
          is preserved AND the neutral axis (r=g=b) is preserved (no hue cast on greys); diagonal
          = 1 - off-diagonal row sum keeps it diagonal-dominant for small params. All params 0 ->
          identity matrix -> input returned BIT-FOR-BIT. Vectorized over (...,3) float64.
@affects  Composed before the S-curves in model.FilmModel. See ADR-0001 batch 1.1, spec §3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CrosstalkParams:
    """The 6 off-diagonal dye-mixing amounts. ``xy`` = how much channel x leaks INTO channel y.

    All default 0 (identity). Each row's diagonal is derived as ``1 - (sum of that row's
    off-diagonals)`` so every row sums to 1 (energy preserving). Keep amounts small
    (|v| well under 0.5) to stay diagonal-dominant — the optimizer bounds them in batch 1.4.
    """

    rg: float = 0.0  # red   -> green
    rb: float = 0.0  # red   -> blue
    gr: float = 0.0  # green -> red
    gb: float = 0.0  # green -> blue
    br: float = 0.0  # blue  -> red
    bg: float = 0.0  # blue  -> green

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())


def crosstalk_matrix(params: CrosstalkParams) -> np.ndarray:
    """Build the 3x3 dest-row mixing matrix ``M`` (``M[dest, source]``).

    Row i is how output channel i is mixed from the inputs; each row sums to 1, so the transform
    is applied as ``rgb @ M.T`` (engine/spaces.py convention) and both energy-per-output and the
    neutral axis are preserved. Entry ``xy`` (x leaks into y) sits at ``M[y, x]``. All-zero
    params yield the identity matrix exactly.
    """
    r, g, b = 0, 1, 2
    m = np.zeros((3, 3), dtype=np.float64)
    m[g, r] = params.rg
    m[b, r] = params.rb
    m[r, g] = params.gr
    m[b, g] = params.gb
    m[r, b] = params.br
    m[g, b] = params.bg
    # diagonal absorbs the remainder so each (dest) row sums to 1
    np.fill_diagonal(m, 1.0 - m.sum(axis=1))
    return m


def apply_crosstalk(rgb: np.ndarray, params: CrosstalkParams) -> np.ndarray:
    """Apply the crosstalk mix to ``rgb`` (...,3). New array; identity params return input as-is.

    Uses ``rgb @ M.T`` with ``M`` dest-row and row-stochastic, so a neutral (r=g=b) input stays
    neutral (no hue cast on greys).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {rgb.shape}")
    if params.is_identity():
        return rgb.copy()
    return rgb @ crosstalk_matrix(params).T
