"""grid — 33-node identity lattice for the DWG/DI input domain.

@context  L1's starting point: a regular cube of input coordinates the conversion samples.
          The app never needs source footage; it synthesizes the input domain here.
@done     identity_grid, reshape_to_lattice, flatten_lattice; blue-fastest flat ordering.
@todo     -
@limits   PURE: no IO. Values in [0,1], float64. Ordering is load-bearing (see ORDER note).
@affects  Consumed by engine/convert.py, engine/strength.py, engine/cube_io.py.
          See codemap/INDEX.md (engine/grid.py) + Plan/20_COLOR_PIPELINE.md §5.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIZE = 65

# Flat ordering of grid nodes: red varies FASTEST (innermost), blue SLOWEST (outermost).
# LOCKED against a real Resolve export in ADR-0001 batch 0.6: Resolve's .cube data lines vary
# the RED channel fastest (line 1 = first red step). NOTE: this is the OPPOSITE of the claim
# in Plan/20_COLOR_PIPELINE.md §5 ("blue fastest") — the Plan was wrong; see BUGS.md B-001.
# In C-order this means the lattice axes are (blue, green, red) with red as the last/fastest.
# LOAD-BEARING: must match how cube_io.py writes/reads .cube lines.


def identity_grid(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Return the identity lattice as a flat ``(size**3, 3)`` array in [0, 1].

    Row ``i`` is the RGB coordinate of node ``i`` in red-fastest order. For ``size`` 33 this
    is the 35,937-row table that maps 1:1 to ``.cube`` data lines.
    """
    if size < 2:
        raise ValueError(f"grid size must be >= 2, got {size}")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    # indexing="ij" with positional axes (blue, green, red); C-order reshape makes red (last
    # axis) fastest, matching Resolve's .cube data-line order.
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((r, g, b), axis=-1).reshape(-1, 3)


def reshape_to_lattice(flat: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Reshape a flat ``(size**3, 3)`` array to ``(size, size, size, 3)`` lattice[blue,green,red]."""
    expected = (size ** 3, 3)
    if flat.shape != expected:
        raise ValueError(f"expected {expected}, got {flat.shape}")
    return flat.reshape(size, size, size, 3)


def flatten_lattice(lattice: np.ndarray) -> np.ndarray:
    """Flatten a ``(size, size, size, 3)`` lattice[blue,green,red] to ``(size**3, 3)``, red-fastest."""
    if lattice.ndim != 4 or lattice.shape[3] != 3 or len(set(lattice.shape[:3])) != 1:
        raise ValueError(f"expected (s, s, s, 3) lattice, got {lattice.shape}")
    return lattice.reshape(-1, 3)
