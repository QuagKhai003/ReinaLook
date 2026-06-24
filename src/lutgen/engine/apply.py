"""apply — apply a 3D LUT to an image by trilinear interpolation (for preview).

@context  The GUI preview needs to see a cube applied to a still. Pure numeric; no Qt. Also
          handy for tests/validation. Mirrors how Resolve samples a .cube (trilinear).
@done     apply_cube(image, samples, size): trilinear over the lattice, vectorized.
@todo     -
@limits   PURE: no IO. Input domain clamped to [0,1]. samples are red-fastest (grid layout).
@affects  Uses grid.reshape_to_lattice. Used by app/preview.py. See ADR-0007.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .grid import DEFAULT_SIZE, reshape_to_lattice


def apply_cube(image: np.ndarray, samples: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Apply a 3D LUT to ``image`` (``(..., 3)`` in [0,1]) via trilinear interpolation.

    ``samples`` is the flat ``(size**3, 3)`` cube (red-fastest). Returns the looked image with
    the same leading shape. Input coordinates are clamped to the [0,1] domain.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.shape[-1] != 3:
        raise ValueError(f"expected trailing dim 3, got {image.shape}")

    axis = np.linspace(0.0, 1.0, size)
    lattice = reshape_to_lattice(samples, size)  # indexed [blue, green, red, channel]
    interp = RegularGridInterpolator(
        (axis, axis, axis), lattice, method="linear", bounds_error=False, fill_value=None
    )

    flat = np.clip(image.reshape(-1, 3), 0.0, 1.0)
    # lattice axes are (blue, green, red) → query in that order
    coords = flat[:, ::-1]
    out = interp(coords)
    return out.reshape(image.shape)
