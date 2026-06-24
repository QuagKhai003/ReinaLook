"""preview — built-in DWG/DI test still + before/after rendering (pure, no Qt).

@context  The GUI previews a look without the user exporting frames from Resolve. This builds a
          synthetic DWG/DI-domain still and applies the base vs final cube to it.
@done     make_test_still(); load_preview_still(); before_after().
@todo     Ship a real DWG/DI photographic still as an asset (nice-to-have).
@limits   Still is a DWG/DI-domain image in [0,1]. load_preview_still does file IO (a frame the
          user exported from Resolve: Node 1 on, Node 2 off).
@affects  Uses engine.apply.apply_cube + orchestration.ingest. Used by app/main_window.py. ADR-0007.
"""

from __future__ import annotations

import numpy as np

from lutgen.engine.apply import apply_cube
from lutgen.engine.base import DEFAULT_SIZE
from lutgen.orchestration.ingest import load_image


def make_test_still(height: int = 256, width: int = 512) -> np.ndarray:
    """A synthetic DWG/DI-domain still in [0,1]: a red×green color plane over a blue gradient,
    with a grayscale ramp band along the bottom so tonal changes are visible."""
    xs = np.linspace(0.0, 1.0, width)
    ys = np.linspace(0.0, 1.0, height)
    gx, gy = np.meshgrid(xs, ys)
    still = np.stack([gx, gy, 0.5 * (gx + gy)], axis=-1)

    band = max(1, height // 8)
    ramp = np.linspace(0.0, 1.0, width)
    still[-band:, :, :] = ramp[None, :, None]  # grayscale ramp strip
    return still.astype(np.float64)


def load_preview_still(path, max_dim: int | None = None) -> np.ndarray:
    """Load a real DWG/DI frame (Node 1 on, Node 2 off) as the preview still. Full resolution by
    default (max_dim=None); the look is applied off the UI thread so a large still stays responsive."""
    return load_image(path, max_dim=max_dim)


def before_after(
    still: np.ndarray,
    base_samples: np.ndarray,
    final_samples: np.ndarray,
    size: int = DEFAULT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (before, after): the still through the base cube vs the final (look) cube."""
    before = apply_cube(still, base_samples, size)
    after = apply_cube(still, final_samples, size)
    return before, after
