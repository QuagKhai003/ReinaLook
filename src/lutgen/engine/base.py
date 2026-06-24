"""base — the protected base layer, loaded from the verified Resolve base cube.

@context  The DWG/DI -> Rec.709 g2.4 conversion uses DaVinci's PROPRIETARY tone mapping +
          saturation compression (CST: Tone Mapping=DaVinci Adapt 9.00, 10000->100 nits; Gamut
          Mapping=Saturation Compression 0.900/1.000; Forward OOTF + White Point Adaptation on).
          These cannot be reproduced in open code, so the protected base is the user's verified
          Resolve export, shipped as a fixed package asset. See BUGS S-001 / ADR-0003.
@done     load_base(): load+cache the bundled base cube samples (red-fastest, size 33).
@todo     -
@limits   PURE: read-only load of a bundled asset. The base is fixed (identical for all clips);
          look + strength only ever blend ON TOP of it. This is the Golden Rule's base.
@affects  Reads data/base_dwg_di_to_rec709_g24.cube via cube_io. Used by strength.py / render.
          Replaces convert.py as the protected base (convert.py is now experimental-only).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files

import numpy as np

from .cube_io import Cube, read_cube

DEFAULT_SIZE = 33
INVERSE_SIZE = 65   # the inverse is higher-resolution (steep ill-conditioned map; ADR-0018)
BASE_ASSET = "base_dwg_di_to_rec709_g24.cube"
INVERSE_ASSET = "base_inverse_rec709_to_dwg_di.cube"  # Rec.709 -> DWG/DI (ADR-0009/0018)


@lru_cache(maxsize=4)
def _load_asset_cube(asset: str, size: int) -> Cube:
    source = files("lutgen.engine").joinpath("data", asset)
    with as_file(source) as path:
        cube = read_cube(path)
    if cube.size != size:
        raise ValueError(f"asset {asset} is size {cube.size}, expected {size}")
    cube.samples.setflags(write=False)  # cached: guard against mutation
    return cube


def _load_base_cube(size: int) -> Cube:
    return _load_asset_cube(BASE_ASSET, size)


def load_base(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Return the protected base samples as a flat ``(size**3, 3)`` array (red-fastest).

    This is the verified DWG/DI -> Rec.709 g2.4 conversion (with DaVinci tone map + saturation
    compression baked in) — the fixed base the look blends over. ``strength = 0`` must equal
    this, bit-for-bit. Returns a fresh writable copy each call.
    """
    return _load_base_cube(size).samples.copy()


def load_base_inverse(size: int = INVERSE_SIZE) -> np.ndarray:
    """Return the inverse base cube samples (Rec.709 -> DWG/DI), flat ``(size**3, 3)``.

    Maps Rec.709 color back into DWG/DI — used by the "between Node 1 & 2" placement (ADR-0017) and
    the log-space pipeline. Higher-resolution (65-point, ADR-0018) because the inverse is steep.
    Precomputed asset; returns a fresh copy.
    """
    return _load_asset_cube(INVERSE_ASSET, size).samples.copy()
