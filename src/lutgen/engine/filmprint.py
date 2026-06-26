"""filmprint — film print emulation as an alternative conversion (ADR-0022, Opt 3).

@context  Reference-matching is statistical; a measured Print-Film-Emulation (PFE) LUT IS real film.
          This builds an alternative "base": DWG/DI → Cineon → [user PFE .cube] → Rec.709, replacing
          the DaVinci CST tone map with a real film print transfer (e.g. Kodak 2383). The look /
          adjustments / film grade then blend on top, exactly like the DaVinci base.
@done     build_film_base(pfe_path, size, exposure) — verified exposure anchoring (18% grey → ~0.46
          Rec.709 with Juan Melara FilmUnlimited_2383_Standard).
@limits   Needs `colour-science` (DWG transfer + Cineon). The PFE LUT must expect CINEON-log input
          (the classic 2383/2393 print LUTs do); feeding the wrong encoding gives garbage. Output is
          a (size**3,3) Rec.709 g2.4 cube, used as the protected base for Replace-CSTout.
@affects  Used by orchestration/pipeline + the GUI "Conversion" choice. See ADR-0022.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .apply import apply_cube
from .cube_io import read_cube
from .grid import DEFAULT_SIZE, identity_grid


def _dwg():
    import colour
    return colour.RGB_COLOURSPACES["DaVinci Wide Gamut"]


@lru_cache(maxsize=8)
def build_film_base(pfe_path: str, size: int = DEFAULT_SIZE, exposure: float = 0.0) -> np.ndarray:
    """Build a film-print conversion cube: DWG/DI identity → linear → Cineon → PFE → Rec.709.

    ``exposure`` (stops) shifts the linear scene exposure before the Cineon encode, to nudge where
    mid-grey lands (match your Resolve setup). Returns flat ``(size**3, 3)`` Rec.709 samples.
    """
    from colour.models import log_encoding_Cineon

    pfe = read_cube(pfe_path)
    grid = identity_grid(size)                              # DI-encoded DWG identity grid
    linear = _dwg().cctf_decoding(grid) * (2.0 ** float(exposure))
    linear = np.clip(linear, 1e-6, None)                   # Cineon undefined at ≤0 (out-of-gamut)
    cineon = np.clip(log_encoding_Cineon(linear), 0.0, 1.0).astype(np.float64)
    return apply_cube(cineon, pfe.samples, pfe.size)       # Rec.709 print
