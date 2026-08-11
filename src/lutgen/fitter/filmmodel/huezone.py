"""huezone — Block D of the v2 film model: per-hue-zone hue/sat trims (in Oklab).

@context  Certain hue regions render with their own character on film — skin, greens, sky
          (spec §2.4). Six zones (R, Y, G, C, B, M) each carry a hue shift + a saturation trim;
          values between zone centres are interpolated smoothly around the hue wheel, so there
          are never hard zone boundaries (no hue breaks — spec §6 checks this).
@done     HueZoneParams (6 x {shift, trim}) + apply_hue_zones on Oklab (...,3) arrays.
@todo     v2.1 replaces the 6 zones with a low-order Fourier hue curve (spec §3, Phase 3).
@limits   PURE: no IO. Operates on Oklab; rotates hue / scales chroma — L untouched. Neutral
          (all 0) returns input BIT-FOR-BIT. Interpolation is smoothstep-eased between adjacent
          zone centres -> C1 and periodic (no wrap seam). Achromatic pixels (chroma 0) are
          fixed points by construction. Sat trim clamped so chroma multiplier >= 0.
@affects  Applied inside model.FilmModel after Block C. Zone centres derived from the Oklab
          hues of the sRGB/Rec.709 primaries+secondaries. See ADR-0001 b1.2, spec §3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lutgen.engine.perceptual import to_oklab

# Zone centres: Oklab hue angles of the six primaries/secondaries (computed once, exact).
# Order is ascending angle so interpolation walks the wheel; each entry is (name, angle).
_CORNERS = {
    "r": (1.0, 0.0, 0.0),
    "y": (1.0, 1.0, 0.0),
    "g": (0.0, 1.0, 0.0),
    "c": (0.0, 1.0, 1.0),
    "b": (0.0, 0.0, 1.0),
    "m": (1.0, 0.0, 1.0),
}


def _zone_angles() -> tuple[tuple[str, ...], np.ndarray]:
    labs = to_oklab(np.array(list(_CORNERS.values()), dtype=np.float64))
    angles = np.arctan2(labs[:, 2], labs[:, 1])
    order = np.argsort(angles)
    names = tuple(np.array(list(_CORNERS.keys()))[order])
    return names, angles[order]


_ZONE_NAMES, _ZONE_ANGLES = _zone_angles()
_TWO_PI = 2.0 * np.pi


@dataclass
class HueZoneParams:
    """Per-zone trims. ``*_shift`` rotates hue (radians, + = counter-clockwise in Oklab a/b);
    ``*_trim`` scales chroma (0 = unchanged, +0.2 = +20%). All default 0 (identity).
    The optimizer bounds these in batch 1.4 (shift ~ +/-0.35 rad, trim ~ +/-0.5)."""

    r_shift: float = 0.0
    r_trim: float = 0.0
    y_shift: float = 0.0
    y_trim: float = 0.0
    g_shift: float = 0.0
    g_trim: float = 0.0
    c_shift: float = 0.0
    c_trim: float = 0.0
    b_shift: float = 0.0
    b_trim: float = 0.0
    m_shift: float = 0.0
    m_trim: float = 0.0

    def is_identity(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())

    def _by_zone(self, kind: str) -> np.ndarray:
        d = asdict(self)
        return np.array([d[f"{n}_{kind}"] for n in _ZONE_NAMES], dtype=np.float64)


def _interp_periodic(hue: np.ndarray, zone_values: np.ndarray) -> np.ndarray:
    """Smoothly interpolate per-zone values around the hue wheel (C1, periodic).

    Between adjacent zone centres the value is smoothstep-eased, so the derivative is zero AT
    every centre — each zone plateaus at its own value and there is no seam at the wrap.
    """
    # position of each hue within [centre_i, centre_i+1), walking the wheel from the first centre
    rel = (hue - _ZONE_ANGLES[0]) % _TWO_PI
    edges = (_ZONE_ANGLES - _ZONE_ANGLES[0]) % _TWO_PI          # ascending, edges[0] = 0
    idx = np.searchsorted(edges, rel, side="right") - 1          # zone to the left
    nxt = (idx + 1) % len(edges)
    span = (edges[nxt] - edges[idx]) % _TWO_PI
    span = np.where(span == 0.0, _TWO_PI, span)                  # last segment wraps to first
    t = ((rel - edges[idx]) % _TWO_PI) / span
    t = t * t * (3.0 - 2.0 * t)                                  # smoothstep ease
    return (1.0 - t) * zone_values[idx] + t * zone_values[nxt]


def apply_hue_zones(lab: np.ndarray, p: HueZoneParams) -> np.ndarray:
    """Apply the zone hue shifts + sat trims to Oklab values. New array; neutral params return
    the input unchanged (bit-for-bit). L untouched; chroma-0 pixels are unaffected."""
    lab = np.asarray(lab, dtype=np.float64)
    if lab.shape[-1] != 3:
        raise ValueError(f"expected (...,3), got {lab.shape}")
    if p.is_identity():
        return lab.copy()

    a, b = lab[..., 1], lab[..., 2]
    hue = np.arctan2(b, a)
    chroma = np.hypot(a, b)

    shift = _interp_periodic(hue, p._by_zone("shift"))
    mult = np.maximum(1.0 + _interp_periodic(hue, p._by_zone("trim")), 0.0)

    new_hue = hue + shift
    new_chroma = chroma * mult
    out = lab.copy()
    out[..., 1] = new_chroma * np.cos(new_hue)
    out[..., 2] = new_chroma * np.sin(new_hue)
    return out
