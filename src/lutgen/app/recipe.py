"""recipe — format a Look Profile as readable text (pure, no Qt).

@context  The fitted recipe is a headline feature (spec §9): users must SEE what was learned
          ("Toe +0.31 · R→G crosstalk 0.04 · Shadow sat −18%"), not just get a black-box file.
          This module renders a LookProfile into grouped human units; kept Qt-free so it is
          headless-testable and reusable by the 2.3 recipe editor.
@done     recipe_summary(profile) -> str (grouped; near-neutral values omitted).
@todo     Two-way editing helpers for the 2.3 edit layer.
@limits   PURE text formatting. Display units: hue shifts in degrees, trims/multipliers in
          percent. Values within EPS of neutral are omitted so the summary shows the look,
          not 33 zeros.
@affects  Used by learn_tab.py (post-fit summary) and later the recipe editor. ADR-0002 b2.1.
"""

from __future__ import annotations

import math

from lutgen.orchestration.profile import LookProfile

_EPS = 5e-3


def _fmt_curves(curves) -> list[str]:
    rows = []
    for name, c in zip(("R", "G", "B"), curves):
        bits = []
        if abs(c.toe) > _EPS:
            bits.append(f"toe {c.toe:+.2f}")
        if abs(c.shoulder) > _EPS:
            bits.append(f"shoulder {c.shoulder:+.2f}")
        if abs(c.slope - 1.0) > _EPS:
            bits.append(f"slope {c.slope:.2f}")
        if abs(c.pivot - 0.5) > _EPS:
            bits.append(f"pivot {c.pivot:.2f}")
        if bits:
            rows.append(f"  {name}: " + " · ".join(bits))
    return rows or ["  neutral"]


def _fmt_crosstalk(ct) -> list[str]:
    pairs = (("R→G", ct.rg), ("R→B", ct.rb), ("G→R", ct.gr),
             ("G→B", ct.gb), ("B→R", ct.br), ("B→G", ct.bg))
    bits = [f"{n} {v:+.3f}" for n, v in pairs if abs(v) > _EPS]
    return ["  " + " · ".join(bits)] if bits else ["  neutral"]


def _fmt_satluma(sl) -> list[str]:
    bits = [f"{n} {v * 100 - 100:+.0f}%" for n, v in
            (("shadow", sl.shadow), ("mid", sl.mid), ("high", sl.high)) if abs(v - 1.0) > _EPS]
    return ["  " + " · ".join(bits)] if bits else ["  neutral"]


def _fmt_zones(hz) -> list[str]:
    rows = []
    for z in ("r", "y", "g", "c", "b", "m"):
        shift = getattr(hz, f"{z}_shift")
        trim = getattr(hz, f"{z}_trim")
        bits = []
        if abs(shift) > _EPS:
            bits.append(f"hue {math.degrees(shift):+.1f}°")
        if abs(trim) > _EPS:
            bits.append(f"sat {trim * 100:+.0f}%")
        if bits:
            rows.append(f"  {z.upper()}: " + " · ".join(bits))
    return rows or ["  neutral"]


def recipe_summary(profile: LookProfile) -> str:
    """The learned recipe as readable grouped text (near-neutral entries omitted)."""
    m = profile.model
    lines = []
    if abs(m.global_trim.exposure) > _EPS:
        lines.append(f"Global\n  exposure {m.global_trim.exposure:+.3f} DI "
                     f"(≈ {m.global_trim.exposure / 0.07:+.1f} stops)")
    lines.append("Tone curves")
    lines += _fmt_curves(m.curves)
    lines.append("Crosstalk")
    lines += _fmt_crosstalk(m.crosstalk)
    lines.append("Saturation vs luminance")
    lines += _fmt_satluma(m.sat_luma)
    lines.append("Hue zones")
    lines += _fmt_zones(m.hue_zones)
    if profile.stage_cost:
        cost = " / ".join(f"{k} {v:.3g}" for k, v in profile.stage_cost.items())
        lines.append(f"Fit: {profile.n_frames} frames · cost {cost}")
    return "\n".join(lines)
