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


def _fmt_fourier(fh) -> list[str]:
    import numpy as np

    from lutgen.fitter.filmmodel.fourierhue import eval_shift, eval_trim
    if fh.is_identity():
        return ["  neutral"]
    theta = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    shift = np.degrees(eval_shift(theta, fh))
    trim = eval_trim(theta, fh) * 100.0
    rows = []
    if np.abs(shift).max() > 0.2:
        i, j = int(np.argmax(shift)), int(np.argmin(shift))
        rows.append(f"  hue: {shift[i]:+.1f}° @ {math.degrees(theta[i]):.0f}° · "
                    f"{shift[j]:+.1f}° @ {math.degrees(theta[j]):.0f}°")
    if np.abs(trim).max() > 0.5:
        i, j = int(np.argmax(trim)), int(np.argmin(trim))
        rows.append(f"  sat: {trim[i]:+.0f}% @ {math.degrees(theta[i]):.0f}° · "
                    f"{trim[j]:+.0f}% @ {math.degrees(theta[j]):.0f}°")
    from lutgen.fitter.filmmodel.fourierhue import eval_lshift
    lmod = np.degrees(eval_lshift(theta, fh))
    if np.abs(lmod).max() > 0.5:
        i = int(np.argmax(np.abs(lmod)))
        rows.append(f"  brightness-split: ±{abs(lmod[i]) / 2:.1f}° shadows↔highlights "
                    f"@ {math.degrees(theta[i]):.0f}°")
    return rows or ["  neutral"]


def _fmt_filmsystem(fs) -> list[str]:
    if fs.is_identity():
        return ["  neutral"]
    rows = []
    n = fs.negative
    bits = [f"{ch} γ ×{v:.2f}" for ch, v in (("R", n.g_r), ("G", n.g_g), ("B", n.g_b))
            if abs(v - 1.0) > _EPS]
    if abs(n.toe) > _EPS:
        bits.append(f"toe {n.toe:.2f} @ {n.toe_at:+.1f} st")
    if bits:
        rows.append("  negative: " + " · ".join(bits))
    cp = fs.coupling
    coup = [f"{k.upper()} {v:.3f}" for k, v in
            (("rg", cp.rg), ("rb", cp.rb), ("gr", cp.gr),
             ("gb", cp.gb), ("br", cp.br), ("bg", cp.bg)) if abs(v) > _EPS]
    if coup:
        rows.append("  coupling: " + " · ".join(coup))
    li = fs.lights
    bits = [f"{ch} {v:+.2f} st" for ch, v in (("R", li.r), ("G", li.g), ("B", li.b))
            if abs(v) > _EPS]
    if bits:
        rows.append("  printer lights: " + " · ".join(bits))
    pr = fs.printer
    bits = []
    if abs(pr.slope - 1.0) > _EPS:
        bits.append(f"contrast ×{pr.slope:.2f}")
    if abs(pr.shoulder) > _EPS:
        bits.append(f"shoulder {pr.shoulder:.2f} @ +{pr.range_hi:.1f} st")
    if abs(pr.ptoe) > _EPS:
        bits.append(f"black conv {pr.ptoe:.2f} @ {pr.range_lo:+.1f} st")
    if bits:
        rows.append("  print: " + " · ".join(bits))
    return rows or ["  neutral"]


def _fmt_splittone(st) -> list[str]:
    if st.is_identity():
        return ["  neutral"]
    a, b = st.poles()
    names = ("shadow", "dark", "mid", "light", "highlight")
    rows = [f"  {n}: a {va * 1000:+.0f} · b {vb * 1000:+.0f} (‰)"
            for n, va, vb in zip(names, a, b) if abs(va) > _EPS / 5 or abs(vb) > _EPS / 5]
    return rows or ["  neutral"]


def recipe_summary(profile: LookProfile) -> str:
    """The learned recipe as readable grouped text (near-neutral entries omitted)."""
    m = profile.model
    lines = []
    if abs(m.global_trim.exposure) > _EPS:
        lines.append(f"Global\n  exposure {m.global_trim.exposure:+.3f} DI "
                     f"(≈ {m.global_trim.exposure / 0.07:+.1f} stops)")
    lines.append("Film system (negative → print)")
    lines += _fmt_filmsystem(m.film_system)
    lines.append("Tone curves")
    lines += _fmt_curves(m.curves)
    lines.append("Crosstalk")
    lines += _fmt_crosstalk(m.crosstalk)
    lines.append("Split tone (tint by brightness)")
    lines += _fmt_splittone(m.split_tone)
    lines.append("Saturation vs luminance")
    lines += _fmt_satluma(m.sat_luma)
    lines.append("Hue zones")
    lines += _fmt_zones(m.hue_zones)
    lines.append("Hue curve (v2.1)")
    lines += _fmt_fourier(m.hue_fourier)
    if profile.stage_cost:
        cost = " / ".join(f"{k} {v:.3g}" for k, v in profile.stage_cost.items())
        lines.append(f"Fit: {profile.n_frames} frames · cost {cost}")
    return "\n".join(lines)
