"""profile — save/load a Look Profile (the fitted film recipe) as JSON.

@context  Learn mode's output (spec §4): a portable file holding the fitted FilmModel params
          plus fit-quality metadata. Users build a LIBRARY of these and re-apply them forever
          — a better product shape than pairwise matching. Human-readable and hand-editable
          (the recipe IS the feature).
@done     LookProfile + save_profile / load_profile (versioned, validated).
@todo     Preset-sharing niceties (Phase 4).
@limits   The IO seam for profiles (JSON text only). Params payload handled by the pure
          fitter/filmmodel/serialize.py. Strict on format/version; lenient on params content
          (missing -> neutral, unknown -> ignored) so hand edits stay safe.
@affects  Consumes FitResult (fitter/fit.py) via from_fit_result. Used by the learn/apply CLI
          (batch 1.6) and the Phase-2 profile library UI. See ADR-0001 b1.5, spec §4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lutgen.fitter.filmmodel import FilmModel
from lutgen.fitter.filmmodel.serialize import model_from_dict, model_to_dict

PROFILE_FORMAT = "reinalook-look-profile"
PROFILE_VERSION = 1


@dataclass
class LookProfile:
    """A saved look: the fitted model + how it was learned (fit-quality metadata)."""

    model: FilmModel
    name: str = "untitled"
    n_frames: int = 0                                  # frames the look was learned from
    stage_cost: dict = field(default_factory=dict)     # per-stage final cost (fit quality)
    stage_nfev: dict = field(default_factory=dict)
    grouping_note: str = ""                             # lighting auto-grouping info (not saved)

    @classmethod
    def from_fit_result(cls, result, name: str = "untitled") -> LookProfile:
        """Build a profile straight from a :class:`fitter.fit.FitResult`."""
        return cls(model=result.model, name=name, n_frames=result.n_frames,
                   stage_cost=dict(result.stage_cost), stage_nfev=dict(result.stage_nfev))


def save_profile(path: str | Path, profile: LookProfile) -> None:
    """Write the profile as versioned, human-readable JSON."""
    data = {
        "format": PROFILE_FORMAT,
        "version": PROFILE_VERSION,
        "name": profile.name,
        "fit": {
            "n_frames": int(profile.n_frames),
            "stage_cost": {k: float(v) for k, v in profile.stage_cost.items()},
            "stage_nfev": {k: int(v) for k, v in profile.stage_nfev.items()},
        },
        "model": model_to_dict(profile.model),
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_profile(path: str | Path) -> LookProfile:
    """Read a profile JSON. Strict on format/version; params are neutral-defaulted."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict) or data.get("format") != PROFILE_FORMAT:
        raise ValueError(f"not a ReinaLook look profile: {path}")
    if data.get("version") != PROFILE_VERSION:
        raise ValueError(
            f"unsupported profile version {data.get('version')!r} (expected {PROFILE_VERSION}): {path}"
        )
    if "model" not in data:
        raise ValueError(f"profile has no 'model' section: {path}")
    fit = data.get("fit", {})
    return LookProfile(
        model=model_from_dict(data["model"]),
        name=str(data.get("name", "untitled")),
        n_frames=int(fit.get("n_frames", 0)),
        stage_cost={k: float(v) for k, v in fit.get("stage_cost", {}).items()},
        stage_nfev={k: int(v) for k, v in fit.get("stage_nfev", {}).items()},
    )
