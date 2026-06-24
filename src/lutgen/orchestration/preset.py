"""preset — save/load a "look" (references + settings) as JSON.

@context  A preset is the reproducible recipe for a look: the reference paths + strength +
          title. Plan §5 / M4. JSON so it's human-editable and diffable.
@done     save_preset / load_preset.
@todo     -
@limits   Stores reference PATHS (not image data). Paths are resolved relative to the caller.
@affects  Used by cli.py. See codemap/INDEX.md + ADR-0006.
"""

from __future__ import annotations

import json
from pathlib import Path

PRESET_VERSION = 1


def save_preset(path: str | Path, refs, strength: float, title: str | None = None) -> None:
    """Write a preset JSON: references + strength + title."""
    data = {
        "version": PRESET_VERSION,
        "refs": [str(r) for r in refs],
        "strength": float(strength),
        "title": title,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_preset(path: str | Path) -> dict:
    """Read a preset JSON. Returns a dict with keys: refs, strength, title."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "refs" not in data:
        raise ValueError(f"invalid preset (no 'refs'): {path}")
    data.setdefault("strength", 1.0)
    data.setdefault("title", None)
    return data
