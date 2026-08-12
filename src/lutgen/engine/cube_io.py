"""cube_io — read & write Resolve-compatible 3D .cube LUTs.

@context  Stage output of L1: turn a flat sample table into a .cube file Resolve loads, and
          parse one back. The sample ORDER is load-bearing and must match grid.py.
@done     Cube dataclass; write_cube / read_cube; blue-fastest ordering; optional clamp.
@todo     -
@limits   Ordering = red slowest / blue fastest (Plan §5), same constant as grid.py. Locked
          vs a real Resolve export in ADR-0001 b0.6. Default writes raw values (may exceed 1);
          pass clamp=True for a [0,1] deliverable cube.
@affects  Mirrors grid.py ordering. Used by CLI/app to emit look.cube; read_cube used by the
          Resolve-parity test (b0.6). See codemap/INDEX.md + Plan/20_COLOR_PIPELINE.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SIZE = 65


@dataclass
class Cube:
    """A parsed 3D LUT. ``samples`` is flat ``(size**3, 3)`` in blue-fastest order."""

    size: int
    samples: np.ndarray
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    title: str | None = None

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float64)
        expected = (self.size ** 3, 3)
        if self.samples.shape != expected:
            raise ValueError(f"samples must be {expected}, got {self.samples.shape}")


def write_cube(
    path: str | Path,
    samples: np.ndarray,
    size: int = DEFAULT_SIZE,
    *,
    title: str | None = None,
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    clamp: bool = False,
    decimals: int = 6,
) -> None:
    """Write a flat ``(size**3, 3)`` sample table (blue-fastest) to a ``.cube`` file.

    With ``clamp=True`` values are clipped to ``[domain_min, domain_max]`` for a clean
    deliverable; otherwise raw values are written (may exceed 1.0 — super-white).
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape != (size ** 3, 3):
        raise ValueError(f"expected {(size ** 3, 3)}, got {samples.shape}")
    if clamp:
        samples = np.clip(samples, domain_min, domain_max)

    lines: list[str] = []
    if title is not None:
        lines.append(f'TITLE "{title}"')
    lines.append(f"LUT_3D_SIZE {size}")
    lines.append("DOMAIN_MIN " + " ".join(f"{v:.6f}" for v in domain_min))
    lines.append("DOMAIN_MAX " + " ".join(f"{v:.6f}" for v in domain_max))
    fmt = f"%.{decimals}f"
    lines.extend(" ".join(fmt % v for v in row) for row in samples)

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_cube(path: str | Path) -> Cube:
    """Parse a ``.cube`` file into a :class:`Cube` (samples in file order = blue-fastest)."""
    size: int | None = None
    title: str | None = None
    dmin = (0.0, 0.0, 0.0)
    dmax = (1.0, 1.0, 1.0)
    rows: list[tuple[float, float, float]] = []

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split(None, 1)[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "TITLE":
            title = line.split('"')[1] if '"' in line else line.split(None, 1)[1]
        elif key == "DOMAIN_MIN":
            dmin = tuple(float(v) for v in line.split()[1:4])  # type: ignore[assignment]
        elif key == "DOMAIN_MAX":
            dmax = tuple(float(v) for v in line.split()[1:4])  # type: ignore[assignment]
        elif key in ("LUT_1D_SIZE", "LUT_3D_INPUT_RANGE"):
            raise ValueError(f"unsupported .cube directive: {key}")
        else:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"malformed data line: {raw!r}")
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if size is None:
        raise ValueError("missing LUT_3D_SIZE")
    if len(rows) != size ** 3:
        raise ValueError(f"expected {size ** 3} data lines, got {len(rows)}")
    return Cube(size=size, samples=np.asarray(rows, dtype=np.float64),
                domain_min=dmin, domain_max=dmax, title=title)
