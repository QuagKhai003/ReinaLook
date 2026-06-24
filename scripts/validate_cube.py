"""validate_cube — compare our generated base against a reference .cube (ADR-0001 b0.6).

@context  Resolve-parity check: generate the protected base (convert_base over the identity
          grid), load a reference .cube (e.g. a Resolve CST export), align by node, and report
          code-value error + perceptual deltaE2000. Repeatable for each re-export.
@done     CLI: python scripts/validate_cube.py <reference.cube> [--clamp].
@limits   Assumes both cubes share the same (red-fastest) ordering and LUT size. Not a test;
          a diagnostic the operator runs.
@affects  Uses engine.grid/convert/cube_io. See Plan/70_RISKS_AND_VALIDATION.md.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from lutgen.engine.convert import convert_base
from lutgen.engine.cube_io import read_cube
from lutgen.engine.grid import identity_grid


def _to_lab(code_g24: np.ndarray) -> np.ndarray:
    import colour

    linear = np.power(np.clip(code_g24, 0.0, None), 2.4)
    cs = colour.RGB_COLOURSPACES["ITU-R BT.709"]
    xyz = colour.RGB_to_XYZ(linear, cs, apply_cctf_decoding=False)
    return colour.XYZ_to_Lab(xyz, illuminant=cs.whitepoint)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare our base cube to a reference .cube.")
    ap.add_argument("reference", help="path to reference .cube (e.g. Resolve export)")
    ap.add_argument("--clamp", action="store_true", help="clamp our base to [0,1] before compare")
    args = ap.parse_args(argv)

    ref = read_cube(args.reference)
    ours = convert_base(identity_grid(ref.size))
    if args.clamp:
        ours = np.clip(ours, 0.0, 1.0)

    if ours.shape != ref.samples.shape:
        print(f"SIZE MISMATCH: ours {ours.shape} vs ref {ref.samples.shape}", file=sys.stderr)
        return 2

    d = np.abs(ours - ref.samples)
    print(f"reference : {args.reference}  (size {ref.size}, title={ref.title!r})")
    print("-- code-value error (Rec.709 g2.4) --")
    print(f"  max |d| = {d.max():.4f}   mean |d| = {d.mean():.5f}   p95 = {np.percentile(d, 95):.4f}")

    de = np.linalg.norm(_to_lab(ours) - _to_lab(ref.samples), axis=-1)
    print("-- perceptual dE (CIELAB Euclidean, D65) --")
    print(f"  max dE  = {de.max():.2f}   mean dE = {de.mean():.2f}   p95 = {np.percentile(de, 95):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
