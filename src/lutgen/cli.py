"""cli — terminal entry point (MVP).

@context  `lutgen render --refs ... --strength ... --out ...`; preset save/load. Thin wrapper
          over orchestration.pipeline — no color math here.
@done     render command, preset load/save, main(argv).
@todo     GUI is M5 (separate).
@limits   Base stays protected (strength=0 -> base). Out cube already regularized; written as-is.
@affects  Console entry point `lutgen = lutgen.cli:main`. Uses orchestration.pipeline + preset +
          engine.cube_io. See ADR-0006.
"""

from __future__ import annotations

import argparse
import sys

from lutgen.engine.cube_io import write_cube
from lutgen.fitter.mid import MidFitter
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.pipeline import render_cube, render_cube_from_pairs, render_look_cube
from lutgen.orchestration.preset import load_preset, save_preset

_FITTERS = {"mid": MidFitter, "rich": RichFitter}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lutgen", description="Generate a look .cube LUT.")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="render references + strength to a .cube")
    r.add_argument("--refs", nargs="*", default=None, help="reference image paths")
    r.add_argument("--strength", type=float, default=None, help="look strength 0..1 (default 1.0)")
    r.add_argument("--out", "-o", required=True, help="output .cube path")
    r.add_argument("--title", default=None, help="cube TITLE")
    r.add_argument("--fitter", choices=list(_FITTERS), default="rich", help="look fitter (default rich)")
    r.add_argument("--tone", type=float, default=None,
                   help="tonal/exposure match 0..1 (lower preserves your footage brightness)")
    r.add_argument("--space", choices=["oklab", "rgb"], default="oklab",
                   help="rich transport space (default oklab, perceptual)")
    r.add_argument("--preset", default=None, help="load refs/strength/title from a preset JSON")
    r.add_argument("--save-preset", default=None, help="write the settings used to a preset JSON")
    r.add_argument("--max-dim", type=int, default=1024, help="downscale refs to this max side")

    rl = sub.add_parser(
        "render-look",
        help="render a LOOK-ONLY cube applied between Node 1 and Node 2 (DWG/DI, recommended)",
    )
    rl.add_argument("--refs", nargs="*", default=None, help="reference image paths")
    rl.add_argument("--strength", type=float, default=None, help="look strength 0..1 (default 1.0)")
    rl.add_argument("--tone", type=float, default=0.0, help="tonal match 0..1 (default 0 = exposure preserved)")
    rl.add_argument("--out", "-o", required=True, help="output .cube path")
    rl.add_argument("--title", default=None, help="cube TITLE")
    rl.add_argument("--preset", default=None, help="load refs/strength/title from a preset JSON")
    rl.add_argument("--save-preset", default=None, help="write the settings used to a preset JSON")
    rl.add_argument("--max-dim", type=int, default=1024, help="downscale refs to this max side")

    rp = sub.add_parser(
        "render-pairs",
        help="learn the EXACT grade from before/after frame pairs (replaces Node 2)",
    )
    rp.add_argument("--before", nargs="+", required=True, help="neutral frames (Node 1+2, no grade)")
    rp.add_argument("--after", nargs="+", required=True, help="graded frames (same frames, final look)")
    rp.add_argument("--strength", type=float, default=1.0, help="look strength 0..1 (default 1.0)")
    rp.add_argument("--smoothing", type=float, default=0.8, help="grade smoothing sigma (default 0.8)")
    rp.add_argument("--out", "-o", required=True, help="output .cube path")
    rp.add_argument("--title", default=None, help="cube TITLE")
    rp.add_argument("--max-dim", type=int, default=1024, help="downscale frames to this max side")
    return parser


def _resolve(args):
    """Merge CLI flags with an optional --preset (flags win). Returns (refs, strength, title)."""
    refs, strength, title = args.refs, args.strength, args.title
    if args.preset:
        preset = load_preset(args.preset)
        refs = refs if refs else preset["refs"]
        strength = strength if strength is not None else preset["strength"]
        title = title if title is not None else preset["title"]
    return refs, (1.0 if strength is None else strength), title


def _cmd_render(args: argparse.Namespace) -> int:
    refs, strength, title = _resolve(args)
    if not refs:
        print("error: no references (pass --refs or --preset)", file=sys.stderr)
        return 2
    kwargs = {} if args.tone is None else {"tone_strength": args.tone}
    if args.fitter == "rich":
        kwargs["space"] = args.space
    fitter = _FITTERS[args.fitter](**kwargs)
    cube = render_cube(refs, strength, title=title, fitter=fitter, max_dim=args.max_dim)
    write_cube(args.out, cube.samples, cube.size, title=cube.title)
    if args.save_preset:
        save_preset(args.save_preset, refs, strength, title)
    print(f"wrote {args.out}  (replace Node 2, {args.fitter} fitter, {len(refs)} refs, strength {strength})")
    return 0


def _cmd_render_look(args: argparse.Namespace) -> int:
    refs, strength, title = _resolve(args)
    if not refs:
        print("error: no references (pass --refs or --preset)", file=sys.stderr)
        return 2
    cube = render_look_cube(refs, strength, tone_strength=args.tone, title=title, max_dim=args.max_dim)
    write_cube(args.out, cube.samples, cube.size, title=cube.title)
    if args.save_preset:
        save_preset(args.save_preset, refs, strength, title)
    print(f"wrote {args.out}  (DWG/DI look between Node 1&2, {len(refs)} refs, "
          f"strength {strength}, tone {args.tone})")
    return 0


def _cmd_render_pairs(args: argparse.Namespace) -> int:
    if len(args.before) != len(args.after):
        print("error: --before and --after must have the same number of frames", file=sys.stderr)
        return 2
    cube = render_cube_from_pairs(
        args.before, args.after, args.strength,
        smoothing=args.smoothing, title=args.title, max_dim=args.max_dim,
    )
    write_cube(args.out, cube.samples, cube.size, title=cube.title)
    print(f"wrote {args.out}  (learned grade from {len(args.before)} pairs, strength {args.strength})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "render":
        return _cmd_render(args)
    if args.command == "render-look":
        return _cmd_render_look(args)
    if args.command == "render-pairs":
        return _cmd_render_pairs(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
