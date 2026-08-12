"""cli — terminal entry point (MVP).

@context  `lutgen render --refs ... --strength ... --out ...`; preset save/load. Thin wrapper
          over orchestration.pipeline — no color math here.
@done     render command, preset load/save, main(argv); v2 learn/apply subcommands (ADR-0001):
          learn refs -> Look Profile JSON, apply profile -> .cube (both placements).
@todo     GUI is M5 (separate).
@limits   Base stays protected (strength=0 -> base). Out cube already regularized; written as-is.
@affects  Console entry point `lutgen = lutgen.cli:main`. Uses orchestration.pipeline + preset +
          engine.cube_io. See ADR-0006.
"""

from __future__ import annotations

import argparse
import sys

from lutgen.engine.adjust import Adjustments
from lutgen.engine.cube_io import write_cube
from lutgen.engine.film import FilmStock
from lutgen.fitter.rich import RichFitter
from lutgen.orchestration.pipeline import (
    render_cube,
    render_cube_dual,
    render_cube_from_pairs,
)
from lutgen.orchestration.preset import load_preset, save_preset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lutgen", description="Generate a look .cube LUT.")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="render references + strength to a .cube")
    r.add_argument("--refs", nargs="*", default=None, help="reference (graded/target) image paths")
    r.add_argument("--source", nargs="*", default=None,
                   help="optional NEUTRAL image pool; transports source→refs (unpaired, any counts)")
    r.add_argument("--strength", type=float, default=None, help="look strength 0..1 (default 1.0)")
    r.add_argument("--out", "-o", required=True, help="output .cube path")
    r.add_argument("--title", default=None, help="cube TITLE")
    r.add_argument("--tone", type=float, default=None,
                   help="tonal/exposure match 0..1 (lower preserves your footage brightness)")
    r.add_argument("--placement", choices=["node2", "between"], default="node2",
                   help="node2: replace Node 2 (DWG/DI->Rec.709+look); between: DWG/DI look between Node 1&2")
    # creative adjustments (baked on top of the look; all default 0 = off; refs optional)
    r.add_argument("--contrast", type=float, default=0.0, help="-1 flat .. +1 punchy")
    r.add_argument("--saturation", type=float, default=0.0, help="-1 grey .. +1 vivid")
    r.add_argument("--temperature", type=float, default=0.0, help="-1 cool .. +1 warm")
    r.add_argument("--tint", type=float, default=0.0, help="-1 green .. +1 magenta")
    r.add_argument("--shadows", type=float, default=0.0, help="-1 crush .. +1 lift shadows")
    r.add_argument("--highlights", type=float, default=0.0, help="-1 pull .. +1 lift highlights")
    r.add_argument("--rolloff", type=float, default=0.0, help="0..1 filmic muted highlights")
    # film-stock transfer (reshapes the color science; all default 0 = off)
    r.add_argument("--film-contrast", type=float, default=0.0, help="-1 flat .. +1 filmic S-curve")
    r.add_argument("--film-toe", type=float, default=0.0, help="0..1 matte/lifted blacks")
    r.add_argument("--film-shoulder", type=float, default=0.0, help="0..1 soft highlight roll-off")
    r.add_argument("--film-bleach", type=float, default=0.0, help="0..1 desaturate highlights to white")
    r.add_argument("--film-split", type=float, default=0.0, help="-1 cool-hi/warm-lo .. +1 warm-hi/cool-lo")
    r.add_argument("--film-saturation", type=float, default=0.0, help="-1 grey .. +1 vivid")
    # film-print conversion (Opt 3): replace the DaVinci CST with DWG/DI->Cineon->PFE->Rec.709
    r.add_argument("--pfe", default=None, help="film print emulation .cube (Cineon-input, e.g. 2383)")
    r.add_argument("--pfe-exposure", type=float, default=0.0, help="film exposure offset in stops")
    r.add_argument("--preset", default=None, help="load refs/strength/title from a preset JSON")
    r.add_argument("--save-preset", default=None, help="write the settings used to a preset JSON")
    r.add_argument("--max-dim", type=int, default=1024, help="downscale refs to this max side")

    rp = sub.add_parser(
        "render-pairs",
        help="learn the EXACT grade from before/after frame pairs (replaces Node 2)",
    )
    rp.add_argument("--before", nargs="+", required=True, help="neutral frames (Node 1+2, no grade)")
    rp.add_argument("--after", nargs="+", required=True, help="graded frames (same frames, final look)")
    rp.add_argument("--strength", type=float, default=1.0, help="look strength 0..1 (default 1.0)")
    rp.add_argument("--smoothing", type=float, default=0.025,
                    help="grade smoothing as a fraction of the colour range (default 0.025)")
    rp.add_argument("--placement", choices=["node2", "between"], default="node2",
                    help="node2: replace Node 2; between: DWG/DI look between Node 1&2")
    rp.add_argument("--out", "-o", required=True, help="output .cube path")
    rp.add_argument("--title", default=None, help="cube TITLE")
    rp.add_argument("--max-dim", type=int, default=1024, help="downscale frames to this max side")

    # v2 Learn/Apply (ADR-0001): learn a film-shaped recipe once, apply it forever
    ln = sub.add_parser("learn", help="learn a film-shaped Look Profile from graded reference frames")
    ln.add_argument("--refs", nargs="+", required=True, help="graded reference frames (5-15 varied)")
    ln.add_argument("--out", "-o", required=True, help="output profile JSON path")
    ln.add_argument("--name", default=None, help="profile name (default: output filename)")
    ln.add_argument("--fast", action="store_true",
                    help="quick draft fit (smaller sample cloud, capped iterations)")
    ln.add_argument("--max-dim", type=int, default=1024, help="downscale refs to this max side")

    ap = sub.add_parser("apply", help="bake a saved Look Profile into a .cube")
    ap.add_argument("--profile", required=True, help="Look Profile JSON (from `learn`)")
    ap.add_argument("--strength", type=float, default=1.0, help="look strength 0..1 (default 1.0)")
    ap.add_argument("--placement", choices=["node2", "between"], default="node2",
                    help="node2: replace Node 2; between: DWG/DI look between Node 1&2")
    ap.add_argument("--out", "-o", required=True, help="output .cube path")
    ap.add_argument("--title", default=None, help="cube TITLE (default: profile name)")
    ap.add_argument("--force", action="store_true",
                    help="export even if stress validation fails (not recommended)")
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
    adjust = Adjustments(
        contrast=args.contrast, saturation=args.saturation, temperature=args.temperature,
        tint=args.tint, shadows=args.shadows, highlights=args.highlights,
        highlight_rolloff=args.rolloff,
    )
    film = FilmStock(
        contrast=args.film_contrast, toe=args.film_toe, shoulder=args.film_shoulder,
        highlight_bleach=args.film_bleach, split_warm=args.film_split, saturation=args.film_saturation,
    )
    if not refs and adjust.is_identity() and film.is_identity() and not args.pfe:
        print("error: nothing to do — pass --refs, --preset, --pfe, or some --contrast/--film-*/…",
              file=sys.stderr)
        return 2
    base = None
    if args.pfe:                          # film-print conversion (Opt 3)
        from lutgen.engine.filmprint import build_film_base
        base = build_film_base(args.pfe, exposure=args.pfe_exposure)
    kwargs = {} if args.tone is None else {"tone_strength": args.tone}
    fitter = RichFitter(**kwargs)         # fixed: Rich / pdf / Oklab
    if args.source:                       # unpaired neutral→graded transport (ADR-0016)
        cube = render_cube_dual(args.source, refs, strength, fitter=fitter, title=title,
                                placement=args.placement, adjust=adjust, film=film, base=base,
                                max_dim=args.max_dim)
        note = f"{len(args.source)} source + {len(refs)} target"
    else:
        cube = render_cube(refs, strength, title=title, fitter=fitter,
                           placement=args.placement, adjust=adjust, film=film, base=base,
                           max_dim=args.max_dim)
        note = f"{len(refs)} refs" if refs else "manual grade (no refs)"
    write_cube(args.out, cube.samples, cube.size, title=cube.title)
    if args.save_preset:
        save_preset(args.save_preset, refs, strength, title)
    where = "between Node 1&2" if args.placement == "between" else "replace Node 2"
    print(f"wrote {args.out}  ({where}, rich pdf/oklab, {note}, strength {strength})")
    return 0


def _cmd_render_pairs(args: argparse.Namespace) -> int:
    if len(args.before) != len(args.after):
        print("error: --before and --after must have the same number of frames", file=sys.stderr)
        return 2
    cube = render_cube_from_pairs(
        args.before, args.after, args.strength,
        smoothing=args.smoothing, title=args.title, placement=args.placement, max_dim=args.max_dim,
    )
    write_cube(args.out, cube.samples, cube.size, title=cube.title)
    print(f"wrote {args.out}  (learned grade from {len(args.before)} pairs, strength {args.strength})")
    return 0


def _cmd_learn(args: argparse.Namespace) -> int:
    from pathlib import Path

    from lutgen.fitter.fit import FitOptions
    from lutgen.orchestration.learn import frame_count_hint, learn_profile
    from lutgen.orchestration.profile import save_profile

    print(frame_count_hint(len(args.refs)))
    options = (FitOptions(n_samples=1200, max_nfev=30, ridge_huesat=0.25)  # draft:
               # stiff colour ridge — small clouds make wiggly hue curves
               if args.fast else None)
    profile = learn_profile(
        args.refs,
        name=args.name or Path(args.out).stem,
        max_dim=args.max_dim,
        options=options,
        progress=lambda stage: print(f"fitting: {stage}"),
    )
    save_profile(args.out, profile)
    cost = ", ".join(f"{k} {v:.4g}" for k, v in profile.stage_cost.items())
    print(f"wrote {args.out}  (learned from {profile.n_frames} frames; fit cost: {cost})")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    from lutgen.orchestration.learn import (
        diagnose_model,
        render_cube_from_profile,
        validate_baked_cube,
    )
    from lutgen.orchestration.profile import load_profile

    profile = load_profile(args.profile)
    cube = render_cube_from_profile(profile, args.strength,
                                    title=args.title or profile.name, placement=args.placement)

    report = validate_baked_cube(cube, args.placement)   # spec §6: mandatory before export
    if not report.ok:
        print("stress validation FAILED:", file=sys.stderr)
        blamed = diagnose_model(profile.model, args.strength, placement=args.placement)
        for block, problems in blamed.items():
            for p in problems:
                print(f"  {block}: {p}", file=sys.stderr)
        for v in report.violations:                       # anything not attributable to a block
            if not any(str(v) in ps for ps in blamed.values()):
                print(f"  (unattributed): {v}", file=sys.stderr)
        if not args.force:
            print("not exported. Re-learn with more/varied frames, or --force to override.",
                  file=sys.stderr)
            return 3
        print("exporting anyway (--force).", file=sys.stderr)

    write_cube(args.out, cube.samples, cube.size, title=cube.title)
    where = "between Node 1&2" if args.placement == "between" else "replace Node 2"
    print(f"wrote {args.out}  ({where}, profile '{profile.name}', strength {args.strength}, "
          f"validation {'OK' if report.ok else 'FORCED'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "render":
        return _cmd_render(args)
    if args.command == "render-pairs":
        return _cmd_render_pairs(args)
    if args.command == "learn":
        return _cmd_learn(args)
    if args.command == "apply":
        return _cmd_apply(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
