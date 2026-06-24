# LookForge — LUT Generator

Desktop app that generates a single **33-point 3D `.cube` LUT** converting footage from
**DaVinci Wide Gamut / DaVinci Intermediate (DWG/DI)** → **Rec.709 Gamma 2.4**, with a
creative **look** extracted from reference images baked on top, controlled by one
**strength** dial.

`final = base + strength · (look − base)` — at `strength = 0` the output equals the exact,
deterministic conversion. The conversion base can never be broken by the look.

## Status
**MVP shipped** (M0–M4). Generate a look `.cube` from reference images:
```bash
lutgen render --refs ref1.png ref2.png --strength 0.8 --out look.cube --title "My look"
```
`--strength 0` emits the pure base; `--save-preset look.json` / `--preset look.json` reuse a recipe.
Apply the `.cube` as Node 2 in Resolve. Next (optional): M5 GUI, M6 Rich fitter.

## Layout
```
src/lutgen/
  engine/         # L1 — color engine: grid, spaces, convert, strength, cube_io, regularize
  fitter/         # L2 — look fitter (Mid MVP, Rich later) behind one interface
  orchestration/  # L3 — references → consensus → look → cube
  app/            # L4 — desktop shell (PySide6)
  cli.py          # terminal entry (MVP)
tests/
```

## Run
```bash
pip install -e ".[dev]"      # first time
pytest -q                    # fast offline suite
```

## Notes
Planning, workflow, and project-management docs are kept **local / untracked** (see
`.gitignore`). Git history is code-only.
