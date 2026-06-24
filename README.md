# LookForge — LUT Generator

Generates a 33-point 3D `.cube` LUT that **replaces Resolve's Node 2** (DaVinci Wide Gamut /
DaVinci Intermediate → Rec.709 Gamma 2.4) and bakes a creative **look** — extracted from your
reference images — on top, driven by one **strength** dial.

Pipeline: `source → [Node 1 → DWG/DI] → [LookForge cube = Node 2] → Rec.709 + look`. The protected
base is your verified Resolve conversion; at `strength = 0` the output equals it exactly.

---

## Install (developers)
```bash
pip install -e ".[dev,gui]"   # first time (gui adds PySide6)
pytest -q                     # 115 offline tests
```

---

## Open the app

**Option A — packaged app (no Python needed):** double-click **`LookForge.exe`** (Windows) from the
`dist/` folder after building (see "Build the app" below). It opens the desktop window directly.

**Option B — from source:**
```bash
lutgen-gui              # if installed (pip install -e .)
# or:
python -m lutgen.app
```

### Using the GUI
1. **Mode** (top-left):
   - **References** — extract the look from graded stills (your reference images).
   - **Before/After pairs** — learn your *exact* grade from matched frames (most accurate).
2. **References mode:** click **+ Add references…**, pick your look stills (e.g. the graded frames).
   Choose:
   - **Fitter:** `Rich` (recommended) or `Mid` (simple baseline).
   - **Method:** `mkl` (palette: mean+covariance) or `pdf` (full distribution, richest).
   - **Space:** `oklab` (perceptual, recommended) or `rgb`.
   - **Tone** slider: lower = keep your footage's brightness; higher = also match the refs' exposure.
3. **Before/After pairs mode:** **+ Add BEFORE** (your frames with the grade OFF — Node 1+2 only) and
   **+ Add AFTER** (the same frames with your grade ON). Equal counts. LookForge learns the exact grade.
4. **Strength** slider: how much of the look to apply (0 = your original, 1 = full look).
5. **Load preview still…** — load a real **DWG/DI** frame (a clip with Node 1 on, Node 2 off) to see a
   faithful *before / after* on the right. (The built-in still is synthetic and just a placeholder.)
6. **Export .cube…** — save the LUT. **Save preset…** — save the recipe (References mode).

### Apply it in Resolve
Replace your **Node 2** (the DWG/DI → Rec.709 CST) with this `.cube` (a LUT or CST node). Keep Node 1
and any adjustments between nodes. The cube does the conversion **and** the look.

---

## Command line
```bash
# References → look cube (Rich, Oklab, full PDF transport):
lutgen render --refs r1.png r2.png r3.png --fitter rich --space oklab --method pdf \
              --strength 0.8 --tone 0.5 --out look.cube --title "My look"

# Unpaired pools: transport your NEUTRAL footage toward GRADED examples (any counts, any scenes):
lutgen render --source neutral1.png neutral2.png --refs graded1.png graded2.png graded3.png \
              --fitter rich --method pdf --out look.cube

# Learn the EXACT grade from before/after frames (same frame, neutral + graded):
lutgen render-pairs --before n1.png n2.png --after g1.png g2.png --out my_grade.cube

# strength 0 = the pure base (your Node 2). Presets:
lutgen render --refs r1.png r2.png --out look.cube --save-preset look.json
lutgen render --preset look.json --out look.cube
```
Flags: `--fitter mid|rich`, `--method mkl|pdf`, `--space oklab|rgb`, `--tone 0..1`, `--strength 0..1`,
`--placement node2|between` (replace Node 2, or a DWG/DI look between Node 1 & 2 keeping both).

---

## Build the app
```bash
pip install -e ".[gui,package]"
pyinstaller --noconfirm packaging/LookForge.spec
# → dist/LookForge.exe  (Windows)
```
The build bundles Python, PySide6, numpy/scipy/colour-science, and the base `.cube` assets into one
file. (macOS: same command produces a `dist/LookForge` binary; wrap into `.app` if desired.)

---

## Fitter quick guide
| Fitter / mode | What it does | Use when |
|---|---|---|
| `mid` | per-channel histogram match | quick, simple |
| `rich --method mkl` | palette transport (mean + covariance) | good default |
| `rich --method pdf` | full distribution transport (Pitié IDT) | richest, references-only |
| `render-pairs` | learns the exact grade from before/after | you can export pairs (best) |

All share the protected base; `strength = 0` is always your untouched Node 2.

---

## Layout
```
src/lutgen/
  engine/         # L1 — base (loaded Resolve cube), grid, spaces, cube_io, strength, regularize, apply, perceptual
  fitter/         # L2 — mid, rich (mkl/pdf, oklab/rgb), pairs — behind one LookFitter interface
  orchestration/  # L3 — ingest, stats, consensus, pipeline, preset
  app/            # L4 — PySide6 desktop shell
  cli.py          # terminal entry
packaging/        # PyInstaller spec + launcher
tests/            # 115 offline tests
```

## Notes
Planning/workflow/PM docs and all images (`*.png`/`*.jpg`) are kept **local / untracked**. Git
history is code-only.
