# DATA_MODEL

> The classes/types/tables and their relationships. Update in the SAME change that adds or
> alters one. This is how a reader understands the shapes without reading every file.
>
> **Growth rule (CONVENTIONS §1):** when this outgrows easy scanning, split by domain
> area — `docs/data_model/` with `README.md` as the index (one line per area) and one
> file per area; API contracts get their own file.

## Core types (current)

### v2 film model (fitter/filmmodel/ — ADR-0001)
- **`CrosstalkParams`** (`fitter/filmmodel/crosstalk.py`) — Block A dye mixing; 6 off-diagonal
  floats `rg,rb,gr,gb,br,bg` (x leaks into y). All 0 = identity. Builds a dest-row 3×3 matrix,
  rows sum to 1 (energy + neutral-axis preserving), applied `rgb @ M.T`.
- **`SCurveParams`** (`fitter/filmmodel/scurve.py`) — one channel's Block B tone curve; fields
  `toe≥0, shoulder≥0, slope, pivot∈(0,1)`. Neutral = `toe=0,shoulder=0,slope=1`. Monotone cubic
  Hermite, C¹, endpoints fixed at (0,0)/(1,1).
- **`GlobalParams`** (`fitter/filmmodel/globaltrim.py`) — Block G; `exposure` (DI code offset,
  log space ≈ 0.07/stop, neutral = 0). Applied first in the pipeline; serialized as `global`.
- **`FourierHueParams`** (`fitter/filmmodel/fourierhue.py`) — Block D v2 (ADR-0007): hue shift
  + sat trim as order-4 Fourier series over Oklab hue (9+9 named coefs `s0,sc1..4,ss1..4` /
  `t0,tc1..4,ts1..4`). C∞ periodic; the FITTED hue block (legacy zones remain for old profiles).
- **`SatLumaParams`** (`fitter/filmmodel/satluma.py`) — Block C; multipliers `shadow, mid, high`
  at L = 0/0.5/1 (neutral = 1). C¹ smoothstep curve over Oklab L; scales chroma only.
- **`HueZoneParams`** (`fitter/filmmodel/huezone.py`) — Block D; 6 zones (r,y,g,c,b,m) ×
  `{*_shift` (rad), `*_trim}` (neutral = 0). C¹ periodic interpolation between Oklab
  primary/secondary hue centres.
- **`FilmModel`** (`fitter/filmmodel/model.py`) — the forward transform; `crosstalk` +
  `curves: (SCurveParams×3)` + `sat_luma` + `hue_zones` (~33 params). `forward(rgb)` applies
  G→A→B→(Oklab: C→D) in DWG/DI (~34 params). `identity()`/`is_identity()`. Serialized as Look Profile in 1.5.
- **`FrameStats`** (`orchestration/poolstats.py`) — one frame's Learn targets: `channel_quantiles
  (3×21)`, `mean_lab (3,)`, `chroma_by_band (5,)` + `band_mean_ab (5×2)` (conditional balance, ADR-0006) + `band_weight (5,)`, `zone_mean_ab (6×2)` +
  `zone_weight (6,)`, `black_point`, `white_point`. Weights = data thickness per bin.
- **`PooledTargets`** (`orchestration/poolstats.py`) — median-pooled FrameStats + `n_frames`;
  what the 1.4 fit targets. `neutral_prior()` returns one with canonical ungraded-world values
  (`n_frames=0`).
- **`FitOptions`** (`fitter/fit.py`) — fit knobs: `n_samples, seed, w0`, per-stage ridge,
  residual weights, `max_nfev, diff_step`. Defaults = production.
- **`FitResult`** (`fitter/fit.py`) — `model: FilmModel` + `stage_cost/stage_nfev` per stage
  (tone/crosstalk/huesat) + `n_frames`. The cost dict is the 1.5 profile's fit-quality metadata.
- **`Violation` / `ValidationReport`** (`engine/validate.py`) — §6 stress-gate result;
  `Violation{check: monotonic-tone|delta-e|hue-break|endpoints|range, detail}`;
  `report.ok` / `.summary()`. `diagnose_model` (learn.py) returns `{block name: [violations]}`.
- **`LookProfile`** (`orchestration/profile.py`) — `model` + `name` + `n_frames` +
  `stage_cost/stage_nfev`; `from_fit_result()`. File: versioned JSON
  (`format: reinalook-look-profile`, `version: 1`, `fit{}`, `model{}` via
  `filmmodel/serialize.py` — missing keys neutral, unknown ignored).

<!-- template stubs — fill as real types land
- **`<TypeName>`** (`<file>`) — <what it is>; fields: <key fields>. Relationships: <…>.
-->


### Class reference
<Optional: a per-type field table for the important ones.>

## API contracts
- **`<METHOD> /api/<path>`** `<request>` → `<response shape>`. <one line>.

## Planned persistence (<DB> — Phase N)
Documented before code so the schema is decided up front.

| Table | Key columns | Relationships |
|-------|-------------|---------------|
| `<table>` | <cols> | <…> |
