# Research brief — film stock physics (agent 2/3, 2026-08-12)

Distilled for ADR-0008 (full sources in agent report; datasheet PDFs + curve images saved in
the session scratchpad).

## The two-stage system (the core architecture fact)
- **Negative** ≈ near-LINEAR in log exposure: γ ≈ 0.50–0.60 per channel (Vision3 5219:
  G/B ≈ 0.57, **R ≈ 0.48 — red runs ~15% flatter**), straight over 9–13 stops, toe at
  ~−3 stops, no practical shoulder. The negative is a capture device, not the look.
- **Print** (2383) = the LOOK: hard S-curve, γ_mid ≈ 2.6, D-min 0.05 → D-max 4.1, ~6-stop
  input window. System gamma = 0.55 × 2.6 ≈ **1.4–1.6** (deliberately >1 for dark-surround).
- Highlight rolloff = print shoulder; shadows = neg toe hitting print's steep region then
  clamping NEUTRAL at D-max (blacks dense and neutral, highlights desaturate to warm-white).

## Numbers an emulator anchors on
- Per-stop neg density ≈ 0.17D; Cineon: 0.002D/CV, black 95, white 685.
- Per-channel print-through inequality: Digital LAD aim gammas **R 0.966 / G 1.063 /
  B 1.082** — equal-log neutrals do NOT track neutral (source of film's dark-color feel).
- 18% gray anchors: neg Status M ≈ 0.80/1.20/1.60; print ≈ 1.0 visual.
- **Saturation gain ∝ local slope of the composed tone curve** (≈1 across mid ~6 stops,
  →0 at both ends) **plus a mid-density interimage (DIR) boost** — film's sat-vs-luma is
  DERIVED from the tone curve, not independent.
- Hue twist: R-γ lower + toe/shoulder offsets + density-dependent interimage → hues rotate
  a few degrees across exposure; the ~590 nm skin crossover is engineered STABLE.
- Halation: red-dominant bloom (3%/0.3%/0.1% RGB, σ≈200 µm) — out of LUT scope, note only.
- Grain: shadow-max on screen (inverse of digital) — out of LUT scope.

## Machine-readable data (the fast path)
- **agx-emulsion (GitHub)**: digitized JSON profiles — sensitivities, characteristic curves,
  dye densities for Vision3/Portra/Fuji/print papers + physics pipeline with DIR/masking.
- vkdt filmsim: 256-sample LUT rows per stock.
- Kodak datasheets (5219, 2383, 2393) digitizable at ±0.05D.

## Implication for ReinaLook (feeds ADR-0008)
The "film-character prior" should be the REAL two-stage structure: a fixed neg(γ per
channel, toe) ∘ print(S-curve, D-max neutral convergence) backbone built from published
curves, with saturation-follows-slope derived, and the LEARNED part = deviations
(per-channel gamma trims, hue personality, interimage strength). This is exactly the
architecture the physics dictates and what our from-zero statistical fit lacked.
