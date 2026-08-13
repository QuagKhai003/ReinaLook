# ADR-0005 — Tone / Color amount dials (scale the recipe, not just blend the cube)

**Status:** Accepted — COMPLETE · 2026-08-12 · Builds on ADR-0004.
Driven by real-world test #3: the do-revenge profile renders the user's bright daylight
footage muddy brown — the pool is dim warm web stills (median 0.226), and refs-only
absolute learning faithfully bakes that in (spec §7 content leakage; not fixable from
screenshots alone).

## Context
The strength dial blends base↔look in cube space — it fades the WHOLE look. What the user
needs on mismatched footage is to keep the film's palette while relaxing its tonal mood
(exposure/contrast). The parametric model makes this trivial and principled: scale each
parameter group toward neutral. This is a capability OT never had — the recipe is the knob.

## Decision & key rules
- Pure `scaled_model(model, tone_amount, color_amount)` in `filmmodel/scale.py`:
  - tone group (G + B): exposure×t; toe/shoulder×t; slope→1+(s−1)t; pivot→0.5+(p−0.5)t.
  - color group (A + C + D): crosstalk×c; sat multipliers→1+(v−1)c; hue shifts/trims×c.
  - t=c=1 → the model unchanged; t=c=0 → identity. Interpolation is convex toward neutral,
    so every bound/monotonicity guarantee survives scaling.
- Apply tab: two sliders (Tone amount, Color amount, 0–100%, default 100) above Strength.
  They modify the BAKED model (preview re-bakes debounced; export uses the scaled model).
  "Save edited profile as…" keeps saving the editor's model UNscaled — dials are per-use.

## Plan (batches)
- [x] **5.1 — scale.py + Apply dials.** Pure scaling (+tests: endpoints, convexity,
      monotonicity preserved, half-exposure) and the two sliders wired into bake/export
      (+tests: dials invalidate cache, export uses scaled model, save-as stays unscaled).
      `Touches: src/lutgen/fitter/filmmodel/scale.py, src/lutgen/app/apply_tab.py, tests/`

## Acceptance
- Tone 30% / Color 100% on do-revenge over the daylight still: exposure near the original,
  palette clearly shifted — no mud. Validation still gates export. Full suite green.
