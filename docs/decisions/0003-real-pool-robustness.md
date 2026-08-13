# ADR-0003 — Learn robustness on real reference pools

**Status:** Accepted — COMPLETE · 2026-08-12 · Builds on ADR-0001/0002. Driven by the first real-world
smoke test (13 *Shirley* (2020) stills).

## Context
First real pool exposed two failure classes the synthetic tests couldn't:
1. **Letterbox risk.** Movie screenshots commonly carry black bars; bars inject a false
   black spike into every tone statistic. (This pool happened to be pre-cropped, but the
   failure class is universal for the product's main input source.)
2. **Exposure contamination (the observed failure).** The pool's median luma is 0.114 vs
   the neutral prior world's 0.389 — *Shirley* is simply a dark film. The tone stage tried
   to reproduce that ABSOLUTE darkness, slammed slope/pivot into their bounds
   (2.00/0.70 ×3 channels), and left tone cost 0.524 (10× the other stages). Scene
   brightness is content, not grade (spec §4); the recipe must learn tone SHAPE.

## Decision & key rules
- **Auto-crop letterbox bars at ingest** (default ON): near-black edge rows/columns are
  detected and cropped with a small inset before any statistic is computed. Conservative:
  only crop clear bars; dark SCENES (no pure-black edge runs) are untouched.
- **Exposure-align the source world before fitting** (default ON, prior path only): scale
  the neutral prior's tone quantiles so the synthesized source cloud's median matches the
  reference pool's median. Curves then fit the residual distribution shape. A user-supplied
  real neutral pool is NEVER rescaled — its exposure is measured, not assumed.
- Both behaviours are options (`autocrop`, `FitOptions.exposure_align`) so tests and edge
  cases can disable them.

## Plan (batches)
- [x] **R.1 — Letterbox auto-crop.** autocrop_letterbox + load_image/load_references(autocrop=True).
      +10 tests (bars/pillarbox/single-sided cropped, barless + dark scenes untouched, absurd-crop
      refusal, opt-out). `Touches: src/lutgen/orchestration/ingest.py, tests/`
- [x] **R.2 — Distribution-aligned tone targets.** DESIGN EVOLVED during acceptance: median
      scaling failed (killed source highlights → slopes hit the LOWER bound, cost 0.615). Final:
      the source world adopts the pool's whole luma distribution (neutral per channel) — tone
      stage learns per-channel DEVIATION (the learnable film signal), colour blocks learn at
      matched exposure. Shirley: cost 0.0251, off bounds, validation OK. +5 tests; ground-truth
      fit tests recalibrated to exposure_align=False; acceptance frames made spatially smooth. `Touches: src/lutgen/fitter/fit.py, tests/`

## Acceptance
- Re-Learn on the *Shirley* pool: tone cost drops well under 0.1 and no curve parameter sits
  on a bound; recipe visually plausible in the Apply preview.
- Full suite green; defaults change no existing passing behaviour except where intended.
