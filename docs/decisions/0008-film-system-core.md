# ADR-0008 — The film system core (physical negative→print model, research-driven)

**Status:** Accepted · 2026-08-12 · Builds on ADR-0007 + the three research briefs in
`docs/research/2026-08-12-*` (film physics, industry practice, academic methods).

## Context
Every user complaint traced to one root: we fit a free statistical model from zero against
unreliable pooled means. The research verdict is unanimous:
- Film's look IS a structure: near-linear negative (γ≈0.55/ch, R ~15% flatter) × steep print
  S-curve (γ≈2.6, blacks converge neutral at D-max, highlights to warm paper-white), with
  saturation EMERGING from the curve slope and saturation-INCREASING channel coupling
  (negative off-diagonals, log/density domain — our fitted matrix converges to the opposite).
- Every successful product (Dehancer/Filmbox/Koji/Truelight) is this two-stage physical
  model; none fit free statistics. Kodak patent US7327382 is the exact skeleton.
- Punch lives in chroma tails/spread, not means (Hasler–Süsstrunk; sliced-OT quantile
  losses); JPEG refs need tail debiasing; Oklab needs Hunt-effect compensation.

## Decision & key rules
- **New Block F — FilmSystem** replaces the per-channel display S-curves as the tonal core:
  DI decode → log-exposure (anchored at 18% grey) → per-channel NEGATIVE curve (relative
  γ, soft toe) → DENSITY-domain coupling (3×3, off-diagonals ≤ 0 ⇒ saturation-non-decreasing,
  rows sum 0 ⇒ neutrals preserved) → PRINT curve (per-channel slope, shoulder, black/white
  convergence) → back to DI. **Identity at neutral parameters** (all contracts survive:
  strength-0 sacred, dials, validator).
- **Film character = a preset parameter vector** derived from published Kodak numbers
  (Vision3 γ ratios, 2383 print-through gammas R0.966/G1.063/B1.082, shoulder/toe shape) —
  the Learn fit INITIALIZES there and is prior-anchored there. Weak pool ⇒ film. Honest
  naming per spec §7: "film-print character", not a certified stock.
- **Fit v2 losses:** quantile residuals (tail-weighted), chroma-spread (Hasler) residual,
  per-tile p95 chroma debiasing of references, Hunt compensation (α≈0.15), asymmetric
  under-saturation penalty, soft-l1. Content anchors separate from style terms.
- Keep: Fourier hue personality (as trim on top), Block G exposure (printer-light offsets
  join it), pooling + auto-grouping, §6 validator, Learn/Apply UX, Tone/Color dials.
- Sat-vs-luma Block C retires from the FIT (redundant: emerges from F); stays for old
  profiles.

## Plan (batches)
- [x] **8.1 — FilmSystem block.** *(done 2026-08-12 — stops domain (log2) chosen so all
      thresholds read in stops; knee-low sign fixed: softplus term must ADD (slope 1−a),
      original subtracted and expanded shadows)* filmsystem.py: neg curve + coupling + print curve,
      identity@neutral bit-for-bit, monotone, C1; EMERGENT tests: saturation follows curve
      slope (mid-sat > end-sat at film params), push ±2 stops changes contrast/colour
      filmically, blacks/whites converge neutral. `Touches: src/lutgen/fitter/filmmodel/, tests/`
- [x] **8.2 — Character preset + model integration.** *(done 2026-08-12 — preset gammas
      from LAD print-through ratios (R 0.909 / B 1.018 rel. G), slope 1.40, shoulder/ptoe
      on; F sits between A and legacy B; dials split F: tone = mean γ + toe + print,
      color = γ deviations + coupling)* FilmModel v3 pipeline G→A→F→(Oklab
      hue trim); datasheet-derived preset vector + serialization (old profiles keep the
      legacy blocks path); recipe display. `Touches: filmmodel/, serialize, app/recipe*, tests/`
- [x] **8.3 — Fit v2 losses.** *(done 2026-08-13 — machinery + statistics landed, tests
      green; ablation-driven placement: spread → polish only, sat-tail floor 0.15, tile
      debias = max-of-tile-p95, Hunt per-band; soft-l1 REJECTED (hue corr 0.92→0.38);
      tail_weight/hunt_alpha default OFF pending the b8.4 stage layout — the legacy
      layout has no punch lever for them to feed)* Quantile/spread residuals, JPEG tail debiasing, Hunt term,
      asymmetric chroma, soft-l1; poolstats gains quantile/tile statistics. `Touches: poolstats, fit, tests/`
- [x] **8.4 — Fit v2 wiring.** *(done 2026-08-13 — 5 stages over Block F: tone [G + neg +
      print + PrinterLights, preset x0/anchor, confidence-aware knee ridge] → crosstalk →
      symmetric coupling → hue curve → polish; satluma retired from fit; tail/hunt/spread
      active; synthetic recovery tests ported to v3-class truths; §6 endpoint tol 0.10→
      0.16 for the film range limiter)*
- [ ] **8.5 — Acceptance.** User pools side-by-side vs reference frames + legacy; validator
      green; push-test; user eyeball is the gate. `Touches: docs/`

## Acceptance
- At preset (no learning): output already reads as film (S-contrast, mid-sat punch, neutral
  dense blacks, warm-white highlight rolloff) on the user's stills.
- After Learn on the day pool: hue/sat/luma per-hue tables track the reference frame better
  than the ADR-0007 engine; chroma p90 within 10% of reference; validator OK.
- Push test: ±2-stop input shift changes contrast & colour the way pushed film does.
