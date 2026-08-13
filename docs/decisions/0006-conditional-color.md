# ADR-0006 — Conditional color learning (per-band balance targets + tamed global bake)

**Status:** Accepted — COMPLETE · 2026-08-12 · Builds on ADR-0004/0005. Stacked on `phase/5-look-amounts`
(ADR-0005 merge pending).
User direction: the app must learn the film's UNDERLYING color behavior — shadows rendered
cool, highlights warm, per the director's intent — not one averaged cast baked over
everything. (A LUT conditions on the pixel, not the scene; dark-pixel≈night is the honest
approximation, and it is how film stock itself behaves — spec §2.3/§2.4.)

## Context
The fit's color targets are global averages (one mean_lab, zone means): day-warm and
night-cool frames pool into a single tint, and the tone stage's laziest fit — global
darkening — carries the rest. Result: mud on mismatched footage. The statistics must be
CONDITIONAL on brightness, and the global shortcut must cost more than the conditional
explanation.

## Decision & key rules
- **A — per-band balance targets:** `FrameStats/PooledTargets` gain `band_mean_ab`
  (mean Oklab a/b per luminance band, 5×2). The fit residuals include conf-weighted
  band-balance error in EVERY stage — per-channel curves (the shadow-tint mechanism),
  crosstalk, and C/D all get the "shadows cool / highlights warm" signal.
- **C — tame the bake:** the exposure parameter gets its own strong ridge
  (`ridge_exposure`, default 0.4 vs 0.05 for curves) — plain darkening only wins when the
  conditional structure can't explain the data. Bounds stay ±0.3.
- Neutral prior: `band_mean_ab = 0` (gray world at every brightness).

## Plan (batches)
- [x] **6.1 — A + C.** poolstats band_mean_ab (+prior, +tests incl. teal-shadow/warm-highlight
      conditional capture); fit band-balance residuals in all stages + per-param ridge; verify
      pure-exposure ground truth still recovered; re-learn the user's two real pools and
      compare visually. `Touches: src/lutgen/orchestration/poolstats.py, src/lutgen/fitter/fit.py, tests/`

## Acceptance
- Synthetic split-tone ground truth (cool shadows / warm highlights) is reproduced at the
  statistics level: fitted band_mean_ab matches the reference's per band, and the global
  mean alone does NOT explain it.
- On the real pools: recipe carries less global exposure; applied daylight still keeps its
  brightness while shadows/highlights pick up the film's respective casts. Suite green.
