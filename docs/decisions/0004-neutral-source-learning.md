# ADR-0004 — Learn from the user's own footage (neutral+graded), global exposure, multi-still preview

**Status:** Accepted — COMPLETE · 2026-08-12 · Builds on ADR-0001/0003.
Driven by the second real-world test: v2 Apply output visibly weaker than the legacy
Before/After transport on the user's real DWG stills.

## Context
Side-by-side on a real still: the v2 shirley profile barely moves the image while the legacy
transport (real neutral pool → graded pool) lands the film's dark warm mood. Two causes:
1. **v2 Learn never sees the user's footage.** It fits against an assumed world; ADR-0003
   R.2 (correctly) stopped learning absolute tone from screenshots alone — but absolute
   tone IS the bulk of this look. The legacy mode wins because it measures a REAL source
   pool. Spec §4 sanctions the fix: re-implement unpaired Neutral+Graded on the new fitter
   ("fit the film model so that pooled neutral stats → pooled graded stats").
2. **The model lacks the spec's global exposure trim** (§3 budget row "Global:
   exposure/black offset trims" — never implemented). S-curves anchored at (0,0)/(1,1)
   cannot express a global level shift; the pre-R.2 bound-slam was exactly this.

## Decision & key rules
- **Global exposure** joins the model as Block G, applied FIRST: `x + exposure` in DI code
  (log space → a code offset is an exposure change). One bounded param (±0.3 ≈ ±4 stops),
  identity at 0, fitted in the tone stage, serialized under `global`, editable in the recipe.
- **Learn accepts an optional NEUTRAL pool** (GUI second list, CLI `--source`): its pooled
  stats become the fit's source targets. With a real source, exposure alignment is OFF
  (level is measured signal); without one, the ADR-0003 aligned path stays (colour-science-
  only recipe — subtle by design, surfaced in UI copy).
- **Multi-still preview**: the Apply tab holds up to 20 DWG stills, navigated by prev/next
  buttons + an index slider; endpoint cache is per-still and bakes stay threaded/debounced.

## Plan (batches)
- [x] **4.1 — Block G: global exposure.** globaltrim.py (DI code offset, identity@0 bit-for-bit),
      model G→A→B→C→D, serialize "global" (back-compat: old profiles neutral-default), editor row,
      recipe summary in stops, tone stage 13-param vector (±0.3 bounds). Pure-exposure ground truth
      recovered to ±0.05. +6 tests. `Touches: src/lutgen/fitter/filmmodel/, src/lutgen/fitter/fit.py, src/lutgen/app/, tests/`
- [x] **4.2 — SCOPE CHANGED BY USER: refs-only remains THE workflow** (the original agreement —
      learn from reference stills, apply to any footage later; no source pool required). Fix
      delivered instead: exposure_align default OFF — with Block G the level is expressible, so
      the FULL look (tonal mood included) is learned against the spec's normal-world prior.
      learn_profile(source_paths=…) kept as an API-level power path only (no GUI/CLI surface).
      Shirley refs-only: dark/warm/moody like legacy, validation OK. `Touches: src/lutgen/orchestration/learn.py, src/lutgen/fitter/fit.py, tests/`
- [x] **4.3 — Multi-still Apply preview.** Up to 20 stills, ◀/▶ + index slider + n/N label,
      per-still endpoint cache (revisit = instant, no rebake), invalidation on profile/edit/
      placement, stale-bake-safe (result cached, only rendered if still current). +7 tests. `Touches: src/lutgen/app/apply_tab.py, tests/`

## Acceptance
- On the user's still: v2 (learned with the neutral pool) mean/character in the same family
  as the legacy transport — dark, warm, clearly transformed; stress validation OK.
- Profiles without `global` still load (neutral default). Full suite green.
