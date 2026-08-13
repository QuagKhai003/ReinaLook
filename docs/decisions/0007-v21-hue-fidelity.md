# ADR-0007 — Phase 3 / v2.1: hue fidelity (Fourier hue curve, hue×luma, fit retune)

**Status:** Accepted · 2026-08-12 · Builds on ADR-0001/0006. The spec's Phase 3 (§3 v2.1,
§10), pulled forward by user verdict: the engine "does not bring the actual look" — the
film's hue personality is what's missing.

## Context
Three fidelity blockers, in order of blame:
1. **6 chunky hue zones** with one shift+trim each — a film's hue behaviour is smoother and
   far more detailed (skin alone deserves sub-region resolution). Spec calls the Fourier
   replacement "the single most worthwhile upgrade".
2. **No hue×brightness coupling** — "blues go teal in the mids only" is inexpressible.
3. **Fit timidity** — hue/sat detail is regularized nearly to zero and targeted by only 6
   coarse zone means, so even the existing capacity goes unused.
Honest ceiling stands: ~55 params emulate the film's colour science; pixel-matching a still
remains Match (legacy)'s job.

## Decision & key rules
- **Block D v2 — Fourier hue curve** (`fourierhue.py`): hue shift and sat trim as order-4
  Fourier series over the hue wheel (9 coefficients each, 18 params). C∞-smooth periodic —
  no zone boundaries at all. Fitted INSTEAD of the zones; zone params remain for old
  profiles (both apply combined in Block D, new fits leave zones neutral).
- **Block E — hue×luma modulation**: shift(θ, L) gains a luma-linear term
  `(L − 0.5) · F1(θ)` with F1 an order-2 series (5 params). The proper split-tone
  generalization (spec §2.4).
- **Finer targets**: poolstats gains 12-bin hue statistics (`hue_mean_ab`, `hue_weight`)
  and 3×12 hue×luma bins for Block E; conf-damping per bin as everywhere.
- **Retune**: `ridge_huesat` 0.15 → 0.05; hue-target weight raised; the §6 validator
  (hue-sweep continuity + ΔE) remains the safety gate for the loosened fit.
- Param budget: ~34 → ~57 — under the spec's hard ceiling of 80.

## Plan (batches)
- [x] **7.1 — Fourier hue curve.** fourierhue.py (order-4, 18 coefs, C∞), model D = legacy
      zones + Fourier, serialize/editor/summary wiring, SOFT-binned 12-bin hue targets (hard
      bins made the Jacobian inconsistent — stage 3 unoptimizable), angle/ratio-unit residuals,
      spectral-decay bounds + k² curvature ridge, secant 0.02, satluma [0.4,1.7], validator
      recalibrated (channel @5%). Localized-twist ground truth: corr 0.83, peak within a bin.
      Real pools: validate OK, recipes carry film hue personality. Retune (part of 7.3) landed
      inline. +10 tests. `Touches: filmmodel/, poolstats.py, fit.py, validate.py, app/recipe*.py, cli.py, learn_tab.py, tests/`
- [x] **7.2 — Block E luma modulation.** shift(θ,L) = F0(θ) + (L−0.5)·F1(θ), F1 order-2
      (5 coefs, fields l0/lc*/ls*); dark/bright-half hue targets (soft in both axes); fit
      +5 params with spectral bounds ±0.2/k; internal cloud re-adopts the pool's hue
      structure (constants invisible under uniform — safe now the mass residual is gone).
      Honest capability: direction reliably, ~1/3 magnitude on the synthetic worst case, no
      sign flips; real pools learn ±12–14° brightness-splits and validate OK. +5 tests. `Touches: same`
- [ ] **7.3 — Retune + acceptance.** Loosen ridges/weights; re-learn both user pools;
      side-by-side stills vs Match (legacy); §6 validation holds; user eyeball is the gate.
      `Touches: src/lutgen/fitter/fit.py, docs/`

## Acceptance
- Synthetic: a smooth hue twist (e.g. oranges pushed amber, blues pushed teal, mids-only)
  is recovered; the old 6-zone model provably could not express it.
- Real: user judges do-revenge/shirley closer to the reference feel than before, without
  validator violations. Old profiles still load and render unchanged.
