# ADR-0001 — v2: parametric film-emulation fitter (Phase 1, fitter core)

**Status:** Accepted — COMPLETE · 2026-08-11 · Foundation ADR (first phase of v2). All 7 batches done; acceptance verified in test_validate.py::test_acceptance_fitted_smoother_than_v1_pdf.
Full design: `docs/REINALOOK_V2_FILM_EMULATION_SPEC.md`.

## Context
v1 has two disconnected halves: a free-form optimal-transport *matcher* (~108k DOF over the
33³ grid — memorizes *this reference's* statistics, content included, and extrapolates badly
on unseen colors) and a manual Film Stock panel (real film-shaped math, but nobody learns its
values). Neither does the actual goal: **learn a film-shaped color science *from* the
reference.** With screenshot-only input there is no extra information to extract — the only
thing that can change is the **form** of the transform. Constraining it to a ~40–80 parameter
film-shaped model (vs 108k free DOF) is the entire upgrade: emulation (parametric fitting)
instead of matching (distribution warp). Phase 1 builds the fitter core; UI comes in Phase 2.

## Decision & key rules (apply to every batch)
- **A parametric film model becomes the primary engine.** Fixed pipeline in DWG/DI:
  A) 3×3 crosstalk matrix → B) per-channel S-curves → C) sat-vs-luma curve (Oklab) →
  D) hue-zone trims (Oklab). ~40 params in v2.0. v1 OT/MKL/PDF is kept as "Match (legacy)".
- **Purity (Golden Rule):** the whole model lives in `fitter/` (+ `engine/` helpers) and stays
  pure — no I/O, no network, no AI. Ingest/stats/bake I/O stays in `orchestration/`.
- **Identity at params = 0**, and **strength 0 = base bit-for-bit** — v1's core promise is untouched.
- **Every curve strictly monotonic + C¹ smooth**, parameter ranges bounded by the optimizer
  (toe/shoulder ∈ [0,max], slope ∈ [0.5,2.0]) — no fitted result can be degenerate.
- **Staged fitting + per-region regularization are non-negotiable** (spec §5): fit tone → crosstalk
  → hue/sat, freezing/anchoring prior stages; relax toward neutral where reference data is thin.
- **Deterministic scipy**, robust pooled statistics; no per-pixel Python loops (vectorized NumPy).
- **Learn/Apply shape:** Learn a pool of 5–15 reference frames → a portable Look Profile (JSON
  recipe + fit metadata); Apply a profile → a `.cube`. Pooling averages out scene content; the
  shared transform is the grade.

## Plan (batches — branch per batch, tested, docs each batch)
> The first unchecked box is "what's next." Tick `[ ]` → `[x]` with a one-line result when a batch merges.

- [x] **1.1 — Forward model skeleton (Blocks A + B).** 3×3 crosstalk (dest-row, rows sum to 1,
      neutral-preserving) + per-channel monotone-cubic-Hermite S-curves; identity@0 bit-for-bit; pure.
      22 tests (identity, monotonicity, C¹, energy, endpoints, channel independence). `Touches: src/lutgen/fitter/filmmodel/, tests/`
- [x] **1.2 — Blocks C + D (Oklab).** satluma (C¹ smoothstep over L) + 6-zone hue/sat trims (C¹
      periodic, no wrap seam) + DI↔Oklab bridge in engine/perceptual.py (lossless); FilmModel
      composes A→B→C→D (~33 params). +20 tests. `Touches: src/lutgen/fitter/filmmodel/, src/lutgen/engine/perceptual.py, tests/`
- [x] **1.3 — Pooled reference statistics + neutral prior.** New orchestration/poolstats.py
      (v1 stats.py untouched): per-frame Oklab targets + thickness weights, median pooling
      (outlier-proof, tested), documented neutral_prior(). +18 tests. `Touches: src/lutgen/orchestration/poolstats.py, src/lutgen/fitter/filmmodel/huezone.py, tests/`
- [x] **1.4 — Staged bounded fit + per-region regularization.** fit_film_model: synthetic source
      cloud, base round-trip, 3 bounded trf stages (tone→crosstalk→hue/sat, frozen), conf-damped
      residuals + ridge-to-neutral. Ground-truth recovery + thin-zone neutrality proven. +8 tests. `Touches: src/lutgen/fitter/fit.py, tests/`
- [x] **1.5 — Look Profile schema (JSON).** serialize.py (pure model↔dict, exact round-trip,
      hand-edit safe) + profile.py (LookProfile, versioned JSON, strict envelope, from_fit_result).
      +15 tests. `Touches: src/lutgen/fitter/filmmodel/serialize.py, src/lutgen/orchestration/profile.py, docs/DATA_MODEL.md, tests/`
- [x] **1.6 — Bake + CLI.** learn.py (learn_profile + render_cube_from_profile: DI→DI direct
      "between", base-after-model "node2", strength 0 bit-for-bit both) + `reinalook learn`
      (--fast, guidance, staged progress) / `reinalook apply`. +10 tests. `Touches: src/lutgen/orchestration/learn.py, src/lutgen/cli.py, tests/`
- [x] **1.7 — Stress-validation harness.** engine/validate.py (slope-fraction tone check, grey diag,
      ΔE vs reference, hue sweep, endpoints) + diagnose_model per-block blame + CLI apply export
      gate (--force override). Found+fixed C/D scene-Oklab corner explosion (→ code-space Oklab).
      ADR acceptance proven: fitted v2 ≤ v1 PDF roughness. +14 tests. `Touches: src/lutgen/engine/validate.py, src/lutgen/orchestration/learn.py, src/lutgen/cli.py, src/lutgen/fitter/filmmodel/model.py, tests/`

## Acceptance
- On one reference pool, fitted-model output is **smoother on the stress chart** than v1's PDF
  engine and **visibly plausible on unseen footage** (no hue breaks / extrapolation artifacts).
- `reinalook learn` on a 5–15 frame pool produces a valid Look Profile; `reinalook apply`
  bakes a `.cube` that at **strength 0 equals the base conversion bit-for-bit**.
- Full suite green, offline, deterministic; `fitter/` + `engine/` stay pure (Golden Rule held).
- A thin-data hue region stays neutral (regularization verified), not fitted to noise.

## Notes for the executor
- Sequence is strict by dependency 1.1 → 1.7. Reuse v1 where the spec §8 says "keep": CST base,
  ingest, `.cube` writer, strength blend, Film Stock panel math (becomes the init + edit layer later).
- Conventions: `docs/CONVENTIONS.md`. Update STATUS + progress + decisions/LOG (+ DATA_MODEL on 1.5)
  each batch; keep in-file briefs current in the same change.
- Git: branch `phase/1-v2-fitter` (or per-batch branches off it); conventional commits under the
  user's identity, no AI attribution; **merge to main only with the user's approval**.
