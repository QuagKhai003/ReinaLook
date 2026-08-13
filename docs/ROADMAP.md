# ROADMAP

> Phases and their batches. Status per batch. Detail + acceptance live in the ADRs.
> Parallel mode: add a `Lane` column to the tables (see `docs/PARALLEL.md`).
>
> **Growth rule (CONVENTIONS §1):** many phases → split into `docs/roadmap/`:
> `README.md` as the index (one line per phase: number — name — status), one
> `phase-N-<slug>.md` per phase.

Source of truth for scope: `docs/REINALOOK_V2_FILM_EMULATION_SPEC.md` (§10 build plan).

## Phase 1 — v2 Fitter core (parametric film emulation) — COMPLETE (ADR-0001)
**Goal:** replace v1's free-form OT *matching* with a ~40-param film-shaped *emulation* model
(Blocks A–D), staged fitting + per-region regularization, and a `learn`/`apply` CLI. No UI changes.

| # | Task | Status |
|---|------|--------|
| 1.1 | Forward model skeleton: Block A (3×3 crosstalk) + Block B (per-channel S-curves), pure, identity@0 | ✅ |
| 1.2 | Blocks C (sat-vs-luma) + D (hue-zone trims) in Oklab; full ~40-param forward model | ✅ |
| 1.3 | Pooled robust reference statistics + neutral prior (Learn-mode targets) | ✅ |
| 1.4 | Staged bounded scipy fit (tone → crosstalk → hue/sat) + per-region regularization | ✅ |
| 1.5 | Look Profile JSON schema (save/load recipe + fit-quality metadata) | ✅ |
| 1.6 | LUT bake from profile (+ strength blend, both placements) + `reinalook learn`/`apply` CLI | ✅ |
| 1.7 | Stress-validation harness (color chart, monotonic tone, ΔE smoothness) as automated tests | ✅ |

**Sequence (by dependency):** 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7.
**Acceptance:** in ADR-0001 (fitted output smoother than v1 PDF on the stress chart; plausible on unseen footage; strength 0 = base bit-for-bit).

## Phase 2 — Learn/Apply UI — COMPLETE (ADR-0002)
**Goal:** restructure the app around Learn/Apply: profile library, recipe display + edit,
frame-count guidance, "never laggy" engineering (spec §9).

| # | Task | Status |
|---|------|--------|
| 2.1 | Learn tab: pool + hint + threaded staged fit (Cancel) + save profile | ✅ |
| 2.2 | Apply tab: profile library, strength/placement preview, gated export | ✅ |
| 2.3 | Recipe display + edit layer (debounced re-bake) | ✅ |
| 2.4 | Mode restructure: Learn / Apply / Match (legacy) tabs | ✅ |
| 2.5 | Performance & layout QA (caches, splitter, timings — proxy unneeded at 225 ms) | ✅ |

**Sequence:** 2.1 → 2.2 → 2.3 → 2.4 → 2.5. **Acceptance:** in ADR-0002.

## Phase R — real-pool robustness — COMPLETE (ADR-0003)
Interim fixes from the first real-world smoke test: letterbox auto-crop at ingest;
source world adopts the pool's tone distribution (tone stage learns per-channel deviation).

## Phase G — full-look Learn + multi-still preview — COMPLETE (ADR-0004)
Block G global exposure; refs-only full-look learning (user decision: no source pool in the
workflow); Apply preview holds up to 20 stills with arrows + slider.

## Phase 3 — v2.1 model upgrades — PLANNED
Fourier hue curve (replaces 6 zones), hue×luma-band grid (Block E); re-run acceptance tests.

## Phase 4 — Polish — PLANNED
Source-adaptive white-balance trim in Apply, preset sharing, DPI/layout QA pass.

## Backlog / deferred
- **Pool brightness-spread warning / auto-grouping**: warn when frame median lumas span wide
  (mixed lighting moods); optionally offer to learn from the dominant cluster (ADR-0007 finding).
- **Film-print character prior** (user-parked 2026-08-12): ship a built-in film-shaped
  starting character; Learn fits DEVIATIONS from film instead of from zero — the Dehancer
  architecture. Candidate ADR-0008. Curve/3D visualization: user explicitly not interested.
- Grain / halation / texture — spatial effects, physically impossible in a LUT; out of scope.
- 65³ LUT option (default stays 33³).
