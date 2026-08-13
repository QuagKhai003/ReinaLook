# ADR-0002 — Phase 2: Learn/Apply UI (profile workflow in the desktop app)

**Status:** Accepted — COMPLETE · 2026-08-12 · Builds on ADR-0001 (fitter core, COMPLETE). All 5 batches merged; acceptance met (staged fit + Cancel responsive, zero-lag strength, gated export with blame, recipe editable <300 ms refresh, legacy intact).
Design source: `docs/REINALOOK_V2_FILM_EMULATION_SPEC.md` §9 (UI/UX + performance) + §10 Phase 2.

## Context
Phase 1 delivered the v2 engine end-to-end on the CLI: `learn` (pool → staged fit → Look
Profile JSON) and `apply` (profile → validated `.cube`). The GUI still only exposes the v1
Before/After foundation mode. Phase 2 restructures the app around the Learn/Apply product
shape: users learn recipes, collect a profile library, inspect/edit fitted values, and export
validated cubes — with the "smooth, never laggy" engineering (§9) built in, not patched later.

## Decision & key rules (apply to every batch)
- **Three top-level modes as tabs: Learn / Apply / Match (legacy).** Each shows only its own
  controls. Legacy = the existing Before/After page, unchanged behaviour.
- **Never block the UI thread.** Every fit/bake/preview ≥ ~50 ms runs in the existing
  `_ComputeThread` pattern; window stays movable; Cancel works for the fit (per-stage
  granularity is enough: check a cancel flag in the fit progress callback).
- **No color math in `app/`** (Golden Rule unchanged): the GUI calls `orchestration/learn.py`,
  `profile.py`, `engine/validate.py` only.
- **Preview discipline:** strength slider = cached-endpoint image lerp (existing pattern);
  recipe edits re-bake debounced (~150 ms); pooled stats cached — recomputed only when the
  pool changes, never per dial tick (§9 caching).
- **Validation gates the GUI export too:** on failure show the per-block blame
  (`diagnose_model`) and require an explicit "Export anyway" — never silently export (§6).
- **Layout rules (§9):** parameter panels inside vertical scroll areas; controls/preview in a
  draggable splitter; thumbnails/text never overflow (ellipsis + tooltip); sane minimum size.
- **Honest guidance:** the Learn pool shows `frame_count_hint` as a persistent colored inline
  label — 1 frame = warning color. No modal nagging.

## Plan (batches — branch per batch, tested, docs each batch)
> The first unchecked box is "what's next." GUI logic that can be tested headless
> (state machines, view-models, formatting) is factored into plain functions/classes and
> tested; Qt widget tests run under `QT_QPA_PLATFORM=offscreen` and skip if PySide6 absent.

- [x] **2.1 — Learn tab.** learn_tab.py (pool + colored hint + draft checkbox + threaded staged
      fit + Cancel + summary + save) / worker.py (shared ComputeThread, Cancelled) / recipe.py
      (pure summary). Tabs mounted in MainWindow. +7 tests. `Touches: src/lutgen/app/, tests/`
- [x] **2.2 — Apply tab + profile library.** apply_tab.py (load + QSettings recents, summary,
      debounced threaded endpoint bake, instant strength lerp, gated export with blame dialog +
      "Export anyway") + qt_image.py (shared dithered pixmap). 3 tabs mounted. +8 tests. `Touches: src/lutgen/app/, tests/`
- [x] **2.3 — Recipe display + edit layer.** recipe_editor.py (spec-table spinbox groups over
      serialize dicts, fit-bound ranges, degree/% units, edited signal); ApplyTab wiring (modified
      flag, debounced re-bake, save-as, export-uses-edits). +9 tests. `Touches: src/lutgen/app/, tests/`
- [x] **2.4 — Mode restructure.** Tabs Learn / Apply / Match (legacy); per-mode window title;
      last mode restored (QSettings); pages self-contained. +2 tests. `Touches: src/lutgen/app/, tests/`
- [x] **2.5 — Performance & layout QA.** Pooled-stats cache (pool_targets seam + LearnTab cache,
      cache-hit proven), Apply splitter, min 900×560, path tooltips. Measured: bake 165 ms +
      still 59 ms = 225 ms full-quality (< 300 ms target — proxy unnecessary); lerp 2.5 ms.
      DPI visual pass flagged for manual QA. `Touches: src/lutgen/app/, src/lutgen/orchestration/learn.py, docs/`

## Acceptance
- Learn: pick 6 frames → staged progress with working Cancel → profile saved; UI responsive
  throughout (window movable while fitting).
- Apply: load profile → tweak strength with zero lag (image lerp) → export runs the §6 gate;
  broken profile shows the offending block and does not export without explicit override.
- Recipe values visible and editable; an edit updates the preview within ~300 ms (debounced).
- Legacy mode behaves exactly as today. Full suite green; `app/` stays math-free.

## Notes for the executor
- Reuse: `_ComputeThread`, `_to_pixmap`, endpoint-cache lerp, `orchestration/learn.py`
  (learn_profile / render_cube_from_profile / validate_baked_cube / diagnose_model),
  `profile.py` (save/load), `serialize.py` (recipe dict for display/edit).
- PySide6 is a `gui` extra — keep all `app/` imports lazy so the CLI never needs Qt.
- Conventions per `docs/CONVENTIONS.md`; record per batch; merge only on user approval.
