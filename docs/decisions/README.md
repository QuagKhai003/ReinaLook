# Decisions (ADRs)

One file per non-obvious decision. Numbered, immutable once accepted (supersede with a new
ADR rather than rewriting history). Each ADR carries a **Plan (batches)** checklist — this is
the resumable work plan the build loop follows.

**Small decisions don't get ADRs — but they DO get logged**: every decision, however
minor, is a one-liner in `LOG.md` (ADR-worthy ones appear in both, linked).

Copy `0000-template.md` → `NNNN-<slug>.md` and link it below.

| # | Title | Status |
|---|-------|--------|
| 0001 | v2: parametric film-emulation fitter (Phase 1, fitter core) | Accepted — COMPLETE |
| 0002 | Phase 2: Learn/Apply UI | Accepted — COMPLETE |
| 0003 | Learn robustness on real reference pools | Accepted — COMPLETE |
| 0004 | Full-look Learn (Block G), refs-only workflow, multi-still preview | Accepted — COMPLETE |
| 0005 | Tone/Color amount dials (scale the recipe) | Accepted — COMPLETE |
| 0006 | Conditional color learning (per-band balance + tamed exposure) | Accepted — COMPLETE |
| 0007 | Phase 3 / v2.1 hue fidelity (Fourier hue, Block E, vividness/brightness contracts, auto grouping) | Accepted — COMPLETE |
| 0008 | Film system core (physical negative→print model) | Accepted |
