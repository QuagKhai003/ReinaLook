# ADR-NNNN — <title>

**Status:** Proposed | Accepted | Accepted — COMPLETE | Superseded by ADR-XXXX · <date>
· Builds on <ADR refs>.

## Context
<The problem / forces. Why we need a decision now. What's broken or missing.>

## Decision & key rules (apply to every batch)
- <The decision, stated plainly.>
- <Rule/constraint that holds across all batches — e.g. "stays pure", "parity on existing
  outputs", "behind a seam".>

## Plan (batches — branch per batch, tested, docs each batch)
> The first unchecked box is "what's next." Tick `[ ]` → `[x]` with a one-line result when
> a batch merges. In parallel mode every batch also declares `Touches:` (files/folders)
> and `(after: N.M)` dependencies — see `docs/PARALLEL.md`.

- [ ] **N.1 — <slice>.** <What it delivers + acceptance in a line.> `Touches: <paths>`
- [ ] **N.2 — <slice>.** <…> `Touches: <paths>` (after: N.1)
- [ ] **N.3 — <slice>.** <…> `Touches: <paths>`

## Acceptance
- <Observable condition 1 that proves the phase is done.>
- <Condition 2. Tests green; build/lint clean; the Golden Rule held.>

## Notes for the executor
- Sequence is by dependency: N.1 → N.2 → N.3 (in parallel mode, `after:` markers are the
  schedule — unmarked batches can run on any free lane).
- Conventions: see `docs/CONVENTIONS.md`. Update STATUS + progress + decisions/LOG
  (+ DATA_MODEL) each batch; update file briefs per the project's brief mode.
- Git: branch per batch (minor fixes gather on `fixes/YYYY-MM`); conventional commits
  under the user's identity, no AI attribution; **merge to main only with the user's
  approval**; no push without approval.
