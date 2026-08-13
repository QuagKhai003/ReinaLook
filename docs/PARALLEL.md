# PARALLEL — running the workflow with concurrent lanes

> Off by default; the base loop in `WORKFLOW.md` is serial (one batch per cycle). Enable
> this at initialization (or later, any time) when the user wants parallel development.
> Everything in `WORKFLOW.md` still applies — this file only adapts it for concurrency.
> When enabled, `CLAUDE.md` records: `Parallel mode: ON — <methodology>`.

## When to use it
Two or more batches with **no dependency between them**, or several contributors/agents
working at once. If every batch depends on the previous one, stay serial — parallel buys
nothing and adds merge cost.

## Pick ONE methodology at enable time (record the choice in CLAUDE.md)
1. **Worktree lanes (default)** — one repo, one `git worktree` per lane:
   `git worktree add ../<project>-<lane> <branch>`. One agent or human per lane, each with
   its own working copy and branch. Best for a solo user driving multiple agents.
2. **Trunk-based short branches** — for several human contributors: batches stay small
   (< 1 day), merge to `integration` at least daily, incomplete work hides behind feature
   flags. Best when everyone communicates in real time.
3. **Integrator pattern** — N executor lanes plus one dedicated integrator who owns all
   merges, resolves doc conflicts, and keeps `integration` green. Executors never merge.
   Best when the lanes are agents and the user wants a single point of control.

## The rules that make parallel safe
1. **Batches declare `Touches:`.** In parallel mode every ADR batch lists the files and
   folders it will change. Two active lanes must have **disjoint** Touches. Overlap →
   serialize those batches (or re-slice them until they don't overlap).
2. **Lane = branch (+ worktree).** Branch naming: `lane/<name>/<type>-<slug>`.
3. **`docs/STATUS.md` gets a Lanes table** — one row per lane: lane, active batch, branch,
   state, blocker. Each lane edits **only its own row**.
4. **Per-lane progress files** — `docs/progress/YYYY-MM-<lane>.md`, so appends never
   conflict. Entries in `docs/decisions/LOG.md` are prefixed with the lane: `[lane-a] …`.
5. **Integration branch.** Lanes merge to `integration` when their batch is green
   (self-serve; the integrator does it in methodology 3). `integration` → `main` happens
   **only with the user's explicit approval** — same rule as serial mode. Run the full
   test suite on `integration` after every lane merge.
6. **Shared-doc ownership.** Shared docs (`ROADMAP.md`, `DATA_MODEL.md`, `docs/filemap/`)
   are edited by the lane that owns the touched area; conflicts are resolved by the
   integrator (or the user) at integration time, never force-pushed over.
7. **Same definition of "done".** Every lane finishes every batch per `WORKFLOW.md` §6 —
   including recording. No lane skips docs because "another lane will do it".

## Dependencies between batches
Mark them in the ADR plan: `(after: N.M)`. A lane may only pick up a batch whose `after`
targets are already merged to `integration`. The dependency graph — not the batch
numbering — is the schedule; anything unmarked is fair game for any free lane.

## Dropping back to serial
Finish or merge all lane branches to `integration`, get user approval to merge to `main`,
remove extra worktrees (`git worktree remove …`), set `Parallel mode: OFF` in `CLAUDE.md`,
and collapse the Lanes table in `STATUS.md` back to the single Active task section.
