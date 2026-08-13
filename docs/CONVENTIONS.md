# CONVENTIONS — how we keep this repo production-grade

Assume **many people work here**. Optimise for a stranger finding their way. These rules
are mandatory.

## 1. Folder & file structure
- **Split by responsibility into folders.** A folder = one job.
- **Keep files small.** Soft cap **~200 lines** — a signal to look, not a trigger to act.
  **Cohesion decides, not the number:** one concept that genuinely needs 250 lines stays
  one file; a 150-line file doing two jobs splits now. Never carve up a single cohesive
  concept just to satisfy the cap.
- One concept per file; name the file after the concept.
- No "utils" dumping ground. If something needs a home, give its category a folder.
- **A file holding multiple things splits into a structure with an index.** When a file
  contains more than one concept — the line cap usually flags this — or a doc outgrows
  easy scanning, turn it into a folder of small parts plus an **index** that routes
  readers:
  - **Code:** a `<concept>/` folder, one part per sub-concept, with an index/barrel file
    (`index.ts`, `__init__.py`, `mod.rs`, …) that re-exports and names each part's role in
    one line. Callers import from the index — the split is invisible to them.
  - **Docs:** a `<doc>/` folder with `README.md` as the index (one line per part), parts
    as separate files. `progress/` (per month) and `filemap/` (per area) already follow
    this pattern; apply it to any other doc that grows — e.g. `BUGS.md` → `docs/bugs/`
    (index + one file per entry), `decisions/LOG.md` → per-year `LOG-YYYY.md` files.
    Living docs carry their own growth rule in their header.
  The index is part of the structure — update it in the SAME change that adds, moves, or
  removes a part. An index that lies is worse than no split.

## 2. Every source file has a brief
```
<Title> — one line.
@context  What this file is and why it exists.
@done     What is implemented here.
@todo     What's left (or "—").
@limits   Hard constraints (e.g. PURE: no network/IO).
@affects  What it depends on / is depended on by.
```
Where it lives depends on the project's **Brief mode** (chosen at init, recorded in
`CLAUDE.md`):
- **docs/filemap (default)** — the brief is an entry in `docs/filemap/<area>.md`; code
  files stay clean. See `docs/filemap/README.md`.
- **in-file** — the brief sits at the top of the source file, in the language's comment
  syntax.

Either way, update the brief in the SAME change that alters the file's behaviour.

## 3. Two always-current "what's happening" files
- **`docs/STATUS.md`** — the truth for *right now* (active task, next, blockers). Update at
  the start and end of every session.
- **`docs/progress/`** — the history (changelog), one file per month, newest on top.
  **Every change gets an entry**, even small ones.

## 4. Keep the model current
- **`docs/DATA_MODEL.md`** — entities/types/tables + relationships. Update whenever you add
  or change a class, type, or table.

## 5. Decisions & issues are logged, not remembered
- **Every decision** — even small (naming, lib pick, a default) — gets a one-liner in
  `docs/decisions/LOG.md` (date, what, why).
- Non-obvious / architectural choice → ALSO an ADR under `docs/decisions/` (one file per
  decision), linked from `docs/decisions/README.md`.
- Bug/limitation → log in `docs/BUGS.md` immediately.
- The only exclusion: discussion that produced no change and no decision.

## 6. Tests
- Core/logic changes ship a deterministic offline test.
- Network/integration tests are marked separately and never gate the fast loop.

## 7. Definition of "done"
1. Code + file brief updated. 2. Tests green. 3. `STATUS.md` + `progress/` +
   `decisions/LOG.md` (+ `DATA_MODEL.md` if types changed) updated. 4. ADR batch ticked;
   new decision/limitation logged if any. 5. Merge proposed — main moves only with the
   user's approval.

## 8. Token discipline (cheap habits, no tools)
Reading the same files over and over is what costs — not writing. Cut the reading:
- **Don't churn the always-read files** (`CLAUDE.md`, `WORKFLOW.md`, this file). Unchanged
  = re-reading them is nearly free; editing one resets that. Change them only when a rule
  changes.
- **Read the filemap brief, not the whole source file**, when you just need its role.
- **Edit one line; don't rewrite the file.** `progress/` + `decisions/LOG.md` are
  append-only — cheapest writes.
- **Orient narrow:** STATUS + the one active ADR + `git log --oneline`, not the whole tree.
- **Write docs short** — lean lines, no filler; small files stay cheap to read.
- **Record once per batch, at the end** — not continuously.

## 9. Git
- **Identity:** the user's global git config; if unset, ask. **No AI attribution ever** —
  no Co-Authored-By trailers, no "generated with" footers.
- Always branch. Features/phases each get a branch (`feature/…`, `phase/…`); minor fixes
  are **gathered** on a rolling `fixes/YYYY-MM` branch. One logical unit per commit.
- Conventional commits.
- **Merge to main only with the user's explicit approval.** No push without approval.
- Workflow files (`docs/`, `CLAUDE.md`, `.claude/`) are gitignored — local-only by design.
