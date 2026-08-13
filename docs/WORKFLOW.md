# WORKFLOW — the operating loop

This is the heart of the kit: how to develop so the work stays correct, legible, and
resumable. Read it once; then it runs in the background.

> The split-with-index rule (`CONVENTIONS.md` §1) applies to the **whole workflow**, this
> file included: every growth-prone doc carries its own growth rule in its header, and if
> this file itself is ever extended past easy scanning (house rules, project-specific
> sections), split it into `docs/workflow/` — `README.md` as the index, one file per
> section group.

## 0. The mental model
> **Durable state lives on disk, not in the conversation.** Git history (the code) plus
> the `docs/` state files (`STATUS.md`, ADR checklists, progress, the decision log) are
> the source of truth. The chat is scratch space and will be lost.

Note: the workflow/docs files are **gitignored by default** (see §13) — durable on disk,
invisible to the repo. The repo history stays clean product code; the process files still
let any agent — fresh, reset, or human — pick up in minutes without guessing.

## 1. The Golden Rule (define one per project)
Pick **one inviolable architectural principle** and write it at the top of `CLAUDE.md`.
It must be specific enough to forbid concrete actions. Examples:
- "The core computes; the AI only explains. An LLM never produces a number the core uses."
- "The domain layer is pure: no I/O, no framework imports."
- "The UI never calls a third-party API directly; everything goes through the gateway."

When you're about to violate it, **stop**. One clear rule prevents most architectural rot.

## 2. The always-current files (never let these go stale)
- **`CLAUDE.md`** (repo root) — auto-loaded project memory: what it is, the Golden Rule,
  stack, where things live, how to run, current state, the working agreement, and the
  project's mode choices (brief mode, parallel mode). Short.
- **`docs/STATUS.md`** — the single source of truth for *right now*: the active task,
  what's next, blockers (plus the Lanes table in parallel mode). Update at the **start
  and end of every session**. An out-of-date status file is worse than none.
- **`docs/progress/YYYY-MM.md`** — the changelog (history), newest entry on top, one file
  per month. Append when a unit of work finishes — **every** change, even small ones.
- **`docs/decisions/LOG.md`** — the running decision log: **every** decision, one line
  each, even the little ones (see §11).

## 3. The rest of the living docs
- **`docs/CONVENTIONS.md`** — the hygiene contract (small files, one concept per file,
  a brief for every file, folder-per-responsibility; a file that accumulates multiple
  concepts is split into a folder of parts with an index — cohesion decides, the ~200-line
  cap only flags). Mandatory, not advisory.
- **`docs/ROADMAP.md`** — phases, each a table of batches with status.
- **`docs/DATA_MODEL.md`** — entities/types/tables + relationships. Update in the *same*
  change that adds or alters a class/table.
- **`docs/decisions/NNNN-*.md`** — one ADR per non-obvious decision, each with a **Plan
  (batches)** checklist and an **Acceptance** section. This is the resumable work plan.
- **`docs/filemap/`** — per-file briefs, the default brief mode (see §7).
- **`docs/BUGS.md`** — log a bug/limitation the moment you hit it, don't wait.
- **`docs/ONBOARDING.md`** — routes a newcomer to the right files by role.
- **`docs/PARALLEL.md`** — the parallel-development supplement; applies only when
  parallel mode is ON (see §12).

## 4. Plan work as ADR batches
For any phase of work, write an ADR with:
- **Context** — why.
- **Decision & rules** — what + the constraints that apply to every batch.
- **Plan (batches)** — a checklist of small, independently shippable units:
  ```
  - [ ] 1.1 <one shippable slice> — <acceptance in a line>
  - [ ] 1.2 ...
  ```
  In parallel mode each batch also declares `Touches:` (files/folders it will change)
  and, where relevant, `(after: N.M)` dependencies — see `docs/PARALLEL.md`.
- **Acceptance** — how you know the whole phase is done.

The first unchecked `[ ]` is always "what's next." No ambiguity.

## 5. The build loop (one batch per cycle)
1. **Orient** — read `STATUS.md` (active task) → the ADR's first unchecked batch →
   `git log` + current branch → `CONVENTIONS.md` → `CLAUDE.md` (Golden Rule + modes).
2. **Branch** — a big change/feature/phase gets **its own branch**; minor fixes and small
   changes are **gathered** on the rolling fixes branch (naming in §8). Never work on main.
3. **Implement** — follow the ADR + conventions. Reuse existing patterns over rebuilding.
   Keep the Golden Rule. Update the file brief for every file you add or change (in
   `docs/filemap/` or in-file, per the project's brief mode — §7).
4. **Verify** — run the test suite to green; for UI, build + lint clean. Add a
   deterministic test for any core/logic change (no exceptions).
5. **Record in the SAME batch** — `STATUS.md` (active + next), `progress/YYYY-MM.md`
   (newest on top), `decisions/LOG.md` (every decision made along the way),
   `DATA_MODEL.md` if types changed; tick the batch `[ ]` → `[x]` in the ADR. What counts
   as recordable: §11.
6. **Commit** — conventional message (§8), under the **user's git identity**, with **no
   AI attribution** of any kind.
7. **Propose the merge** — when green, tell the user the branch is ready. **Merge to main
   only after the user explicitly approves.** Never self-merge.
8. **Decide** — more batches + context still roomy → next batch (if approval is pending
   and the next batch depends on the unmerged work, stack on it — §8). Context getting
   large → stop after the commit; everything needed is on disk, so a reset resumes cleanly.

## 6. Definition of "done" (every task)
1. Code written **and** its file brief updated (filemap or in-file, per mode).
2. Tests green (offline/fast; plus integration where relevant).
3. `STATUS.md` + `progress/` updated; `decisions/LOG.md` appended; `DATA_MODEL.md` if
   classes/tables changed.
4. ADR batch ticked; any new non-obvious decision gets an ADR; limitations → `BUGS.md`.
5. Merge **proposed** to the user — merged to main only with their approval.

If any of these is missing, the task is not done.

## 7. File briefs (one mode per project; default: `docs/filemap/`)
Every source file has a brief so anyone (or any agent) knows its role + state at a glance:
```
<Title> — one line.
@context  What this file is and why it exists.
@done     What is implemented here.
@todo     What's left (or "—").
@limits   Hard constraints (e.g. PURE: no network/IO).
@affects  What it depends on / is depended on by.
```
Two modes — the project picks ONE at initialization (the user is asked; the choice is
recorded in `CLAUDE.md`):
- **`docs/filemap/` (default)** — briefs live in `docs/filemap/<area>.md`, one entry per
  source file, mirroring the folder structure. Code files stay clean of meta headers.
- **in-file** — the brief sits at the top of each source file, in the language's comment
  syntax.

Either way: update the brief in the SAME change that alters the file's behaviour.

## 8. Git workflow
- **Identity:** commits use the user's **global git config** (`git config --global
  user.name` / `user.email`). If either is unset, **ask the user** — never invent an
  identity, never commit as the AI.
- **No AI attribution — ever.** No `Co-Authored-By: Claude` trailers, no "Generated with
  Claude Code" footers, no AI mentions in commit messages, PRs, or code. The work is
  recorded under the user's name only.
- **Always branch.** Never commit straight to the main branch.
  - Big change / feature / phase → its own branch: `feature/<slug>`, `phase/<n>-<slug>`.
  - Minor fixes / small changes → **gathered** on a rolling `fixes/YYYY-MM` branch (one
    commit per fix; the branch is proposed for merge as a bundle).
- **Conventional commits:** `feat(scope): …`, `fix: …`, `refactor: …`, `docs: …`. Body
  explains the *why* when it isn't obvious.
- **Merging is the user's call.** When a branch is green, **propose** the merge and wait.
  Merge to main **only after explicit approval** — never self-merge, never assume. If the
  next batch depends on work whose approval is pending, branch from that unmerged branch
  (stacked) and note it in `STATUS.md`.
- **Never push without explicit approval.** Don't skip hooks or bypass signing.

## 9. Test discipline
- Every core/logic change ships a **deterministic, offline** test.
- Keep the fast suite network-free; mark network/integration tests separately so they
  don't gate the loop.
- Parity rule of thumb: a change that's meant to be invisible (refactor, new optional
  path) must leave existing outputs identical — assert it.

## 10. The seam pattern (for external dependencies)
Put every external dependency (DB, cache, queue, third-party API, crawler) **behind an
interface** with a real *local* implementation now and a *documented* production swap
later. Ship value today without standing up heavy infra; swap impls with no caller changes.

## 11. Recording discipline (what gets written down)
Record **all of it** — every change, every fix, every decision, even the little ones.
- **Changes** → `docs/progress/YYYY-MM.md`: one entry per unit of work, listing what
  changed and why, file by file. Small tweaks get small entries — but they get entries.
- **Decisions** → one line in `docs/decisions/LOG.md` (date, what, why). Naming choices,
  library picks, defaults, small tradeoffs — all of it. Non-obvious or architectural
  decisions ALSO get a full ADR under `docs/decisions/`.
- **Bugs/limitations** → `docs/BUGS.md` the moment they're hit.

The ONE exclusion: discussion that produced no change and no decision. Brainstorming that
went nowhere isn't logged. If it changed the code or settled a question — it goes on disk.

## 12. Parallel development (optional)
The default loop is **serial**: one batch per cycle. When the user wants concurrent work
(multiple agents or contributors), enable parallel mode — at initialization or any time
later. `docs/PARALLEL.md` adapts this workflow: worktree lanes (or trunk-based /
integrator methodologies), batches that declare disjoint `Touches:`, a Lanes table in
`STATUS.md`, per-lane progress files, and an `integration` branch. Main stays
approval-gated exactly as in §8. Record `Parallel mode: ON — <methodology>` in `CLAUDE.md`.

## 13. Initialization & self-removal (once, at adoption)
`prompts/kickoff.md` drives this. In short: verify the git identity (§8), copy the kit's
templates into the repo (including the `/resume-ongoing-work` command into
`.claude/commands/`), fill the placeholders, ask the user the init questions (file-brief
mode, serial vs parallel, ignore scope, enforcement hooks), install the hooks (§15),
append the workflow ignores to `.gitignore` (workflow docs, plans, dev/test artifacts are
local-only), **delete the kit source folder**, then read the product plan and **propose
what to start with** — waiting for the user's go-ahead before building.

## 14. Resuming after a context reset
Two pieces, one command:
- **`/resume-ongoing-work`** (`.claude/commands/resume-ongoing-work.md`) — the user types
  it after a clear or in a new session. It re-reads `STATUS.md` + the ADR checklist +
  `decisions/LOG.md` + `git log` + its own snapshot section, then continues the next
  unchecked batch.
- **The announce-clear protocol** — when the user says they're about to clear the chat
  ("I'm going to clear", "update the command for resume", …), BEFORE the clear: bring
  `STATUS.md` current, WIP-commit any uncommitted work on the branch, and rewrite the
  **Ongoing-work snapshot** section inside the command file.

Because §6 keeps disk truthful and the snapshot catches mid-batch state, a clear at any
moment resumes cleanly.

## 15. Enforcement hooks (Claude Code — the anti-drift mechanism)
Files remind passively; over long sessions attention drifts and recording gets skipped.
Three hooks (installed at init, optional, default ON) enforce the workflow at the harness
level, where memory can't decay:
- **UserPromptSubmit → per-turn reminder** — injects a one-line workflow summary into
  EVERY turn: branch per batch, record everything, merge only on approval, no AI
  attribution.
- **Stop → recording guard** — the agent cannot finish a turn while tracked code is
  modified but `docs/STATUS.md` / `docs/progress/` are older than those changes; the stop
  is blocked once with "record first" feedback (§5.5, §6).
- **PreToolUse(Write) → rewrite guard** — overwriting an always-read stable file
  (`CLAUDE.md`, `docs/WORKFLOW.md`, `docs/CONVENTIONS.md`) resets its free re-read cache,
  so the hook asks the user to confirm; a line change should use `Edit` (§16). Silent for
  every other file and for `Edit`.

Config: `.claude/settings.json`; scripts: `.claude/hooks/` (sh + PowerShell variants —
init installs the right one; see `.claude/hooks/README.md`). Gitignored like the rest of
the workflow. Hooks are Claude Code-specific — other agents and humans ride on §5.1
orientation and `CLAUDE.md` alone.

## 16. Token discipline (spend less to do the same work)
This workflow reads and writes a lot. The cost is almost all in **re-reading the same
files**, not in writing — writes are mostly small appends. Cut the reading and you cut the
bill. No tools or plugins needed; these are habits:

1. **Don't churn the hot files.** `CLAUDE.md`, `docs/WORKFLOW.md`, `docs/CONVENTIONS.md`
   are read at nearly every session start. When they're unchanged, re-reading them is
   almost free (the harness caches them); **editing one throws that discount away.** Set
   them up, then leave them alone — change them only when a rule actually changes. Keep
   `CLAUDE.md` stable by design: nothing that changes as work progresses lives there (it
   points to `STATUS.md`). The §15 rewrite guard backs this — it asks before a whole-file
   overwrite of these three; use `Edit` for line changes.
2. **Read the brief, not the file.** Need to know what a source file is for? Read its
   one-line entry in `docs/filemap/` (or its in-file brief), not the whole file. That's
   what the filemap is for.
3. **Edit, don't rewrite.** Updating `STATUS.md` or ticking an ADR box → change the one
   line, don't re-emit the whole document. `progress/` and `decisions/LOG.md` are
   append-only (add to the end) — the cheapest writes there are.
4. **Orient narrow.** At batch start read only three things: `docs/STATUS.md`, the one
   active ADR, and `git log --oneline` — not the whole `docs/` tree.
5. **Write it short the first time.** Keep every doc lean — short lines, no filler, one
   line per entry. Small files stay cheap to read forever (this replaces any "compress it
   later" step — just write briefly up front).
6. **Don't re-read what's already in context.** A file read this session and unchanged
   doesn't need re-opening.
7. **One recording pass per batch.** Update all the docs once, at the end of the batch
   (§5.5) — not continuously through it.

Optional, still no plugin: for a lookup that spans many files, a plain scouting subagent
(built into Claude Code) can read them and hand back just the answer, so the main context
skips the file dumps.
