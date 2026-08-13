# ONBOARDING

Welcome. This routes you to the right files by role. Read `CONVENTIONS.md` either way — it's
the contract everyone follows.

## Everyone, first 10 minutes
1. `CLAUDE.md` (repo root) — what this is + the **Golden Rule** (never violate it) + the
   project's modes (brief mode, parallel mode).
2. `docs/STATUS.md` — what's happening right now + what's next (+ lanes, if parallel).
3. `docs/ROADMAP.md` — the phases and where we are.
4. `docs/filemap/` — the per-file briefs: what every file is for (default brief mode).

## By role
- **Core / domain dev** → `<core dir>/` + its ADRs in `docs/decisions/`. Every change needs a
  deterministic test. Respect the Golden Rule.
- **Backend / API dev** → `<api dir>/` + `docs/DATA_MODEL.md` for contracts.
- **Frontend dev** → `<ui dir>/`; read any framework gotchas noted in `CLAUDE.md` first.
- **Data / integrations** → `<data dir>/`; external deps go behind a seam (see `docs/WORKFLOW.md` §10).

## Your first task
Pick the first unchecked batch in the active ADR (`docs/decisions/`), branch (features get
their own branch; small fixes gather on `fixes/YYYY-MM`), and follow the build loop in
`docs/WORKFLOW.md` §5. Finish per the "done" definition (§6) — record every change and
decision, and remember: **main moves only with the user's approval**.
