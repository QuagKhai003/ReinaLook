# FILEMAP — per-file briefs (the default brief mode)

> Every source file has a brief describing its role and state. In the **default mode** the
> briefs live here — one entry per source file — so the code files themselves stay clean.
> (If the project chose **in-file** mode at initialization, this folder is unused and the
> same fields sit at the top of each source file instead. The mode is recorded in
> `CLAUDE.md`.)

Mirror the source tree: one `docs/filemap/<area>.md` per top-level source folder
(e.g. `src-core.md`, `src-api.md`, `ui.md`). Keep entries in file-path order.

## Entry format (same fields as an in-file header brief)

```markdown
## `src/<area>/<file>`
<Title — one line.>
- **@context** — what this file is and why it exists.
- **@done** — what is implemented here.
- **@todo** — what's left (or "—").
- **@limits** — hard constraints (e.g. PURE: no network/IO).
- **@affects** — what it depends on / is depended on by.
```

## Rules
- **New file** → add its entry in the SAME change.
- **Behaviour change** → update the entry in the SAME change.
- **Moved/deleted file** → fix or remove the entry immediately.
- An out-of-date filemap is worse than none — it lies to the next reader.
- **Growth rule (CONVENTIONS §1):** an area file that gets long splits deeper, mirroring
  the subfolders — either finer files (`src-core-parsers.md`) or a subfolder with its own
  `README.md` index. This file stays the top-level index either way.
