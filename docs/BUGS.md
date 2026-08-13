# BUGS & LIMITATIONS

> Log it the moment you hit it. Don't wait, don't rely on memory. Each entry: an id, the
> symptom, the cause if known, status, and where it's handled.
>
> **Growth rule (CONVENTIONS §1):** when this file gets long, split it — replace with a
> `docs/bugs/` folder: `README.md` as the index (one line per entry: id — title — status),
> one file per entry (`L-001-<slug>.md`); resolved entries may batch into
> `resolved-YYYY.md`. Ids stay stable across the split.

Format: `L-NNN` (limitation) / `B-NNN` (bug) / `S-NNN` (security/launch risk).

---

## L-001 — pre-existing ruff findings in v1 files
- **Symptom:** `ruff check src` reports 9 findings in files untouched by v2 work: unused
  imports (`app/main_window.py`, `engine/cube_io.py`), blind `except Exception`
  (`main_window.py` ×2), unsorted imports (`cli.py`, `orchestration/pipeline.py`),
  `typing.Callable` deprecation (`fitter/interface.py`), `dict()` literal (`main_window.py`).
- **Cause:** v1 code predates lint enforcement.
- **Status:** RESOLVED 2026-08-12 — remaining findings (cube_io, interface, pipeline, cli) cleared by a ruff --fix sweep during ADR-0004.
- **Where:** found 2026-08-11 during batch 1.3; v2 files are kept clean per batch.
- **Notes:** mostly auto-fixable (`ruff check --fix` + 2 manual). Gather on a `fixes/2026-08`
  branch — don't mix into feature batches.

## L-002 — intermittent interpreter crash in full-suite runs (OpenBLAS × Qt threads)
- **Symptom:** rarely, a full `pytest` run dies mid-run with a bare `frozen runpy` traceback
  (native crash, no Python assertion). Rerun passes clean.
- **Cause (suspected):** OpenBLAS threading × QThread workers — same class of crash v1 noted
  in `orchestration/stats.py` (kept stats serial for this reason).
- **Status:** open, mitigated (recurred 2026-08-12 in the 4.1 full-suite run; rerun with
  `OPENBLAS_NUM_THREADS=1` passed clean — use that for full-suite runs)
- **Where:** test env; GUI worker threads + BLAS-heavy fits in one process.
- **Notes:** mitigation confirmed: `OPENBLAS_NUM_THREADS=1 python -m pytest`.
