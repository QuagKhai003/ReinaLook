# STATUS — what's happening right now

> Single source of truth for the CURRENT moment. Update at the start and end of every
> session. History goes in `docs/progress/`, not here.
> **This file never grows** — replace, don't append. If it's getting long, you're keeping
> history in it: move that to `docs/progress/` and trim.

**Last updated:** 2026-08-13 (ADR-0008 COMPLETE — b8.5 merged + pushed, main @ c97627c.
RELEASE TRACK: clean venv READY, exe rebuild IN FLIGHT (background) (v1 was 508 MB of global-Python bloat — torch/cv2 etc; see LOG); README v2 + 2.0.0 next. Backlog explicitly
deferred by user except an explanation of outlier down-weighting)

## Phase
Phases 1–3 complete + pushed (main @ wash fix 73b31ff). App: Learn / Apply / Match modes,
threaded staged fit, profile library, recipe editor, multi-still preview, §6-gated export,
auto lighting grouping, vividness contract, opt-in film brightness, exposure-align default.
v2 design: `docs/REINALOOK_V2_FILM_EMULATION_SPEC.md`.

## Active task
**ADR-0008 — the film system core (research-driven rebuild of the tonal engine).** User
approved ("go") after 3-agent deep research (`docs/research/2026-08-12-*`). Root cause of
every look complaint: free statistical fit ≠ film's physical structure. New Block F =
negative (per-channel relative γ, toe) → DIR coupling (density domain, saturation-
INCREASING) → print (slope, shoulder, black/white convergence), identity at neutral.

**b8.1+b8.2 MERGED** (main): Block F core + character preset + FilmModel v3 (G→A→F→B→Oklab).
**b8.3 DONE** on `phase/8.3-fit-losses` (not merged): poolstats sat_quantiles (tail floor
0.15) + sat_tile_p95 (max-of-tile-p95 debias); spread residual (level-free, asymmetric)
in POLISH only; tail-weight + per-band Hunt implemented, default OFF pending b8.4 layout;
soft-l1 REJECTED (hue corr 0.92→0.38). All placement decisions ablation-measured — see
progress + LOG. +8 tests; suite 372 passed / 1 skipped; ruff clean.
**Incident (b8.2, 2026-08-12)**: D: disk "fatal device hardware error" during ruff —
truncated a local test file; recovered from session transcript. Disk health check advised.

## Awaiting approval
- **USER EYEBALL on the b8.5 renders** (3 sheets sent: do-revenge before/after — validator
  GREEN; shirley before/after — 2 marginal validator misses, thin 3-frame pool; push test).
- b8.5 merge after verdict.

## Next action (whoever picks this up)
- If user accepts do-revenge: shirley needs either more bright frames from the user or a
  thin-pool tame (the 2 misses are marginal: ΔE 0.205/0.171, white 0.165/0.16).
- After acceptance: ADR-0008 closes; release track (exe build, README v2, version 2.0.0).
- Suite runs `pytest -n auto` (~40 s on 8 cores; BLAS pinned to 1 thread per worker).
- ADR-0007 7.3 formally closes on the user's accepted eyeball (still open).

## Watch / before launch
- Golden Rule: keep `fitter/`+`engine/` pure — no I/O, no network, no AI. Strength 0 = base bit-for-bit.
- Enforcement hooks are installed but inert until a **session restart** (Claude Code snapshots hooks at startup).
- Fit stability: staged fitting + per-region regularization are non-negotiable (spec §5).
