# ReinaLook v2 — Film Emulation Engine Upgrade Specification

**Status:** Design locked from discussion. Ready for implementation planning.
**Scope:** Color-only film emulation. No grain, halation, bloom, texture, or film damage (spatial effects are out of scope for a LUT by physical definition).
**Target output:** DaVinci Resolve 3D `.cube` LUT, riding on the existing sacred DWG/DI → Rec.709 CST architecture.

---

## 1. The Core Problem With ReinaLook v1

ReinaLook v1 has two disconnected halves:

1. **Reference matching** (Optimal Transport / Pitié PDF / MKL in Oklab) — a **free-form statistical warp**. It matches the reference image's color distribution, but the transform has no coherent structure. It memorizes *this image's statistics* (content included), not a color science.
2. **Film Stock panel** (S-curve, toe, shoulder, split-tone, saturation) — real film-shaped math, but **manual dials**. Nothing learns these values from a reference.

**Neither half does the actual goal: learn a film-shaped color science *from* the reference.**

### The one hard truth

With only screenshots/exported frames as input, there is **no extra information available** beyond what v1 already reads. The pixels are the pixels. The only thing that can change — and it is exactly the right thing — is the **form (shape) of the transform**:

| | v1 (matching) | v2 (emulation) |
|---|---|---|
| Transform shape | Free-form warp, ~108,000 effective degrees of freedom (33³ grid) | Constrained film-shaped model, ~40–80 parameters |
| What it learns | This reference image's statistics (content + grade mixed) | The film-like *recipe* that best explains the reference |
| Behavior on unseen colors | Extrapolation artifacts, hue breaks | Sane by construction (S-curves + matrix are plausible everywhere) |
| Output | A LUT | A LUT **plus** an inspectable, editable, savable recipe |

**Emulation vs. matching = parametric model *fitting* vs. distribution *matching*.** That is the entire upgrade in one sentence.

---

## 2. What Film Actually Does (the model's justification)

These four behaviors are the complete category list of what a pointwise color transform can express. Every film "look" is a combination of them:

1. **Characteristic curve.** Nonlinear S-shaped response per dye layer: shadows compress gently (**toe**), midtones respond steeply, highlights roll off gradually (**shoulder**), never clipping hard. Critically, the R/G/B curves are **slightly different from each other** — this is why film shadows drift toward a color instead of staying neutral.
2. **Channel crosstalk.** Dye layers contaminate each other: red exposure slightly affects green dye, etc. Mathematically a small mixing matrix. This single ingredient produces the characteristic film hue twists (a film orange ≠ a digital orange). Basic grading tools do not do this.
3. **Brightness-dependent saturation.** Film desaturates in deep shadows and bright highlights, holds rich saturation in midtones. Saturation is a *curve over luminance*, not a global number.
4. **Hue-dependent personality.** Certain hue regions render with their own character (skin, greens, sky). Includes hue shifts that differ by brightness (shadows→cyan, highlights→warm — split-toning is the crude manual version).

---

## 3. The Parametric Film Model

Fixed processing pipeline, applied in **DaVinci Wide Gamut / DaVinci Intermediate** (keep v1's sacred CST architecture — footage → CST into DWG/DI → LUT → CST to Rec.709; the `.cube` is technically log-to-log with both v1 placements preserved: Replace CSTout / Between CSTs).

### Pipeline order (fixed)

```
input (DWG/DI)
  → [Block A] 3×3 crosstalk matrix
  → [Block B] per-channel S-curves (R, G, B independently)
  → [Block C] saturation-vs-luminance curve      (in Oklab)
  → [Block D] hue-zone hue/sat trims             (in Oklab)
  → [Block E, v2.1] hue-shift × luma-band grid   (in Oklab)
  → output (DWG/DI)
```

### Parameter budget

**v2.0 baseline (~40 parameters):**

| Block | Content | Params |
|---|---|---|
| A — Crosstalk | 3×3 matrix, rows constrained: diagonal-dominant, rows sum to 1 (energy preserving) | 6 free |
| B — Tone | 3 channels × {toe strength, shoulder strength, mid slope, pivot} | 12 |
| C — Sat vs luma | 3 control points (shadow / mid / highlight sat multipliers), smooth monotone-safe spline | 3 |
| D — Hue zones | 6 zones (R, Y, G, C, B, M) × {hue shift, sat trim} | 12 |
| Global | exposure/black offset trims | ~4–7 |
| **Total** | | **~40** |

**v2.1 upgrades (→ ~70–80 parameters), in priority order:**

1. **Smooth periodic hue curve** replacing the 6 chunky zones: low-order Fourier series over the hue wheel (~8–10 coefficients per attribute for hue-shift and sat). Single most worthwhile upgrade — film hue behavior is smoother and more detailed than 6 zones (skin alone may deserve sub-region resolution). ≈ +15–20 params.
2. **Hue-shift × luma-band grid** (Block E): hue shift as a function of (hue zone × luma band), the proper generalization of split-toning. ≈ +15 params.

**Hard ceiling: ~80. Never fix quality problems with more parameters** — see §5. If more expressiveness seems needed, the honest fix is more reference frames.

### Curve formula guidance (Block B)

Each channel curve: smooth monotonic S-curve parameterized by toe/shoulder/slope/pivot (e.g., a filmic tone-curve family or monotone cubic spline through anchored control points). Requirements:

- Strictly monotonic (no reversals ever).
- C¹ smooth (no kinks → no banding).
- Identity at parameters = 0 (so strength 0 = pure conversion, preserving v1's core promise).
- Bounded parameter ranges enforced by the optimizer (e.g., toe/shoulder ∈ [0, max], slope ∈ [0.5, 2.0]) so no fitted result can be degenerate.

---

## 4. Learn Mode / Apply Mode Architecture

The pairwise "source vs reference" workflow is replaced by a two-phase product shape:

### Learn Mode — "learn the reference's color science first"

**Input:** a **pool of 5–15 reference frames only** (different scenes from the same film / graded video). No source frames required at learn time.
**Output:** a saved **Look Profile** — the fitted parameter set as a portable JSON recipe file, plus fit-quality metadata.

**Why pooling is the key idea (the strongest lever available):**
Scene content is *random* across different shots; the grade is *constant*. Whatever color behavior all frames share despite showing different things is, almost by definition, the grade. Teal shadows appearing in a kitchen scene AND a street scene AND a portrait is not scenery — it is the look. Pooled statistics average content out; the shared transform remains.

**The single-image wall (must be surfaced in UI):** from one frame, "the scene was warm" and "the grade is warm" are mathematically indistinguishable — missing information, not a solvable puzzle. Single-frame learning is scene-contaminated by nature.

**Neutral prior (fallback for thin data):** assume the ungraded world was statistically normal (average color near gray, moderate saturation, blacks near black). The fit pulls toward this prior wherever pooled evidence is thin. Weak alone (misreads sunsets/neon/forests as "the grade") — acceptable only as regularization, never as the core method.

**Frame weighting:** robust statistics so one unusually colorful reference cannot hijack the fit (e.g., median/trimmed pooling per statistic bin; optionally down-weight frames whose stats are outliers vs. the pool consensus).

**Expected quality by frame count (show in UI):**

| Frames | Result |
|---|---|
| 1 | Scene-contaminated. Warn the user explicitly. |
| 5+ | Genuinely good. |
| 10–15 varied | About as close to the film's actual color science as physics allows without the original negative. |

### Apply Mode — "then apply to my image"

**Input:** any saved Look Profile + optional user footage still.
**Output:** baked `.cube` over the CST conversion.
**Optional source-adaptive trim:** small white-balance anchor measured from the user's footage so the recipe sits correctly on *their* image (trim applies before the film model; keep it small and clearly labeled).

**Product consequence:** users build a **library of learned looks** and reuse them forever — a much better shape than pairwise matching.

### Legacy pairwise mode (keep)

v1's Before/After Pairs mode (exact grade from paired frames) remains the strongest option **when the user has true pairs** — keep it as-is. The unpaired Neutral+Graded mode can be re-implemented on the new fitter (fit the film model so that pooled neutral stats → pooled graded stats).

---

## 5. Fitting Strategy

**Optimizer:** deterministic `scipy` least-squares / bounded minimization. No AI, no black box — preserves v1's "explainable math" positioning.

**Loss:** distance between (reference pool statistics) and (model applied to prior/source statistics), measured in **Oklab** on robust statistics:

- Per-channel tone distributions (cumulative distributions → smooth curve targets).
- Mean chroma per luminance band.
- Mean hue/sat per hue zone (per hue-Fourier bin in v2.1).

~40–80 parameters against thousands of pixel statistics = heavily over-determined = stable and smooth. (Contrast: raw OT effectively fits ~108,000 DOF — v1's known behavior.)

**Staged fitting (required for stability):**

1. **Tone curves first** (Block B) — determined by the most abundant statistic.
2. **Crosstalk matrix** (Block A).
3. **Saturation & hue detail** (Blocks C/D/E).

Each stage freezes or softly anchors previous stages. Fitting all ~70 parameters simultaneously is not stable; staged fitting is.

**Per-region regularization (required):** where reference data is thin (rare hues — pure violet in a typical movie frame ≈ nothing), relax the fit toward neutral instead of extrapolating a guess. Effects:

- Fitted-to-noise parameters (which show up as hue wobbles on real footage) are suppressed.
- Model size becomes nearly self-correcting — unused capacity gracefully does nothing.
- **A 40–80 parameter model on thin data beats a 500-parameter model on the same data, every time.** The data is the bottleneck, not the model.

**Initialization:** the existing Film Stock panel defaults. After fitting, the same panel becomes the **edit layer** — users can hand-tweak fitted values.

---

## 6. LUT Baking & Validation

1. Generate identity grid (33³ default, 65³ option) in DWG/DI.
2. Push every grid point through the fitted model (+ optional trims + strength blend).
3. Strength dial preserved: 0 = bit-exact conversion, 1 = full look (v1 core promise intact).
4. Write `.cube`, both placements supported.

**Mandatory stress validation before every export (automated):**

- Synthetic test chart: color checker patches + smooth gradient ramps + hue wheel sweep.
- Checks: monotonic tone (no reversals), no hue breaks across zone/curve boundaries, ΔE smoothness between adjacent LUT nodes above a threshold, black/white point sanity.
- Fail → show which block violated, do not silently export.

---

## 7. Honest Limits (surface these in UI/docs — do not oversell)

- **Not a certified stock profile.** Output = "the film-shaped transform that best explains this reference." It cannot certify "this is exactly Kodak 2383," because the reference's ungraded original is never observed. Dehancer/FilmConvert accuracy comes from shooting charts on real film — data this tool does not have. No screenshot-only tool can do better.
- **Color-only.** Grain/halation/texture are spatial; a LUT physically cannot carry them. If wanted, they are separate passes outside this scope.
- **Content leakage** shrinks with pool size but never reaches zero. The parametric shape itself already filters much of it (the model *cannot* memorize content — it can only express relationship patterns like "shadows lean teal"), pooling removes most of the rest.

---

## 8. What to Reuse From v1 (large)

| Keep as-is | Replace |
|---|---|
| CST architecture (DWG/DI → Rec.709, both placements) | OT / MKL / PDF engines as the *primary* path (keep available as "Match (legacy)" mode) |
| Image pool ingestion (JPG/PNG/TIFF, add/remove) | Pairwise-only mental model → Learn/Apply modes |
| `.cube` writer, strength blend | — |
| Film Stock panel (becomes init + edit layer for fitted profiles) | — |
| Preview still workflow, CLI skeleton, packaging/installer | — |

---

## 9. UI / UX & Performance Requirements

### Performance — "smooth, never laggy"

- **Never block the UI thread.** All fitting, preview rendering, and LUT baking run on worker threads/processes (Qt: `QThread`/`QThreadPool` or `multiprocessing` for the scipy fit). The window must stay responsive — movable, cancellable — during every computation.
- **Progressive preview:** render preview at reduced resolution first (e.g., 960px proxy), refine to full quality after. Target: proxy preview updates in **< 300 ms** after a dial change; full-quality refine can lag behind.
- **Debounce dial input:** while a slider is being dragged, recompute on a short debounce (~100–150 ms after last movement), not on every tick.
- **Cache aggressively:** decoded reference images, pooled statistics (recompute only when the pool changes, not per fit), per-stage fit results, and the identity grid. A dial tweak in Apply mode must never re-run Learn-mode statistics.
- **Vectorize everything:** all image math in NumPy batch operations; no per-pixel Python loops anywhere. Fit target: full Learn-mode fit on a 10-frame pool completes in seconds, not minutes; show staged progress ("Fitting tone… Fitting crosstalk… Fitting hue detail…") with a real progress indicator and a working Cancel.
- **Startup:** lazy-load heavy modules (scipy fit code) so the window appears fast.

### Layout — "no overflow, no hidden controls"

- **No clipped or hidden controls at any window size.** Every panel lives in a proper layout manager with sane minimum sizes; the main window has a workable minimum size below which it cannot shrink.
- **Scroll where needed:** parameter panels (Film Stock / fitted recipe / Adjustments) go inside vertical scroll areas so long parameter lists never overflow or get cut off — they scroll.
- **Resizable split:** controls pane vs. preview pane in a draggable splitter; preview image scales to fit its pane (letterboxed, never cropped or overflowing).
- **Reference pool as a scrollable thumbnail grid** with per-item remove; grid reflows on resize — no horizontal overflow.
- **Long text truncates with ellipsis + tooltip** (file paths, profile names) — never pushes the layout apart.
- **Mode clarity:** Learn / Apply / Match(legacy) as top-level tabs or a clear mode switch; each mode shows only its own controls (no dead/irrelevant dials visible).
- **Fitted recipe display:** after Learn, show the recipe as readable grouped values ("Toe: +0.31 · R→G crosstalk: 0.04 · Shadow sat: −18%") — inspectable and editable, in a scrollable group. This is a headline feature OT could never offer; give it screen space.
- **Inline guidance, not modal nagging:** frame-count quality hint ("1 frame: look will absorb scene colors — add 4+ more for a clean profile") as a persistent inline label near the pool, colored by state.
- **DPI/scaling:** test at 100/125/150% Windows scaling; no overlapping widgets at any of them.

---

## 10. Phased Build Plan

**Phase 1 — Fitter core (no UI changes)**
- Implement Blocks A–D (~40 params), staged fitting, per-region regularization, neutral prior.
- CLI: `reinalook learn --refs *.png --out look.json` and `reinalook apply --profile look.json --out look.cube`.
- Validation harness (§6) as automated tests.
- **Acceptance test:** on the same reference pool, compare fitted-model output vs. v1's PDF engine — the fitted result must be smoother on the stress chart and visibly plausible on unseen footage.

**Phase 2 — Learn/Apply UI**
- Mode restructure, profile save/load library, recipe display + edit layer wired to Film Stock panel, frame-count guidance.
- Threading/caching/debounce infrastructure (§9) — this phase is where "not laggy" is engineered, not patched later.

**Phase 3 — v2.1 model upgrades**
- Fourier hue curve (replaces 6 zones), hue×luma grid (Block E). Re-run acceptance tests; verify regularization keeps thin-data regions neutral.

**Phase 4 — Polish**
- Source-adaptive trim in Apply mode, preset sharing format, layout QA pass at all DPI scales and window sizes.

---

## 11. Design Decisions Locked (with reasons)

1. **Parametric fitting replaces OT as the primary engine** — the only path from "matching" to "emulation" given screenshot-only input.
2. **~40 params v2.0 → ~70–80 v2.1, hard ceiling ~80** — data, not model size, is the bottleneck; beyond this, more frames beat more parameters.
3. **Learn/Apply separation with multi-frame pooling** — pooling is the strongest de-contamination lever that exists for this problem; also the better product shape (look library).
4. **Staged fitting + per-region regularization are non-negotiable** — they are what make 70 parameters stable and thin-data regions safe.
5. **Deterministic scipy, no AI** — preserves explainability positioning and offline promise.
6. **Strength-0 = exact conversion stays sacred** — v1's core promise carries over untouched.
7. **All heavy work off the UI thread with proxy-first preview** — smoothness is an architecture requirement from Phase 2, not a polish item.
