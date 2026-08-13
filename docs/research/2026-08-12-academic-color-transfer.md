# Research brief — academic color transfer & film modeling (agent 1/3, 2026-08-12)

Key findings (full sources in the agent report; distilled for ADR-0008 planning):

1. **Mean-matching provably mutes.** Least-squares parametric maps shrink variance
   (regression to the mean). Perceived colorfulness ≈ σ of chroma (Hasler–Süsstrunk:
   M = σ + 0.3·μ — spread weighs ~3× the mean). Punch lives in the TAILS (p90/p99 chroma,
   p5/p95 L). → Fit QUANTILE residuals (5/25/50/75/90/99), up-weight tails ×3; add a
   chroma-SPREAD residual (std(a), std(b), Hasler term). 1-D quantile matching = exact
   sliced-OT; slots directly into scipy least_squares.
2. **Our crosstalk has the wrong sign & domain.** Film inter-image effects ≈ linear in
   LOG-DENSITY with NEGATIVE off-diagonals (DIR couplers) → saturation-INCREASING.
   Unconstrained L2 fits converge to positive off-diagonals = channel averaging =
   desaturation. → Apply the 3×3 in log domain, constrain off-diagonals ≤ 0 (rows sum 1).
3. **Film-curve prior exists as data:** EMoR/DoRF (Columbia CAVE) — 201 measured response
   curves incl. film H&D curves; PCA shows a 3–5-dim subspace dominated by S-curves with
   mid-slope > 1. → Fit tone curves as EMoR PCA coefficients (or minimally: constrain
   mid-tone slope ≥ 1.05–1.2 + RMS-contrast residual std(L)).
4. **Oklab is blind to appearance effects.** Hunt effect (colorfulness grows with
   luminance): a darkened look at "matched" chroma looks duller. H–K effect similar.
   → chroma target scaled by (Ȳ_ref/Ȳ_out)^α, α≈0.1–0.2, or Hellwig-2022 colorfulness
   (in colour-science); or asymmetric chroma loss (under-sat penalized ~2×).
5. **JPEG debiasing:** web stills under-measure chroma tails (4:2:0 subsampling + clip).
   → measure reference chroma via per-tile p95 pooling (32×32 tiles); treat gamut-clipped
   upper tail as censored: inequality residual max(0, C_ref_p99 − C_out_p99).
6. **Loss architecture lesson (learned-LUT lit):** content-fidelity and style must be
   SEPARATE terms; style term must be distributional (quantiles/SW), never a mean.
   Monotonicity penalty + light TV smoothness (Zeng 3D-LUT: smoothness weight deliberately
   tiny — over-smoothing causes flatness).
