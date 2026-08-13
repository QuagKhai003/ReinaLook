# Research brief — industry film-emulation practice (agent 3/3, 2026-08-12)

Distilled for ADR-0008 (full sources in agent report).

## The census: how the winners work
- **Every successful product is a physical two-stage negative→print model in the DENSITY
  domain** (Dehancer, Filmbox, Koji, FilmLight Truelight, FilmConvert). Purely statistical
  fitting from images: "essentially nobody successful in this market."
- Dehancer: darkroom-printed profiles, 3 exposure states (−2/0/+2 EV), subtractive Color
  Head, Analog Range Limiter (ship measured Dmin/Dmax, never normalize to 0-1).
- Filmbox: 20-stop half-stop chart sweeps; measured 250D→2383 contact print; ACEScct
  internally; other stocks = offsets from the measured backbone.
- Kodak patent US7327382 = the canonical emulation skeleton: 1D linearize → 3×3 in
  exposure → per-channel exposure→density curves → 3×3 in density — calibrated by
  fitting stage parameters to chart pairs. "Directly the shape of the model your engine
  should fit."

## The 5 architecture choices that matter
1. Two coupled stages in density domain (lin → log10 → neg H&D → density → T=10^-D →
   printer lights → print H&D → projector white), never one display-referred map.
2. Per-channel S-curves in log with a rendering-primaries 3×3 in front — the hue
   crossovers ARE the look (Yedlin). No ratio-preserving tone mapping.
3. Exposure-indexed behavior (push ±2 stops must change color/contrast like film) —
   free if the model is parametric-physical.
4. Honor Dmin/Dmax; the print IS the gamut mapper; highlights converge to paper-white.
5. Fit the ~30–50 physical parameters to references (never a free lattice); bake .cube
   last for one declared viewing condition; grain/halation stay out of the cube.

## Implementation references
spektrafilm/agx-emulsion (open spectral pipeline + digitized stock JSONs, exports LUTs) ·
vkdt filmsim (compact data + DIR params) · thatcherfreeman/utility-dctls (minimal correct
H&D DCTLs) · Filmbox Technical FAQ · Truelight FL-TL-TN-0416 · Kodak US7327382.
