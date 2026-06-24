"""orchestration — L3 (references -> consensus -> look -> cube).

@context  Ingests N reference images, extracts per-image stats, fuses them into one robust
          fitter-agnostic ConsensusLook, then drives the active fitter + engine.
@done     Scaffold only.
@todo     ingest.py, stats.py, consensus.py, pipeline.py (M2).
@limits   Produces ConsensusLook only; no color-space conversion here.
@affects  Feeds fitter; orchestrates engine. See codemap/INDEX.md.
"""
