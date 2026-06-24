"""engine — L1 color engine (protected base + strength + cube I/O).

@context  Pure, stateless color math. Owns the deterministic DWG/DI -> Rec.709 g2.4
          conversion (the protected base) and the strength blend onto it.
@done     Scaffold only.
@todo     grid.py, spaces.py, convert.py, cube_io.py (ADR-0001); strength.py, regularize.py (M1).
@limits   PURE: no IO except cube_io. Golden Rule: base never altered; s=0 == base.
@affects  Depended on by fitter, orchestration, app, cli. See codemap/INDEX.md.
"""
