"""lutgen — LUT generator package.

@context  Top-level package. Generates a 33-point 3D .cube LUT (DWG/DI -> Rec.709 g2.4)
          with a creative look blended over a protected conversion base.
@done     Package scaffold only.
@todo     Implement L1 engine (ADR-0001), then L2/L3/L4.
@limits   Golden Rule: the conversion base is sacred; s=0 == pure conversion, bit-for-bit.
@affects  Subpackages: engine (L1), fitter (L2), orchestration (L3), app (L4).
"""

__version__ = "0.0.0"
