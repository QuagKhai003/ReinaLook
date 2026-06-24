"""fitter — L2 look fitter (one swappable interface).

@context  Converts a ConsensusLook into a LookTransform the engine can blend. Mid (MVP)
          and Rich (phase 2) are two impls of the same LookFitter interface.
@done     Scaffold only.
@todo     Mid fitter (M3); Rich fitter (M6).
@limits   Never touches the base; outputs a LookTransform only.
@affects  Consumes ConsensusLook (orchestration); output blended by engine/strength.py.
"""
