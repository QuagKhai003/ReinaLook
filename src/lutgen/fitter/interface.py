"""interface — the one swappable Look Fitter contract (L2).

@context  The stable seam between L3 (ConsensusLook) and L1 (engine). Mid (MVP) and Rich (phase
          2) are two implementations of the SAME interface, so swapping them changes only L2.
@done     LookFitter Protocol + LookTransform type alias.
@todo     -
@limits   PURE: types only. A LookTransform maps neutral RGB -> looked RGB, vectorized (N,3).
@affects  Implemented by fitter/mid.py (and later rich.py). Output consumed by engine/strength.py.
          See Plan/30_LOOK_FITTER.md §4 + ADR-0005.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from lutgen.orchestration.consensus import ConsensusLook

# A LookTransform maps neutral-converted RGB to looked RGB, vectorized over (N, 3).
LookTransform = Callable[[np.ndarray], np.ndarray]


@runtime_checkable
class LookFitter(Protocol):
    """Turns a fitter-agnostic ConsensusLook into an engine-consumable LookTransform."""

    def fit(self, consensus: ConsensusLook) -> LookTransform: ...
