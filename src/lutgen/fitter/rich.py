"""rich — the quality Look Fitter (Monge-Kantorovich linear optimal transport).

@context  Captures the references' PALETTE (mean + full color covariance / channel correlations),
          not just per-channel stats like Mid. Closed-form MKL (Pitie-Kokaram): the affine map
          that matches both mean and covariance. Drop-in for MidFitter (same interface, ADR-0010).
@done     RichFitter.fit (MKL via PSD matrix square roots) + _RichLookTransform (+ tone_strength).
@todo     Perceptual-space (Oklab) MKL; nonlinear N-d PDF transfer (Plan §3 rung 2).
@limits   PURE numeric (fit reads the fixed base via load_base unless given source_samples).
          Rec.709 g2.4 space. Final clamp left to regularize. eps regularizes Sigma_source.
@affects  Implements fitter/interface.LookFitter. Consumes ConsensusLook (mean+covariance).
          See ADR-0010 + Plan/30_LOOK_FITTER.md §3.
"""

from __future__ import annotations

import numpy as np

from lutgen.engine.base import load_base
from lutgen.engine.perceptual import from_oklab, to_oklab
from lutgen.orchestration.consensus import ConsensusLook
from lutgen.orchestration.stats import LUMA_WEIGHTS

from .interface import LookTransform

_EPS = 1e-5  # regularize covariances for invertible / well-conditioned roots


def _sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _psd_pow(m: np.ndarray, power: float) -> np.ndarray:
    """Raise a symmetric PSD matrix to a real power via eigendecomposition (clamped eigenvalues)."""
    w, v = np.linalg.eigh(_sym(m))
    w = np.clip(w, _EPS, None)
    return (v * (w ** power)) @ v.T


def _mkl_matrix(cov_s: np.ndarray, cov_t: np.ndarray) -> np.ndarray:
    """Monge-Kantorovich linear map A with A.Sigma_s.A^T = Sigma_t (symmetric solution)."""
    s_half = _psd_pow(cov_s, 0.5)
    s_ihalf = _psd_pow(cov_s, -0.5)
    middle = _psd_pow(s_half @ cov_t @ s_half, 0.5)
    return s_ihalf @ middle @ s_ihalf


class _RichLookTransform:
    """Callable neutral_rgb -> looked_rgb: affine MKL transport in RGB or Oklab, + optional
    lightness preservation. In Oklab, transport runs on (L,a,b) and tone preserves L (channel 0);
    in RGB it runs on RGB and tone preserves luma."""

    def __init__(self, mu_s, matrix, mu_t, tone_strength, space="rgb"):
        self._mu_s = np.asarray(mu_s, dtype=np.float64)
        self._A = np.asarray(matrix, dtype=np.float64)
        self._mu_t = np.asarray(mu_t, dtype=np.float64)
        self._tone = float(tone_strength)
        self._space = space

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float64)
        x = to_oklab(rgb) if self._space == "oklab" else rgb
        looked = (x - self._mu_s) @ self._A.T + self._mu_t
        if self._tone < 1.0:
            if self._space == "oklab":
                looked[..., 0] = x[..., 0] + self._tone * (looked[..., 0] - x[..., 0])  # preserve L
            else:
                lin = x @ LUMA_WEIGHTS
                lout = looked @ LUMA_WEIGHTS
                looked = looked + ((lin + self._tone * (lout - lin)) - lout)[..., None]
        return from_oklab(looked) if self._space == "oklab" else looked


class RichFitter:
    """Optimal-transport Look Fitter (ADR-0010/0011). `fit(consensus) -> LookTransform`.

    ``space`` = "oklab" (default, perceptual) or "rgb". ``tone_strength`` (0..1, default 1.0)
    preserves input lightness while keeping the transported palette (lower = keep brightness)."""

    def __init__(self, tone_strength: float = 1.0, space: str = "oklab"):
        self._tone = float(np.clip(tone_strength, 0.0, 1.0))
        if space not in ("oklab", "rgb"):
            raise ValueError(f"space must be 'oklab' or 'rgb', got {space!r}")
        self._space = space

    def fit(self, consensus: ConsensusLook, source_samples=None) -> LookTransform:
        src = load_base() if source_samples is None else np.asarray(source_samples, dtype=np.float64)
        src = src.reshape(-1, 3)
        if self._space == "oklab":
            src = to_oklab(src)
            mu_t, cov_t = consensus.mean_oklab, consensus.cov_oklab
        else:
            mu_t, cov_t = consensus.mean, consensus.covariance
        mu_s = src.mean(axis=0)
        cov_s = np.cov(src, rowvar=False) + _EPS * np.eye(3)
        matrix = _mkl_matrix(cov_s, _sym(cov_t) + _EPS * np.eye(3))
        return _RichLookTransform(mu_s, matrix, mu_t, self._tone, space=self._space)
