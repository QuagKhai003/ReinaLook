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

from lutgen.engine.base import DEFAULT_SIZE, load_base
from lutgen.engine.perceptual import from_oklab, to_oklab
from lutgen.orchestration.consensus import ConsensusLook
from lutgen.orchestration.stats import LUMA_WEIGHTS

from ._gradecube import CubeLookTransform, learn_grade_cube
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


def _cdf_match_1d(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map values ``s`` onto the 1D distribution of ``t`` (rank → target quantile)."""
    ranks = np.empty(s.shape[0])
    ranks[np.argsort(s)] = np.linspace(0.0, 1.0, s.shape[0])
    return np.interp(ranks, np.linspace(0.0, 1.0, t.shape[0]), np.sort(t))


def _idt(source: np.ndarray, target: np.ndarray, iterations: int, seed: int = 0) -> np.ndarray:
    """Pitié Iterated Distribution Transfer: transport ``source`` onto ``target``'s full
    distribution via random rotations + per-axis 1D CDF matching."""
    rng = np.random.default_rng(seed)
    s = source.copy()
    for _ in range(iterations):
        rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))  # random orthonormal basis
        sr = s @ rot
        tr = target @ rot
        for ax in range(3):
            sr[:, ax] = _cdf_match_1d(sr[:, ax], tr[:, ax])
        s = sr @ rot.T
    return s


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
    """Optimal-transport Look Fitter (ADR-0010/0011/0013). `fit(consensus) -> LookTransform`.

    ``method`` = "mkl" (closed-form, mean+covariance) or "pdf" (Pitié IDT, full distribution).
    ``space`` = "oklab" (default, perceptual) or "rgb". ``tone_strength`` (0..1) preserves input
    lightness while keeping the transported palette (lower = keep brightness)."""

    def __init__(self, tone_strength: float = 1.0, space: str = "oklab", method: str = "mkl",
                 iterations: int = 16, smoothing: float = 0.025, size: int = DEFAULT_SIZE):
        self._tone = float(np.clip(tone_strength, 0.0, 1.0))
        if space not in ("oklab", "rgb"):
            raise ValueError(f"space must be 'oklab' or 'rgb', got {space!r}")
        if method not in ("mkl", "pdf"):
            raise ValueError(f"method must be 'mkl' or 'pdf', got {method!r}")
        self._space = space
        self._method = method
        self._iters = int(iterations)
        self._smoothing = float(smoothing)
        self._size = size

    def fit(self, consensus: ConsensusLook, source_samples=None) -> LookTransform:
        src = load_base() if source_samples is None else np.asarray(source_samples, dtype=np.float64)
        src = src.reshape(-1, 3)
        if self._method == "pdf":
            return self._fit_pdf(consensus, src)

        if self._space == "oklab":
            x = to_oklab(src)
            mu_t, cov_t = consensus.mean_oklab, consensus.cov_oklab
        else:
            x = src
            mu_t, cov_t = consensus.mean, consensus.covariance
        mu_s = x.mean(axis=0)
        cov_s = np.cov(x, rowvar=False) + _EPS * np.eye(3)
        matrix = _mkl_matrix(cov_s, _sym(cov_t) + _EPS * np.eye(3))
        return _RichLookTransform(mu_s, matrix, mu_t, self._tone, space=self._space)

    def _fit_pdf(self, consensus: ConsensusLook, src: np.ndarray) -> LookTransform:
        # Pitié IDT in Oklab, then fit a smooth continuous grade cube from source -> transported.
        src_lab = to_oklab(src)
        looked_lab = _idt(src_lab, consensus.samples_oklab, self._iters)
        if self._tone < 1.0:
            looked_lab[:, 0] = src_lab[:, 0] + self._tone * (looked_lab[:, 0] - src_lab[:, 0])
        looked_rgb = from_oklab(looked_lab)
        grade = learn_grade_cube(src, looked_rgb, self._size, self._smoothing, 1e-3)
        return CubeLookTransform(grade, self._size)
