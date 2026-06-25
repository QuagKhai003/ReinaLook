"""rich — the quality Look Fitter (Monge-Kantorovich linear optimal transport).

@context  Captures the references' PALETTE (mean + full color covariance / channel correlations),
          not just per-channel stats like Mid. Closed-form MKL (Pitie-Kokaram): the affine map
          that matches both mean and covariance. Drop-in for MidFitter (same interface, ADR-0010).
@done     RichFitter.fit (MKL via PSD matrix square roots) + _RichLookTransform (+ tone_strength).
@todo     Perceptual-space (Oklab) MKL; nonlinear N-d PDF transfer (Plan §3 rung 2).
@limits   PURE numeric (fit reads the fixed base via load_base unless given source_samples).
          Rec.709 g2.4 space. Final clamp left to regularize. eps regularizes Sigma_source.
          Robustness: cross-channel covariance is shrunk (_CROSS) and the MKL stretch is capped
          (_MAX_STRETCH) so an ill-conditioned source (e.g. a tiny neutral pool) can't rotate hue
          or explode colors out of gamut (would otherwise turn mid-tones magenta).
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


_MAX_STRETCH = 3.0    # cap how much MKL may scale any color axis (prevents gamut explosion)
_CROSS = 0.0          # how much cross-channel (hue-rotating) covariance to keep; <1 = more robust


def _shrink_cross(cov: np.ndarray, keep: float) -> np.ndarray:
    """Shrink the off-diagonal (cross-channel) covariance toward 0. Full cross terms let MKL rotate
    hue, which can send mid-tones to a wrong hue (e.g. skin → magenta) when source and target
    distributions differ a lot. Keeping the diagonal preserves per-channel scale (safe)."""
    d = np.diag(np.diag(cov))
    return d + keep * (cov - d)


def _mkl_matrix(cov_s: np.ndarray, cov_t: np.ndarray, max_stretch: float = _MAX_STRETCH) -> np.ndarray:
    """Monge-Kantorovich linear map A with A.Sigma_s.A^T = Sigma_t (symmetric solution).

    A's eigenvalues are clipped to [1/max_stretch, max_stretch]: when the source distribution is
    narrow/ill-conditioned (e.g. a small neutral pool), the raw map can scale a color axis enormously
    and blow colors out of gamut (hue-rotated, posterized). Capping the stretch keeps the mean +
    moderate covariance match while bounding the worst case; gentle, well-conditioned fits are
    unaffected (their eigenvalues are already within the cap)."""
    cov_s = _shrink_cross(cov_s, _CROSS)
    cov_t = _shrink_cross(cov_t, _CROSS)
    s_half = _psd_pow(cov_s, 0.5)
    s_ihalf = _psd_pow(cov_s, -0.5)
    middle = _psd_pow(s_half @ cov_t @ s_half, 0.5)
    a = s_ihalf @ middle @ s_ihalf
    w, v = np.linalg.eigh(_sym(a))                       # symmetric PSD → eig = singular values
    w = np.clip(w, 1.0 / max_stretch, max_stretch)
    return (v * w) @ v.T


def _cdf_match_1d(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map values ``s`` onto the 1D distribution of ``t`` (rank → target quantile)."""
    ranks = np.empty(s.shape[0])
    ranks[np.argsort(s)] = np.linspace(0.0, 1.0, s.shape[0])
    return np.interp(ranks, np.linspace(0.0, 1.0, t.shape[0]), np.sort(t))


def _idt(source: np.ndarray, target: np.ndarray, iterations: int, seed: int = 0) -> np.ndarray:
    """Pitié Iterated Distribution Transfer: transport ``source`` onto ``target``'s full
    distribution via random rotations + per-axis 1D CDF matching. (Vectorized over the 3 axes;
    bit-for-bit identical to the per-axis loop.)"""
    rng = np.random.default_rng(seed)
    s = source.copy()
    s_pos = np.linspace(0.0, 1.0, s.shape[0])              # source rank positions (constant)
    t_pos = np.linspace(0.0, 1.0, target.shape[0])         # target rank positions (constant)
    for _ in range(iterations):
        rot, _ = np.linalg.qr(rng.standard_normal((3, 3)))  # random orthonormal basis
        sr = s @ rot
        tr = np.sort(target @ rot, axis=0)                  # target sorted per axis, once
        order = np.argsort(sr, axis=0)                      # one call for all 3 axes
        for ax in range(3):
            ranks = np.empty(sr.shape[0])
            ranks[order[:, ax]] = s_pos
            sr[:, ax] = np.interp(ranks, t_pos, tr[:, ax])
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
        # mkl in Oklab clips/hue-rotates badly on strong looks (the a/b axes blow out of gamut),
        # so mkl is forced to RGB. Oklab is for pdf, where it helps.
        self._space = "rgb" if method == "mkl" else space
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

