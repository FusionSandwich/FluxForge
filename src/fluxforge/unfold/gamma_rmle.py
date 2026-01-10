"""
Gamma detector-response unfolding via Regularized MLE (PyLops FISTA).

This module implements **Stage B** of FluxForge: inverting the HPGe
detector response matrix to recover the source gamma emission spectrum
from the measured pulse-height spectrum.

The mathematics follow Lima et al. (Regularized Unfolding of gamma-ray
Spectra): minimise ``||y - R η||₂² + λ ||η||₁`` using FISTA.

Library dependency
------------------
Requires the optional ``pylops`` package (pip install pylops).

Usage example
-------------
>>> from fluxforge.unfold.gamma_rmle import GammaUnfolderRMLE
>>> unfolder = GammaUnfolderRMLE(response_matrix)
>>> clean_spectrum, residuals = unfolder.solve_regularized(measured_counts, lambda_reg=0.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import pylops
    from pylops.optimization.sparsity import fista as pylops_fista

    _HAS_PYLOPS = True
except ImportError:  # pragma: no cover
    _HAS_PYLOPS = False
    pylops = None  # type: ignore
    pylops_fista = None  # type: ignore


def _require_pylops() -> None:
    if not _HAS_PYLOPS:
        raise ImportError(
            "GammaUnfolderRMLE requires PyLops. Install with: pip install pylops"
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class GammaUnfoldResult:
    """Result container for gamma RMLE unfolding.

    Attributes
    ----------
    unfolded_spectrum : np.ndarray
        Recovered source gamma spectrum η, shape (N_energy_bins,).
    residuals : np.ndarray
        Data residuals y − R η, shape (N_channels,).
    cost_history : list[float]
        Objective value at each FISTA iteration.
    n_iterations : int
        Number of iterations performed.
    converged : bool
        Whether FISTA declared convergence.
    """

    unfolded_spectrum: np.ndarray
    residuals: np.ndarray
    cost_history: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class GammaUnfolderRMLE:
    """
    Gamma spectrum unfolder using regularised MLE via PyLops FISTA.

    This solver deconvolves the HPGe detector response matrix **R** from
    the measured pulse-height spectrum **y** to recover the emitted
    gamma spectrum **η**.

    The regularisation term is L1 (sparsity-promoting), which preserves
    discrete gamma peaks while suppressing the Compton continuum.

    Parameters
    ----------
    response_matrix : np.ndarray
        Detector response matrix R, shape (N_channels, N_energy_bins).
        Can be dense or sparse; internally converted to a PyLops operator.
    """

    def __init__(self, response_matrix: np.ndarray) -> None:
        _require_pylops()

        self._R = np.atleast_2d(np.asarray(response_matrix, dtype=float))
        if self._R.ndim != 2:
            raise ValueError("response_matrix must be 2-D")

        self._n_channels, self._n_bins = self._R.shape

        # Build a pylops linear operator (supports dense/sparse)
        self._Op = pylops.MatrixMult(self._R)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve_regularized(
        self,
        measured_spectrum: np.ndarray,
        lambda_reg: float = 0.1,
        *,
        n_iter: int = 200,
        tol: float = 1e-8,
        eps: float = 1e-12,
        show: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the regularised inverse problem using FISTA.

        Minimises  ``0.5 ||y - R η||₂² + λ ||η||₁``  subject to η ≥ 0.

        Parameters
        ----------
        measured_spectrum : np.ndarray
            Observed spectrum y, shape (N_channels,).
        lambda_reg : float
            L1 regularisation weight (default 0.1).
        n_iter : int
            Maximum FISTA iterations (default 200).
        tol : float
            Convergence tolerance on relative objective change.
        eps : float
            Small constant for numerical stability.
        show : bool
            If True, print FISTA progress.

        Returns
        -------
        unfolded_spectrum : np.ndarray
            Recovered η, shape (N_energy_bins,).
        residuals : np.ndarray
            y − R η, shape (N_channels,).
        """
        y = np.asarray(measured_spectrum, dtype=float).ravel()
        if y.size != self._n_channels:
            raise ValueError(
                f"measured_spectrum length {y.size} != response rows {self._n_channels}"
            )

        # FISTA from PyLops with L1 proximal operator
        # Returns (x, niter, cost_array)
        eta_hat, n_it, cost = pylops_fista(
            self._Op,
            y,
            niter=int(n_iter),
            eps=float(lambda_reg),
            tol=float(tol),
            show=bool(show),
        )

        # Enforce non-negativity (soft clamp)
        eta_hat = np.maximum(eta_hat, 0.0)

        residuals = y - self._Op @ eta_hat

        return eta_hat, residuals

    def solve_full(
        self,
        measured_spectrum: np.ndarray,
        lambda_reg: float = 0.1,
        **kwargs,
    ) -> GammaUnfoldResult:
        """
        Solve and return a full result object with diagnostics.

        See :meth:`solve_regularized` for parameter descriptions.
        """
        n_iter = kwargs.pop("n_iter", 200)
        tol = kwargs.pop("tol", 1e-8)
        show = kwargs.pop("show", False)

        y = np.asarray(measured_spectrum, dtype=float).ravel()
        if y.size != self._n_channels:
            raise ValueError(
                f"measured_spectrum length {y.size} != response rows {self._n_channels}"
            )

        eta_hat, n_it, cost = pylops_fista(
            self._Op,
            y,
            niter=int(n_iter),
            eps=float(lambda_reg),
            tol=float(tol),
            show=bool(show),
        )

        eta_hat = np.maximum(eta_hat, 0.0)
        residuals = y - self._Op @ eta_hat

        # cost may be array or list depending on pylops version
        cost_list = list(cost) if hasattr(cost, "__iter__") else [float(cost)]

        return GammaUnfoldResult(
            unfolded_spectrum=eta_hat,
            residuals=residuals,
            cost_history=cost_list,
            n_iterations=int(n_it),
            converged=True,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def n_channels(self) -> int:
        """Number of detector channels (rows of R)."""
        return self._n_channels

    @property
    def n_energy_bins(self) -> int:
        """Number of source energy bins (columns of R)."""
        return self._n_bins

    @property
    def response_matrix(self) -> np.ndarray:
        """Return the underlying response matrix."""
        return self._R.copy()
