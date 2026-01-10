"""
Neutron spectrum unfolding via Iterative Bayesian Unfolding (PyUnfold).

This module implements an *optional* cross-check solver for **Stage G**
of FluxForge: unfolding the neutron flux spectrum from activation
reaction-rate measurements.

It wraps the PyUnfold library (D'Agostini iterative Bayesian unfolding)
and provides a FluxForge-idiomatic interface.

Library dependency
------------------
Requires the optional ``pyunfold`` package (pip install pyunfold).

Usage example
-------------
>>> from fluxforge.unfold.neutron_ibu import NeutronUnfolderIBU
>>> solver = NeutronUnfolderIBU()
>>> result = solver.solve(reaction_rates, response_bundle, prior_flux)
>>> print(result.unfolded_flux)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    from pyunfold import iterative_unfold as _pyunfold_unfold

    _HAS_PYUNFOLD = True
except ImportError:  # pragma: no cover
    _HAS_PYUNFOLD = False
    _pyunfold_unfold = None  # type: ignore


def _require_pyunfold() -> None:
    if not _HAS_PYUNFOLD:
        raise ImportError(
            "NeutronUnfolderIBU requires PyUnfold. Install with: pip install pyunfold"
        )


# ---------------------------------------------------------------------------
# Local type definitions (mirroring FluxForge domain objects)
# ---------------------------------------------------------------------------
from ._types import ReactionRates, ResponseBundle


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class NeutronIBUResult:
    """Result container for neutron IBU unfolding.

    Attributes
    ----------
    unfolded_flux : np.ndarray
        Posterior estimate of the neutron flux spectrum, shape (N_groups,).
    statistical_uncertainty : np.ndarray
        Statistical (Poisson) uncertainty on unfolded_flux.
    systematic_uncertainty : np.ndarray
        Systematic uncertainty from response matrix statistics.
    flux_covariance : np.ndarray | None
        Full covariance matrix if available, shape (N_groups, N_groups).
    n_iterations : int
        Number of unfolding iterations performed.
    test_statistic : float
        Final iteration test statistic value.
    unfolding_matrix : np.ndarray | None
        Bayesian unfolding matrix (posterior probabilities).
    diagnostics : dict
        Additional diagnostic info returned by PyUnfold.
    """

    unfolded_flux: np.ndarray
    statistical_uncertainty: np.ndarray
    systematic_uncertainty: np.ndarray
    flux_covariance: Optional[np.ndarray] = None
    n_iterations: int = 0
    test_statistic: float = 0.0
    unfolding_matrix: Optional[np.ndarray] = None
    diagnostics: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NeutronUnfolderIBU:
    """
    Neutron flux unfolder using PyUnfold (D'Agostini Iterative Bayesian).

    This solver provides an independent cross-check to the primary STAYSL-style
    generalized least-squares (GLS) solver.  It is guaranteed to produce
    non-negative fluxes and handles non-Gaussian priors gracefully.

    The mapping of FluxForge concepts to PyUnfold inputs is:

    +-----------------------+---------------------------+
    | FluxForge             | PyUnfold                  |
    +=======================+===========================+
    | Reaction rates        | data (effects / counts)   |
    +-----------------------+---------------------------+
    | Cross-section matrix  | response (mixing matrix)  |
    +-----------------------+---------------------------+
    | Prior neutron flux    | prior / efficiencies      |
    +-----------------------+---------------------------+

    Parameters
    ----------
    ts : str
        Test statistic for stopping: 'ks' (default), 'chi2', 'bf', 'rmd'.
    ts_stopping : float
        Stopping threshold (default 0.01).
    max_iter : int
        Maximum iterations (default 100).
    cov_type : str
        Covariance form: 'multinomial' or 'poisson'.
    """

    def __init__(
        self,
        ts: str = "ks",
        ts_stopping: float = 0.01,
        max_iter: int = 100,
        cov_type: str = "multinomial",
    ) -> None:
        _require_pyunfold()

        self._ts = ts
        self._ts_stopping = ts_stopping
        self._max_iter = max_iter
        self._cov_type = cov_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(
        self,
        reaction_rates: ReactionRates,
        response: ResponseBundle,
        prior_flux: Optional[np.ndarray] = None,
        *,
        efficiencies: Optional[np.ndarray] = None,
        efficiencies_err: Optional[np.ndarray] = None,
        response_err: Optional[np.ndarray] = None,
    ) -> NeutronIBUResult:
        """
        Perform iterative Bayesian unfolding.

        Parameters
        ----------
        reaction_rates : ReactionRates
            Measured SigPhi values and uncertainties.
        response : ResponseBundle
            Energy-dependent cross-section mixing matrix.
        prior_flux : np.ndarray, optional
            A priori neutron flux (e.g., from OpenMC). If None, a uniform
            prior is used internally.
        efficiencies : np.ndarray, optional
            Detection efficiencies per energy group. If None, assumed 1.0.
        efficiencies_err : np.ndarray, optional
            Efficiency uncertainties. If None, assumed 1% of efficiency.
        response_err : np.ndarray, optional
            Uncertainties on response matrix. If None, assumed 5% relative.

        Returns
        -------
        NeutronIBUResult
            Unfolded flux, uncertainties, covariance, and diagnostics.
        """
        # --- Unpack inputs ---
        data = np.asarray(reaction_rates.values, dtype=float)
        data_err = np.asarray(reaction_rates.uncertainties, dtype=float)

        R = np.asarray(response.matrix, dtype=float)
        n_effects, n_causes = R.shape

        if data.shape[0] != n_effects:
            raise ValueError(
                f"reaction_rates.values length {data.shape[0]} != "
                f"response rows {n_effects}"
            )

        # --- Response uncertainty ---
        if response_err is not None:
            R_err = np.asarray(response_err, dtype=float)
        else:
            R_err = np.abs(R) * 0.05  # default 5% relative

        # --- Efficiencies (detection efficiency per cause bin) ---
        if efficiencies is not None:
            eff = np.asarray(efficiencies, dtype=float)
        else:
            eff = np.ones(n_causes, dtype=float)

        if efficiencies_err is not None:
            eff_err = np.asarray(efficiencies_err, dtype=float)
        else:
            eff_err = eff * 0.01

        # --- Prior ---
        if prior_flux is not None:
            prior = np.asarray(prior_flux, dtype=float)
            if prior.size != n_causes:
                raise ValueError(
                    f"prior_flux size {prior.size} != response columns {n_causes}"
                )
            # Normalize to be a probability distribution
            prior = prior / np.sum(prior)
        else:
            prior = None  # PyUnfold will use uniform

        # --- Call PyUnfold ---
        result = _pyunfold_unfold(
            data=data,
            data_err=data_err,
            response=R,
            response_err=R_err,
            efficiencies=eff,
            efficiencies_err=eff_err,
            prior=prior,
            ts=self._ts,
            ts_stopping=self._ts_stopping,
            max_iter=self._max_iter,
            cov_type=self._cov_type,
            return_iterations=False,
        )

        # --- Build output ---
        unfolded = np.asarray(result["unfolded"], dtype=float)
        stat_err = np.asarray(result["stat_err"], dtype=float)
        sys_err = np.asarray(result["sys_err"], dtype=float)

        # PyUnfold doesn't directly return full covariance; approximate from uncertainties
        total_var = stat_err**2 + sys_err**2
        cov = np.diag(total_var)

        return NeutronIBUResult(
            unfolded_flux=unfolded,
            statistical_uncertainty=stat_err,
            systematic_uncertainty=sys_err,
            flux_covariance=cov,
            n_iterations=int(result.get("num_iterations", 0)),
            test_statistic=float(result.get("ts_iter", 0.0)),
            unfolding_matrix=result.get("unfolding_matrix"),
            diagnostics={
                k: v
                for k, v in result.items()
                if k
                not in (
                    "unfolded",
                    "stat_err",
                    "sys_err",
                    "num_iterations",
                    "ts_iter",
                    "unfolding_matrix",
                )
            },
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def compare_with_gls(
        self,
        gls_flux: np.ndarray,
        ibu_result: NeutronIBUResult,
        *,
        rtol: float = 0.25,
    ) -> Dict:
        """
        Compare IBU result to a GLS solution.

        Parameters
        ----------
        gls_flux : np.ndarray
            Flux spectrum from GLS solver.
        ibu_result : NeutronIBUResult
            Result from :meth:`solve`.
        rtol : float
            Relative tolerance for agreement (default 25%).

        Returns
        -------
        dict
            Comparison diagnostics including agreement flag.
        """
        gls = np.asarray(gls_flux, dtype=float)
        ibu = ibu_result.unfolded_flux

        if gls.size != ibu.size:
            raise ValueError("GLS and IBU flux arrays must have same size")

        diff = np.abs(gls - ibu)
        scale = np.maximum(np.abs(gls), np.abs(ibu)) + 1e-30
        rel_diff = diff / scale

        agrees = bool(np.all(rel_diff < rtol))

        return {
            "agrees": agrees,
            "max_relative_difference": float(np.max(rel_diff)),
            "mean_relative_difference": float(np.mean(rel_diff)),
            "gls_has_negatives": bool(np.any(gls < 0)),
            "ibu_has_negatives": bool(np.any(ibu < 0)),
        }
