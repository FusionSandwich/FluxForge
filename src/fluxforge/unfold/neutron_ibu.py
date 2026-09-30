"""
Neutron spectrum unfolding via Iterative Bayesian Unfolding (PyUnfold).

This optional wrapper runs PyUnfold on probability/count inputs. Activation
reaction rates and cross-section responses are dimensionful and cannot be
passed directly as a physical neutron-flux comparator.

It wraps the PyUnfold library (D'Agostini iterative Bayesian unfolding)
and provides a FluxForge-idiomatic interface.

Library dependency
------------------
Requires the optional ``pyunfold`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

from fluxforge.core.unfolding_diagnostics import merge_flux_diagnostics
from fluxforge.core.unfolding_inputs import require_nonnegative

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
            "NeutronUnfolderIBU requires the optional PyUnfold package"
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
        Legacy field name for PyUnfold's cause-count estimate, not flux.
    statistical_uncertainty : np.ndarray
        Statistical uncertainty on the unfolded cause distribution.
    systematic_uncertainty : np.ndarray
        Systematic uncertainty from response matrix statistics.
    flux_covariance : np.ndarray | None
        None: PyUnfold returns marginal errors, not full covariance.
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

    def __post_init__(self) -> None:
        self.diagnostics = merge_flux_diagnostics(
            self.diagnostics,
            self.unfolded_flux,
        )

    @property
    def cause_counts(self) -> np.ndarray:
        """PyUnfold's unfolded cause distribution in count units."""
        return self.unfolded_flux


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NeutronUnfolderIBU:
    """
    Neutron flux unfolder using PyUnfold (D'Agostini Iterative Bayesian).

    This wrapper is usable for PyUnfold's cause/effect count model. It is not
    a physical activation-spectrum comparator until a qualified adapter maps
    rates and covariance into that model.

    The mapping of FluxForge concepts to PyUnfold inputs is:

    +-----------------------+---------------------------+
    | FluxForge             | PyUnfold                  |
    +=======================+===========================+
    | Effect counts         | data                      |
    +-----------------------+---------------------------+
    | Conditional P(E|C)    | response                  |
    +-----------------------+---------------------------+
    | Cause distribution    | prior                     |
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
            Effect counts and uncertainties, marked ``quantity='effect_counts'``.
        response : ResponseBundle
            Conditional probabilities, marked
            ``quantity='conditional_probability'``.
        prior_flux : np.ndarray, optional
            Prior cause distribution. If None, PyUnfold uses a uniform prior.
        efficiencies : np.ndarray, optional
            Detection efficiencies per cause bin; required.
        efficiencies_err : np.ndarray, optional
            Efficiency uncertainties; required.
        response_err : np.ndarray, optional
            Probability-response uncertainties; required.

        Returns
        -------
        NeutronIBUResult
            Cause-count estimate, marginal uncertainties, and diagnostics.
        """
        # --- Unpack inputs ---
        data = require_nonnegative("data", reaction_rates.values).reshape(-1)
        data_err = require_nonnegative("data_err", reaction_rates.uncertainties).reshape(-1)

        if reaction_rates.quantity != "effect_counts" or response.quantity != "conditional_probability":
            raise ValueError(
                "PyUnfold requires effect_counts and conditional_probability inputs; "
                "activation rates/cross sections need a qualified adapter"
            )
        if response_err is None or efficiencies is None or efficiencies_err is None:
            raise ValueError(
                "Explicit response_err, efficiencies, and efficiencies_err are required"
            )

        R = require_nonnegative("response", response.matrix)
        n_effects, n_causes = R.shape
        if np.any(R > 1):
            raise ValueError("Conditional response entries must be probabilities <= 1")

        if data.shape[0] != n_effects:
            raise ValueError(
                f"reaction_rates.values length {data.shape[0]} != "
                f"response rows {n_effects}"
            )

        # --- Response uncertainty ---
        R_err = require_nonnegative("response_err", response_err)
        if R_err.shape != R.shape:
            raise ValueError("response_err shape must match the response shape")

        # --- Efficiencies (detection efficiency per cause bin) ---
        eff = require_nonnegative("efficiencies", efficiencies).reshape(-1)
        if eff.size != n_causes:
            raise ValueError(
                f"efficiencies size {eff.size} != response columns {n_causes}"
            )

        eff_err = require_nonnegative("efficiencies_err", efficiencies_err).reshape(-1)
        if eff_err.size != n_causes:
            raise ValueError(
                f"efficiencies_err size {eff_err.size} != response columns {n_causes}"
            )
        if np.any(eff <= 0) or np.any(eff > 1) or not np.allclose(
            np.sum(R, axis=0), eff, rtol=1e-8, atol=1e-12
        ):
            raise ValueError("Efficiencies must equal positive response column sums <= 1")

        # --- Prior ---
        if prior_flux is not None:
            prior = require_nonnegative("prior", prior_flux).reshape(-1)
            if prior.size != n_causes:
                raise ValueError(
                    f"prior_flux size {prior.size} != response columns {n_causes}"
                )
            # Normalize to be a probability distribution
            if not np.any(prior > 0):
                raise ValueError("Prior cause distribution must have positive mass")
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

        return NeutronIBUResult(
            unfolded_flux=unfolded,
            statistical_uncertainty=stat_err,
            systematic_uncertainty=sys_err,
            flux_covariance=None,
            n_iterations=int(result.get("num_iterations", 0)),
            test_statistic=float(result.get("ts_iter", 0.0)),
            unfolding_matrix=result.get("unfolding_matrix"),
            diagnostics=merge_flux_diagnostics(
                {
                    **{
                        k: v
                        for k, v in result.items()
                        if k not in (
                            "unfolded", "stat_err", "sys_err",
                            "num_iterations", "ts_iter", "unfolding_matrix",
                        )
                    },
                    "output_quantity": "cause_counts",
                    "physical_activation_comparator": False,
                    "covariance_status": "full covariance unavailable; marginal errors only",
                },
                unfolded,
                negative_policy="bayesian_posterior_nonnegative",
                nonnegativity_enforced=True,
            ),
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
        comparison_kind: str = "",
    ) -> Dict:
        """
        Compare synthetic cause-count vectors after explicit basis assertion.

        Parameters
        ----------
        gls_flux : np.ndarray
            Synthetic cause-count estimate to compare.
        ibu_result : NeutronIBUResult
            Result from :meth:`solve`.
        rtol : float
            Relative tolerance for agreement (default 25%).
        comparison_kind : str
            Must be ``synthetic_cause_counts``. Physical flux cannot be
            compared directly to PyUnfold's cause-count output.

        Returns
        -------
        dict
            Comparison diagnostics including agreement flag.
        """
        if comparison_kind != "synthetic_cause_counts":
            raise ValueError(
                "Comparison requires an explicit synthetic cause-count basis; "
                "PyUnfold output is not physical neutron flux"
            )
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
