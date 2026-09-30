"""Source-bound, linear Gaussian adjustment of group-integral neutron flux.

This module does not construct activities, cross sections, shielding, or a
prior. Callers must qualify and freeze those inputs before using the result.
The returned fit is a mathematical adjustment, not experimental validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fluxforge.analysis.holdout_validation import predict_holdouts


def _array(values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


def _hash(array: np.ndarray) -> str:
    """Hash exact float64 values and shape in a platform-independent order."""
    canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
    shape = ",".join(map(str, canonical.shape)).encode("ascii")
    return hashlib.sha256(shape + b"\0" + canonical.tobytes()).hexdigest()


def _covariance(values: object, size: int, name: str) -> np.ndarray:
    cov = _array(values, (size, size), name)
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=0):
        raise ValueError(f"{name} must be symmetric")
    cov = (cov + cov.T) / 2
    scale = float(np.max(np.abs(cov)))
    if scale:
        eigenvalues = np.linalg.eigvalsh(cov / scale)
        tolerance = 10 * size * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(eigenvalues)))
        )
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite")
    return cov


@dataclass(frozen=True)
class MonitorRow:
    observation_id: str
    sample_id: str
    cover: str
    reaction_id: str
    product_id: str

    def __post_init__(self) -> None:
        if not all((self.observation_id, self.sample_id, self.cover,
                    self.reaction_id, self.product_id)):
            raise ValueError("Every physical monitor identity field is required")


@dataclass(frozen=True)
class SourceBinding:
    uri: str
    sha256: str
    units: str

    def __post_init__(self) -> None:
        if not self.uri or not self.units or not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("Source binding needs URI, SHA256, and units")


@dataclass
class PhysicalGLSResult:
    energy_edges_eV: np.ndarray
    rows: tuple[MonitorRow, ...]
    fit_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    prior_flux: np.ndarray
    prior_covariance: np.ndarray
    flux: np.ndarray
    posterior_covariance: np.ndarray
    observation_covariance: np.ndarray
    response_error_covariance: np.ndarray
    response_matrix: np.ndarray
    measured_rates: np.ndarray
    predicted_rates: np.ndarray
    residuals: np.ndarray
    fit_predictions: np.ndarray
    fit_residuals: np.ndarray
    prior_innovation_chi2: float
    postfit_observation_chi2: float
    effective_residual_df: float
    response_rank: int
    response_nullity: int
    response_condition: float
    negative_flux_groups: tuple[int, ...]
    input_hashes: Mapping[str, str]
    sources: Mapping[str, SourceBinding]
    source_commit: str
    implementation_sha256: str
    solver_flags: Mapping[str, object]
    holdout_predictions: np.ndarray
    holdout_residuals: np.ndarray
    holdout_predictive_covariance: np.ndarray
    holdout_standardized_chi2: float | None

    @property
    def flux_uncertainty(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.posterior_covariance), 0))

    def receipt(self) -> dict:
        """JSON-compatible input and diagnostic receipt; no validation claim."""
        return {
            "method": "physical_linear_gls", "scientific_admission": False,
            "source_qualification": "caller supplied bindings; source bytes not verified",
            "flux_definition": "group_integral", "flux_units": "n/cm2/s",
            "rate_units": "reactions/target_atom/s", "response_units": "cm2",
            "energy_edges_eV": self.energy_edges_eV.tolist(),
            "rows": [vars(row) for row in self.rows],
            "fit_ids": list(self.fit_ids), "holdout_ids": list(self.holdout_ids),
            "prior_flux": self.prior_flux.tolist(), "flux": self.flux.tolist(),
            "prior_covariance": self.prior_covariance.tolist(),
            "posterior_covariance": self.posterior_covariance.tolist(),
            "observation_covariance": self.observation_covariance.tolist(),
            "response_error_covariance": self.response_error_covariance.tolist(),
            "response_matrix": self.response_matrix.tolist(),
            "measured_rates": self.measured_rates.tolist(),
            "predicted_rates": self.predicted_rates.tolist(),
            "predicted_rates_definition": "unconditional forward fold of fitted flux",
            "residuals": self.residuals.tolist(),
            "fit_predictions": self.fit_predictions.tolist(),
            "fit_residuals": self.fit_residuals.tolist(),
            "prior_innovation_chi2": self.prior_innovation_chi2,
            "postfit_observation_chi2": self.postfit_observation_chi2,
            "postfit_chi2_definition": "fit residual^T fit observation covariance^-1 fit residual; descriptive for regularized fit",
            "effective_residual_df": self.effective_residual_df,
            "response_rank": self.response_rank,
            "response_nullity": self.response_nullity,
            "response_condition": self.response_condition,
            "negative_flux_groups": list(self.negative_flux_groups),
            "holdout_predictions": self.holdout_predictions.tolist(),
            "holdout_predictions_definition": "joint-Gaussian prediction conditioned on fit rows",
            "holdout_residuals": self.holdout_residuals.tolist(),
            "holdout_predictive_covariance": self.holdout_predictive_covariance.tolist(),
            "holdout_standardized_chi2": self.holdout_standardized_chi2,
            "input_hashes": dict(self.input_hashes),
            "sources": {name: vars(binding) for name, binding in self.sources.items()},
            "source_commit": self.source_commit,
            "implementation_sha256": self.implementation_sha256,
            "solver_flags": dict(self.solver_flags),
        }


def unfold_gls_physical(
    *, rows: Sequence[MonitorRow], energy_edges_eV: object,
    measured_rates: object, response_matrix: object, prior_flux: object,
    prior_covariance: object, observation_covariance: object,
    sources: Mapping[str, SourceBinding], source_commit: str,
    holdout_ids: Sequence[str] = (), response_error_covariance: object | None = None,
) -> PhysicalGLSResult:
    """Adjust a group-integral prior with an explicit sample-specific operator.

    Response maps group-integral n/cm2/s to reactions/target_atom/s, so each
    coefficient has units cm2. Observation and response-error covariances have
    squared rate units. Response error is frozen observation-space uncertainty
    at the supplied prior; it is not recomputed after adjustment. Holdout rows
    are predicted only after fit, and their activities never enter the solve.
    Correlated fit and holdout errors are included in the conditional
    predictive distribution. Holdout rates never enter the fit.
    """
    rows = tuple(rows)
    m = len(rows)
    if not m or any(not isinstance(row, MonitorRow) for row in rows):
        raise ValueError("Explicit physical monitor rows are required")
    ids = tuple(row.observation_id for row in rows)
    if len(set(ids)) != m:
        raise ValueError("Observation IDs must be unique, including bare/Cd rows")
    physical_keys = [(r.sample_id, r.cover, r.reaction_id, r.product_id) for r in rows]
    if len(set(physical_keys)) != m:
        raise ValueError("Duplicate physical monitor/product identity")
    edges = np.asarray(energy_edges_eV, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("Energy edges must be finite, positive, and strictly increasing")
    n = len(edges) - 1
    rates = _array(measured_rates, (m,), "measured_rates")
    response = _array(response_matrix, (m, n), "response_matrix")
    prior = _array(prior_flux, (n,), "prior_flux")
    if np.any(prior < 0) or np.any(response < 0):
        raise ValueError("Prior flux and physical response must be nonnegative")
    prior_cov = _covariance(prior_covariance, n, "prior_covariance")
    obs_cov = _covariance(observation_covariance, m, "observation_covariance")
    response_cov = (np.zeros((m, m)) if response_error_covariance is None else
                    _covariance(response_error_covariance, m, "response_error_covariance"))
    required = {"row_identities", "energy_edges", "rates", "response", "prior", "prior_covariance", "observation_covariance"}
    if response_error_covariance is not None:
        required.add("response_error_covariance")
    if not source_commit or not required.issubset(sources) or any(
        not isinstance(sources[key], SourceBinding) for key in required
    ):
        raise ValueError("Commit and source bindings for every numerical input are required")
    expected_units = {"row_identities": "identity", "energy_edges": "eV", "rates": "reactions/target_atom/s",
                      "response": "cm2", "prior": "n/cm2/s",
                      "prior_covariance": "(n/cm2/s)^2",
                      "observation_covariance": "(reactions/target_atom/s)^2",
                      "response_error_covariance": "(reactions/target_atom/s)^2"}
    if any(sources[key].units != expected_units[key] for key in required):
        raise ValueError("Source binding units disagree with group-integral GLS contract")
    holdout = tuple(holdout_ids)
    if len(set(holdout)) != len(holdout) or not set(holdout).issubset(ids):
        raise ValueError("Holdout IDs must be unique observation IDs")
    fit_index = np.array([i for i, identity in enumerate(ids) if identity not in holdout], dtype=int)
    hold_index = np.array([i for i, identity in enumerate(ids) if identity in holdout], dtype=int)
    if not len(fit_index):
        raise ValueError("At least one fit observation is required")
    total_cov = obs_cov + response_cov
    a = response[fit_index]
    y = rates[fit_index]
    v = total_cov[np.ix_(fit_index, fit_index)]
    v_scale = float(np.max(np.abs(v)))
    if not v_scale:
        raise ValueError("Total fit covariance must be positive definite")
    try:
        np.linalg.cholesky(v / v_scale)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Total fit covariance must be positive definite") from exc
    # Scale rate rows and flux groups before factorization; no absolute floor.
    row_scale = np.sqrt(np.diag(v))
    flux_scale = max(float(np.max(np.abs(prior))),
                     float(np.sqrt(np.max(np.diag(prior_cov)))), np.finfo(float).tiny)
    a_scaled = a * (flux_scale / row_scale[:, None])
    y_scaled = y / row_scale
    p_scaled = prior / flux_scale
    c_scaled = prior_cov / flux_scale**2
    v_scaled = v / row_scale[:, None] / row_scale[None, :]
    innovation = y_scaled - a_scaled @ p_scaled
    s = a_scaled @ c_scaled @ a_scaled.T + v_scaled
    try:
        gain = np.linalg.solve(s, a_scaled @ c_scaled).T
        adjusted = p_scaled + gain @ innovation
        identity_minus = np.eye(n) - gain @ a_scaled
        posterior_scaled = (identity_minus @ c_scaled @ identity_minus.T
                            + gain @ v_scaled @ gain.T)
        innovation_chi2 = float(innovation @ np.linalg.solve(s, innovation))
        fit_residual_scaled = y_scaled - a_scaled @ adjusted
        postfit_chi2 = float(fit_residual_scaled @ np.linalg.solve(v_scaled, fit_residual_scaled))
    except np.linalg.LinAlgError as exc:
        raise ValueError("GLS covariance factorization failed") from exc
    flux = adjusted * flux_scale
    posterior = (posterior_scaled + posterior_scaled.T) / 2 * flux_scale**2
    prediction = response @ flux
    residual = rates - prediction
    singular = np.linalg.svd(a_scaled, compute_uv=False)
    tolerance = max(a_scaled.shape) * np.finfo(float).eps * singular[0] if len(singular) else 0
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    if len(hold_index):
        hold = predict_holdouts(response, rates, prior, prior_cov, total_cov, hold_index)
        hold_prediction = hold.holdout_mean
        hold_residual = hold.holdout_residuals
        hold_cov = hold.holdout_covariance
        hold_chi2 = hold.holdout_standardized_chi2
    else:
        hold_prediction = np.empty(0)
        hold_residual = np.empty(0)
        hold_cov = np.empty((0, 0))
        hold_chi2 = None
    identity_bytes = json.dumps(
        [vars(row) for row in rows], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    hashes = {"row_identities": hashlib.sha256(identity_bytes).hexdigest(),
              "energy_edges": _hash(edges), "rates": _hash(rates),
              "response": _hash(response), "prior": _hash(prior),
              "prior_covariance": _hash(prior_cov),
              "observation_covariance": _hash(obs_cov),
              "response_error_covariance": _hash(response_cov)}
    return PhysicalGLSResult(
        edges.copy(), rows, tuple(ids[i] for i in fit_index), tuple(ids[i] for i in hold_index),
        prior, prior_cov, flux, posterior, obs_cov, response_cov, response, rates, prediction,
        residual, prediction[fit_index], residual[fit_index],
        innovation_chi2, postfit_chi2,
        float(len(fit_index) - np.trace(a_scaled @ gain)), rank, n - rank,
        condition, tuple(int(i) for i in np.flatnonzero(flux < 0)), hashes,
        dict(sources), source_commit, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        {"nonnegativity_constraint": False, "absolute_covariance_floor": False,
         "response_error_linearization": "frozen_at_supplied_prior",
         "rank_tolerance": float(tolerance), "physical_validation": False},
        hold_prediction, hold_residual, hold_cov, hold_chi2,
    )
