"""Explicit combinations of an already-qualified gamma-line activity set.

Qualification belongs to the upstream engine: this module never identifies,
filters outliers, or admits lines. Historical A/sigma weights reproduce only
that equation, not the unavailable QuantumGold error model. Uncertainties are
absolute one-standard-deviation Bq. Reported uncertainty is sqrt(w.T C w),
conditional on fixed weights (also for data-dependent historical weights).
No nonlinear weight-estimation uncertainty or excess-scatter inflation is added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence
from numbers import Real

import numpy as np

Method = Literal["historical_activity_over_sigma", "inverse_variance", "gls"]
METHODS = ("historical_activity_over_sigma", "inverse_variance", "gls")


@dataclass(frozen=True)
class ActivityLine:
    line_id: str
    isotope: str
    activity_bq: float | None
    sigma_bq: float | None
    qualified: bool
    qualification: str
    source_identity: str
    count_basis: str
    activity_reference: str


@dataclass(frozen=True)
class CovarianceComponent:
    """Absolute BqÃƒâ€šÃ‚Â² covariance in *qualified-line order*; None is unavailable.

    Sources must describe provenance and any assumptions about correlation.
    Components are additive and must not double-count total and subcomponents.
    """

    name: str
    matrix_bq2: object | None
    source_identity: str
    definition: str


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be explicitly declared")


def _matrix(value: object, n: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name}: covariance must be real")
    c = np.asarray(value, dtype=float)
    if c.shape != (n, n) or not np.all(np.isfinite(c)):
        raise ValueError(f"{name}: covariance must be finite with shape {(n, n)}")
    if not np.allclose(c, c.T, rtol=1e-12, atol=0):
        raise ValueError(f"{name}: covariance must be symmetric")
    c = c / 2 + c.T / 2
    scale = float(np.max(np.abs(c))) if n else 0.0
    if scale:
        eig = np.linalg.eigvalsh(c / scale)
        tol = 10 * n * np.finfo(float).eps * max(1.0, float(np.max(abs(eig))))
        if float(eig.min()) < -tol:
            raise ValueError(f"{name}: covariance must be positive semidefinite")
    if np.any(np.diag(c) < 0):
        raise ValueError(f"{name}: negative variance")
    return c


def combine_activity_lines(
    lines: Sequence[ActivityLine],
    *,
    method: Method,
    uncertainty_definition: str,
    engine_identity: str,
    analysis_role: Literal[
        "historical_reproduction_control", "physical_analysis", "method_control"
    ],
    covariance_components: Sequence[CovarianceComponent] = (),
    allow_incomplete_uncertainty: bool = False,
    singular_policy: Literal["reject", "exact_constraints"] = "reject",
) -> dict:
    """Return a JSON-ready receipt; the method is required, with no new default.

    Without supplied covariance, the two weighting controls explicitly assume
    independent declared sigmas. GLS requires explicit covariance sources.
    Declare all required missing components using matrix_bq2=None. The default
    returns unavailable; opting into partial uncertainty returns conditional.
    Supplied component diagonals must reproduce the declared line sigmas.

    For GLS, singular/numerically unresolved covariance is unavailable by default.
    Fixed-weight controls can propagate a declared singular covariance without
    inversion; exact cancellation is reported explicitly. Their rank-sensitive
    diagnostics are unavailable unless exact_constraints is declared.
    With singular_policy="exact_constraints", the caller explicitly declares
    unresolved modes to be exact physical constraints. Singular PSD covariance
    is then solved using its stochastic range and exact
    null-space constraints. Incompatible exact constraints flag inconsistency.
    A positive-definite covariance with condition >1e12 is refused, with no
    ridge/floor or truncated inverse. Numerical rank uses 10*n*epsilon times
    the spectral scale and is exported; unresolved modes are not hidden.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown combination method: {method}")
    if analysis_role not in (
        "historical_reproduction_control",
        "physical_analysis",
        "method_control",
    ):
        raise ValueError("Unknown analysis_role")
    if method == METHODS[0] and analysis_role == "physical_analysis":
        raise ValueError("Historical weights are a reproduction/control option")
    if singular_policy not in ("reject", "exact_constraints"):
        raise ValueError("Unknown singular_policy")
    if type(allow_incomplete_uncertainty) is not bool:
        raise ValueError("allow_incomplete_uncertainty must be boolean")
    _text(uncertainty_definition, "uncertainty_definition")
    _text(engine_identity, "engine_identity")
    rows = tuple(lines)
    ids = [r.line_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("line_id must be unique")
    for row in rows:
        for key in (
            "line_id",
            "isotope",
            "qualification",
            "source_identity",
            "count_basis",
            "activity_reference",
        ):
            _text(getattr(row, key), key)
        if type(row.qualified) is not bool:
            raise ValueError("qualification must be an explicit boolean")
        for key in ("activity_bq", "sigma_bq"):
            value = getattr(row, key)
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    f"{row.line_id}: {key} must be a finite real scalar or unavailable"
                )
    selected = tuple(r for r in rows if r.qualified)
    if len({(r.isotope, r.count_basis, r.activity_reference) for r in selected}) > 1:
        raise ValueError(
            "Qualified lines must share isotope, count basis and activity reference"
        )
    components = tuple(covariance_components)
    if len({c.name for c in components}) != len(components):
        raise ValueError("Covariance component names must be unique")
    for c in components:
        _text(c.name, "component name")
        _text(c.source_identity, "covariance source_identity")
        _text(c.definition, "covariance definition")
    result = {
        "schema_version": 1,
        "method": method,
        "analysis_role": analysis_role,
        "engine_identity": engine_identity,
        "uncertainty_definition": uncertainty_definition,
        "uncertainty_propagation": "fixed-weight sqrt(w.T C w); no scatter inflation",
        "qualified_line_ids": [r.line_id for r in selected],
        "input_lines": [
            {
                **asdict(r),
                "activity_bq": None if r.activity_bq is None else float(r.activity_bq),
                "sigma_bq": None if r.sigma_bq is None else float(r.sigma_bq),
            }
            for r in rows
        ],
        "exclusions": [
            {"line_id": r.line_id, "reason": r.qualification}
            for r in rows
            if not r.qualified
        ],
        "covariance_sources": [],
        "unavailable_components": [],
        "status": "unavailable",
        "activity_bq": None,
        "sigma_bq": None,
        "normalized_weights": None,
        "diagnostics": None,
    }
    n = len(selected)
    missing = [c.name for c in components if c.matrix_bq2 is None]
    result["unavailable_components"] = missing
    covariance = np.zeros((n, n))
    for c in components:
        matrix = None if c.matrix_bq2 is None else _matrix(c.matrix_bq2, n, c.name)
        result["covariance_sources"].append(
            {
                "name": c.name,
                "source_identity": c.source_identity,
                "definition": c.definition,
                "availability": "unavailable" if matrix is None else "available",
                "matrix_bq2": None if matrix is None else matrix.tolist(),
            }
        )
        if matrix is not None:
            covariance += matrix
    if not selected:
        result["reason"] = "No qualified lines; qualification was preserved"
        return result
    for row in selected:
        if row.activity_bq is None or row.sigma_bq is None:
            result["reason"] = f"{row.line_id}: activity or uncertainty unavailable"
            return result
        if row.sigma_bq <= 0:
            raise ValueError(
                f"{row.line_id}: sigma_bq must be strictly positive; no uncertainty floor"
            )
    a = np.array([r.activity_bq for r in selected], dtype=float)
    sigma = np.array([r.sigma_bq for r in selected], dtype=float)
    if missing and not allow_incomplete_uncertainty:
        result["reason"] = "Required uncertainty components unavailable"
        return result
    if (method == "gls" or components) and not any(
        c.matrix_bq2 is not None for c in components
    ):
        result["reason"] = (
            "Explicit available covariance components required for the declared budget"
        )
        return result
    if not components:
        covariance = np.diag(sigma * sigma)
        result["covariance_sources"] = [
            {
                "name": "declared_independent_line_sigmas",
                "availability": "available",
                "source_identity": "input line sources",
                "definition": "independent declared sigmas; correlation unavailable",
                "matrix_bq2": covariance.tolist(),
            }
        ]
    covariance = _matrix(covariance, n, "total")
    if not np.allclose(np.sqrt(np.diag(covariance)), sigma, rtol=1e-10, atol=0):
        raise ValueError(
            "Covariance diagonal must match declared sigmas (partial sigmas if conditional)"
        )
    scale = float(np.max(np.abs(covariance)))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance / scale)
    rank_tol = 10 * n * np.finfo(float).eps * max(1.0, float(eigenvalues.max()))
    positive = eigenvalues > rank_tol
    null = eigenvectors[:, ~positive]
    one = np.ones(n)
    condition = (
        float(eigenvalues.max() / eigenvalues.min()) if np.all(positive) else None
    )
    solver = {
        "covariance_rank": int(positive.sum()),
        "scaled_eigenvalues": eigenvalues.tolist(),
        "relative_rank_tolerance": rank_tol,
        "condition_number": condition,
        "weight_clipping": False,
        "regularization": None,
        "singular_policy": singular_policy,
    }
    if method == "gls" and not np.all(positive) and singular_policy == "reject":
        result["reason"] = (
            "Singular or numerically unresolved covariance; exact_constraints requires a declared model"
        )
        result["diagnostics"] = solver
        return result
    if method == "gls" and condition is not None and condition > 1e12:
        result["reason"] = (
            "Near-singular covariance: condition exceeds 1e12; supply a resolved model"
        )
        result["diagnostics"] = solver
        return result
    if method == METHODS[0]:
        if np.any(a <= 0):
            raise ValueError(
                "Historical A/sigma weighting requires strictly positive activities"
            )
        log_w = np.log(a) - np.log(sigma)
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
    elif method == "inverse_variance":
        log_w = -2 * np.log(sigma)
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
    else:
        projected_one = null @ (null.T @ one)
        norm = float(one @ projected_one)
        if norm > 10 * n * np.finfo(float).eps:
            w = projected_one / norm
        else:
            basis = eigenvectors[:, positive]
            solved = basis @ ((basis.T @ one) / eigenvalues[positive])
            w = solved / (one @ solved)
    centered = a - a[0]
    offset = float(w @ centered)
    estimate = float(a[0] + offset)
    if (
        not np.all(np.isfinite(w))
        or not np.all(np.isfinite(centered))
        or not np.isfinite(estimate)
    ):
        raise ValueError("Combination exceeds floating point range")
    variance = float(w @ covariance @ w)
    # Roundoff in a PSD quadratic form can be negative at machine resolution.
    if not np.isfinite(variance):
        raise ValueError("Propagated variance exceeds floating point range")
    if variance < -rank_tol * scale * float(w @ w):
        raise ValueError("Negative propagated variance")
    variance = max(0.0, variance)
    residual = centered - offset
    transform = np.eye(n) - np.outer(one, w)
    residual_cov = transform @ covariance @ transform.T
    residual_sigma = np.sqrt(np.maximum(0, np.diag(residual_cov)))
    null_residual = null.T @ residual
    exact_tolerance = 10 * n * np.finfo(float).eps * float(np.max(abs(centered)))
    exact_model = singular_policy == "exact_constraints"
    incompatible = bool(exact_model and np.linalg.norm(null_residual) > exact_tolerance)
    basis = eigenvectors[:, positive]
    rank_diagnostics_available = bool(np.all(positive) or exact_model)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        stochastic_z = (basis.T @ residual) / (
            np.sqrt(eigenvalues[positive]) * np.sqrt(scale)
        )
        chi2 = (
            float(stochastic_z @ stochastic_z) if rank_diagnostics_available else None
        )
    if not np.all(np.isfinite(residual_cov)) or (
        chi2 is not None and not np.isfinite(chi2)
    ):
        raise ValueError("Residual diagnostics exceed floating point range")
    result.update(
        {
            "status": (
                "inconsistent"
                if incompatible
                else ("conditional" if missing else "available")
            ),
            "activity_bq": estimate,
            "sigma_bq": float(np.sqrt(variance)),
            "normalized_weights": w.tolist(),
            "covariance_bq2": covariance.tolist(),
            "diagnostics": {
                **solver,
                "negative_weight_line_ids": [
                    selected[i].line_id for i in range(n) if w[i] < 0
                ],
                "residuals_bq": residual.tolist(),
                "residual_sigmas_bq": residual_sigma.tolist(),
                "standardized_residuals": [
                    (
                        float(residual[i] / residual_sigma[i])
                        if residual_sigma[i] > 0
                        else None
                    )
                    for i in range(n)
                ],
                "chi_square": chi2,
                "chi_square_definition": "residual.T C+ residual on stochastic range",
                "chi_square_dof": (
                    (
                        int(positive.sum())
                        - (
                            0
                            if np.linalg.norm(null.T @ one)
                            > np.sqrt(10 * n * np.finfo(float).eps)
                            else 1
                        )
                    )
                    if rank_diagnostics_available
                    else None
                ),
                "rank_diagnostics_available": rank_diagnostics_available,
                "residual_definition": "centered line activities minus weighted centered mean",
                "exact_cancellation": variance == 0.0,
                "chi_square_interpretation": "GLS goodness of fit only; other weights are descriptive",
                "null_constraint_residuals_bq": null_residual.tolist(),
                "null_constraint_tolerance_bq": exact_tolerance,
                "incompatible_exact_constraints": incompatible,
                "line_inconsistency_flags": [
                    bool(abs(residual[i]) > max(3 * residual_sigma[i], exact_tolerance))
                    for i in range(n)
                ],
            },
        }
    )
    return result
