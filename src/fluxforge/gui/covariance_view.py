"""Validated covariance presentation without inferring missing uncertainty."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class CovarianceView:
    covariance: np.ndarray
    correlation: np.ndarray
    labels: tuple[str, ...]
    title: str = "Covariance"
    source: str = ""


def prepare_covariance_view(covariance, labels=None, *, title="Covariance", source=""):
    """Validate a covariance and preserve undefined zero-variance correlations.

    Validation uses standardized coordinates so small physical units do not
    hide indefiniteness. Singular positive-semidefinite matrices are supported.
    The supplied covariance is neither regularized nor replaced by a diagonal.
    """
    if covariance is None:
        raise ValueError(
            "Covariance is unavailable; supply a measured or declared matrix"
        )
    matrix = np.array(covariance, dtype=float, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("Covariance must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Covariance must contain only finite values")
    diagonal = matrix.diagonal()
    if (diagonal < 0).any():
        raise ValueError("Covariance has a negative variance")
    positive = diagonal > 0
    if np.any(matrix[~positive] != 0) or np.any(matrix[:, ~positive] != 0):
        raise ValueError("Zero-variance entries must have zero covariance")
    correlation = np.full(matrix.shape, np.nan)
    if positive.any():
        sigma = np.sqrt(diagonal[positive])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (
                matrix[np.ix_(positive, positive)] / sigma[:, None] / sigma[None, :]
            )
        if not np.isfinite(standardized).all():
            raise ValueError("Covariance cannot be standardized to finite correlation")
        if not np.allclose(standardized, standardized.T, rtol=1e-10, atol=1e-12):
            raise ValueError("Covariance must be symmetric")
        if np.any(np.abs(standardized) > 1 + 1e-12):
            raise ValueError("Covariance must be positive semidefinite")
        eigenvalues = np.linalg.eigvalsh((standardized + standardized.T) / 2)
        tolerance = 64 * np.finfo(float).eps * max(len(sigma), 1)
        if eigenvalues.min() < -tolerance:
            raise ValueError("Covariance must be positive semidefinite")
        correlation[np.ix_(positive, positive)] = standardized
    n = len(matrix)
    if labels is None:
        names = tuple(str(i + 1) for i in range(n))
    else:
        if isinstance(labels, (str, bytes)):
            raise ValueError("Matrix labels must be a sequence of names")
        names = tuple(str(label).strip() for label in labels)
        if len(names) != n or any(not label for label in names):
            raise ValueError(
                "Matrix labels must match the matrix dimension and be nonempty"
            )
    matrix.setflags(write=False)
    correlation.setflags(write=False)
    return CovarianceView(matrix, correlation, names, str(title), str(source))


def read_covariance_view(path: str | Path) -> CovarianceView:
    """Read a declared matrix and its identities from a JSON artifact."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("A covariance artifact must be a JSON object")
    present = [
        key
        for key in ("covariance", "measurement_covariance", "rate_covariance")
        if key in payload
    ]
    if len(present) != 1:
        raise ValueError(
            "Declare exactly one covariance, measurement_covariance or rate_covariance matrix"
        )
    matrix = payload[present[0]]
    if matrix is None:
        diagnostics = payload.get("diagnostics")
        reason = (
            diagnostics.get("uncertainty_unavailable_reason", "No matrix was supplied")
            if isinstance(diagnostics, dict)
            else "No matrix was supplied"
        )
        raise ValueError(f"Covariance is unavailable: {reason}")
    labels = next(
        (
            payload[key]
            for key in (
                "labels",
                "observation_labels",
                "parameter_names",
                "covariance_parameters",
                "measurement_labels",
            )
            if key in payload
        ),
        None,
    )
    if labels is None and "rates" in payload:
        if not isinstance(payload["rates"], list) or any(
            not isinstance(row, dict) for row in payload["rates"]
        ):
            raise ValueError("Rate identities must be a list of row objects")
        labels = [
            row.get("observation_id", row.get("reaction_id", ""))
            for row in payload["rates"]
        ]
    return prepare_covariance_view(
        matrix,
        labels,
        title=payload.get("title", present[0].replace("_", " ").title()),
        source=str(source),
    )
