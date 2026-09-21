"""Sparse count covariance validation and lossless JSON representation."""

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, eig_banded
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh


CSR_SCHEMA = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["format", "shape", "data", "indices", "indptr"],
    "properties": {
        "format": {"const": "csr"},
        "shape": {"type": "array", "minItems": 2, "maxItems": 2,
                  "items": {"type": "integer", "minimum": 0}},
        "data": {"type": "array", "items": {"type": "number"}},
        "indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "indptr": {"type": "array", "items": {"type": "integer", "minimum": 0}},
    },
}


def validate_count_covariance(value, size):
    if not sparse.issparse(value):
        raise TypeError("counts_covariance must be a scipy sparse matrix")
    result = sparse.csr_matrix(value, dtype=float, copy=True)
    if result.shape != (size, size):
        raise ValueError(f"counts_covariance must have shape ({size}, {size})")
    result.check_format(full_check=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.all(np.isfinite(result.data)):
        raise ValueError("counts_covariance must be finite")
    scale = max(1.0, float(np.max(np.abs(result.data), initial=0)))
    difference = result - result.T
    if np.max(np.abs(difference.data), initial=0) > 1e-12 * scale:
        raise ValueError("counts_covariance must be symmetric")
    result = ((result + result.T) * 0.5).tocsr()
    diagonal = result.diagonal()
    if np.any(diagonal < 0):
        raise ValueError("counts_covariance diagonal must be nonnegative")
    # Gershgorin gives an inexpensive sufficient PSD certificate for the common
    # diagonally dominant 8192-channel counting covariance.
    radius = np.asarray(abs(result).sum(axis=1)).ravel() - diagonal
    if np.any(diagonal < radius):
        rows, columns = result.nonzero()
        bandwidth = int(np.max(np.abs(rows - columns), initial=0))
        if size <= 128:
            minimum = np.linalg.eigvalsh(result.toarray() / scale)[0]
        elif bandwidth <= 32:
            # Overlap rebinning of similar channel widths is narrowly banded.
            # A selected banded eigenvalue handles many exact zero bins without
            # slow iterative convergence or allocating an N by N dense matrix.
            band = np.zeros((bandwidth + 1, size))
            for offset in range(bandwidth + 1):
                band[offset, :size-offset] = result.diagonal(-offset) / scale
            # Exact zero-variance bins must have zero covariance with all bins.
            positive = diagonal > 0
            if np.any(radius[~positive] > 0):
                raise ValueError("counts_covariance must be positive semidefinite")
            reduced = result[positive][:, positive]
            compact = np.zeros((bandwidth + 1, reduced.shape[0]))
            for offset in range(min(bandwidth + 1, reduced.shape[0])):
                compact[offset, :reduced.shape[0]-offset] = reduced.diagonal(-offset) / scale
            try:
                cholesky_banded(compact, lower=True)
                minimum = 0.0  # Positive-definite principal block; zero rows are exact.
            except np.linalg.LinAlgError:
                minimum = eig_banded(band, lower=True, eigvals_only=True,
                                     select="i", select_range=(0, 0))[0]
        else:
            try:
                minimum = eigsh(result / scale, k=1, which="SA", tol=1e-8,
                                v0=np.ones(size), maxiter=5000, return_eigenvectors=False)[0]
            except (ArpackError, ArpackNoConvergence) as exc:
                raise ValueError("could not verify counts_covariance positive semidefiniteness") from exc
        if minimum < -1e-10:
            raise ValueError("counts_covariance must be positive semidefinite")
    return result


def covariance_from_payload(payload, size):
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {"format", "shape", "data", "indices", "indptr"}:
        raise ValueError("counts_covariance requires a complete CSR object")
    if payload["format"] != "csr" or payload["shape"] != [size, size]:
        raise ValueError("counts_covariance format or shape is invalid")
    data = np.asarray(payload["data"], dtype=float)
    indices = np.asarray(payload["indices"])
    indptr = np.asarray(payload["indptr"])
    for name, array in (("indices", indices), ("indptr", indptr)):
        if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
            raise ValueError(f"counts_covariance {name} must contain integers")
    if (data.ndim != 1 or data.size != indices.size or indptr.size != size + 1
            or indptr[0] != 0 or indptr[-1] != data.size or np.any(np.diff(indptr) < 0)
            or np.any(indices < 0) or np.any(indices >= size)):
        raise ValueError("counts_covariance CSR structure is invalid")
    result = sparse.csr_matrix((data, indices.astype(int), indptr.astype(int)), shape=(size, size))
    if not result.has_canonical_format:
        raise ValueError("counts_covariance CSR indices must be sorted and unique")
    return validate_count_covariance(result, size)


def covariance_to_payload(covariance):
    if covariance is None:
        return None
    result = validate_count_covariance(covariance, covariance.shape[0])
    return {"format": "csr", "shape": list(result.shape), "data": result.data.tolist(),
            "indices": result.indices.tolist(), "indptr": result.indptr.tolist()}


def linear_variance(weights, uncertainty, covariance=None):
    weights = np.asarray(weights, dtype=float)
    variance = (float(weights @ (covariance @ weights)) if covariance is not None
                else float(np.sum((weights * uncertainty) ** 2)))
    if not np.isfinite(variance) or variance < -1e-10:
        raise ValueError("linear estimate has invalid variance")
    return max(variance, 0.0)
