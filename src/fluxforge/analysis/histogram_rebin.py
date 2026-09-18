"""Conservative rebinning for one-dimensional histogram spectra.

Counts are assumed to be uniformly distributed within each source bin.  No
extrapolation is performed outside the source-edge interval.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh


@dataclass(frozen=True)
class HistogramRebinResult:
    """Result of :func:`rebin_histogram`."""

    counts: np.ndarray
    covariance: sparse.csr_matrix
    overlap_matrix: sparse.csr_matrix
    coverage_fractions: np.ndarray
    discarded_source_counts: float

    @property
    def overlap(self):
        """Alias for :attr:`overlap_matrix`."""
        return self.overlap_matrix

    @property
    def coverage(self):
        """Alias for :attr:`coverage_fractions`."""
        return self.coverage_fractions


def _edges(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two edges")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore"):
        widths = np.diff(result)
    if not np.all(np.isfinite(widths)):
        raise ValueError(f"{name} bin widths must be finite")
    if np.any(widths <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _vector(values, size, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_covariance(value, size):
    if not sparse.issparse(value):
        raise TypeError("source_covariance must be a scipy sparse matrix")
    covariance = sparse.csr_matrix(value, dtype=float)
    if covariance.shape != (size, size):
        raise ValueError(f"source_covariance must have shape ({size}, {size})")
    if covariance.data.size and not np.all(np.isfinite(covariance.data)):
        raise ValueError("source_covariance must contain only finite values")

    difference = covariance - covariance.T
    scale = max(1.0, float(np.max(np.abs(covariance.data), initial=0.0)))
    if difference.data.size and np.max(np.abs(difference.data)) > 1e-12 * scale:
        raise ValueError("source_covariance must be symmetric")
    covariance = (covariance * 0.5 + covariance.T * 0.5).tocsr()

    diagonal = covariance.diagonal()
    off_diagonal = covariance - sparse.diags(diagonal, format="csr")
    off_diagonal.eliminate_zeros()
    if off_diagonal.nnz == 0:
        minimum = float(np.min(diagonal, initial=0.0)) / scale
    elif covariance.nnz == 0:
        minimum = 0.0
    else:
        try:
            minimum = float(
                eigsh(covariance / scale, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0]
            )
        except (ArpackError, ArpackNoConvergence) as exc:
            raise ValueError("could not verify that source_covariance is positive semidefinite") from exc
    if minimum < -1e-10:
        raise ValueError("source_covariance must be positive semidefinite")
    return covariance


def _overlap_matrix(source_edges, target_edges):
    source_widths = np.diff(source_edges)
    rows = []
    columns = []
    values = []
    source_index = 0
    for target_index, (left, right) in enumerate(zip(target_edges[:-1], target_edges[1:])):
        while source_index < source_widths.size and source_edges[source_index + 1] <= left:
            source_index += 1
        index = source_index
        while index < source_widths.size and source_edges[index] < right:
            overlap = min(right, source_edges[index + 1]) - max(left, source_edges[index])
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(index)
                values.append(overlap / source_widths[index])
            index += 1
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(target_edges.size - 1, source_edges.size - 1),
        dtype=float,
    )


def rebin_histogram(
    source_edges,
    source_counts,
    target_edges,
    source_variance=None,
    source_covariance=None,
    coverage="strict",
):
    """Rebin counts using geometric overlaps and a constant within-bin density.

    ``coverage='strict'`` requires every target bin to be fully covered by the
    source interval.  ``coverage='partial'`` permits cropped target bins and
    reports their covered fractions; uncovered portions always contribute zero.
    """

    source_edges = _edges(source_edges, "source_edges")
    target_edges = _edges(target_edges, "target_edges")
    source_counts = _vector(source_counts, source_edges.size - 1, "source_counts")
    if coverage not in {"strict", "partial"}:
        raise ValueError("coverage must be 'strict' or 'partial'")
    if source_variance is not None and source_covariance is not None:
        raise ValueError("provide only one of source_variance and source_covariance")

    overlap = _overlap_matrix(source_edges, target_edges)
    target_widths = np.diff(target_edges)
    covered_lengths = np.asarray(overlap @ np.diff(source_edges)).ravel()
    coverage_fractions = np.clip(covered_lengths / target_widths, 0.0, 1.0)
    if coverage == "strict" and np.any(coverage_fractions < 1.0 - 1e-12):
        raise ValueError("strict coverage requires every target bin to be fully covered")

    if source_covariance is not None:
        input_covariance = _validate_covariance(source_covariance, source_counts.size)
    else:
        if source_variance is None:
            if np.any(source_counts < 0.0):
                raise ValueError("signed source_counts require explicit variance or covariance")
            variance = source_counts
        else:
            variance = _vector(source_variance, source_counts.size, "source_variance")
            if np.any(variance < 0.0):
                raise ValueError("source_variance must be nonnegative")
        input_covariance = sparse.diags(variance, format="csr")

    with np.errstate(over="ignore", invalid="ignore"):
        rebinned_counts = np.asarray(overlap @ source_counts).ravel()
        rebinned_covariance = (overlap @ input_covariance @ overlap.T).tocsr()
    if not np.all(np.isfinite(rebinned_counts)):
        raise ValueError("rebinned counts are nonfinite")
    if rebinned_covariance.data.size and not np.all(np.isfinite(rebinned_covariance.data)):
        raise ValueError("rebinned covariance is nonfinite")
    retained_fractions = np.asarray(overlap.sum(axis=0)).ravel()
    with np.errstate(over="ignore", invalid="ignore"):
        discarded = float(source_counts @ (1.0 - retained_fractions))
    if not np.isfinite(discarded):
        raise ValueError("discarded source counts are nonfinite")
    return HistogramRebinResult(
        counts=rebinned_counts,
        covariance=rebinned_covariance,
        overlap_matrix=overlap,
        coverage_fractions=coverage_fractions,
        discarded_source_counts=discarded,
    )
