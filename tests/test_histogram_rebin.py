import numpy as np
import pytest
from scipy import sparse

from fluxforge.analysis.histogram_rebin import rebin_histogram


def test_split_bin_counts_and_shared_covariance_are_hand_computed():
    result = rebin_histogram([0, 2], [10], [0, 1, 2], source_variance=[4])

    np.testing.assert_allclose(result.counts, [5, 5])
    np.testing.assert_allclose(result.overlap.toarray(), [[0.5], [0.5]])
    np.testing.assert_allclose(result.covariance.toarray(), [[1, 1], [1, 1]])
    np.testing.assert_allclose(result.coverage, [1, 1])
    assert result.discarded_source_counts == pytest.approx(0)


def test_conserves_counts_across_different_complete_binning():
    result = rebin_histogram([0, 1, 3, 4], [2, 6, 4], [0, 0.5, 2, 4])

    assert result.counts.sum() == pytest.approx(12)
    assert result.discarded_source_counts == pytest.approx(0)


def test_crop_reports_signed_discarded_counts_and_partial_target_coverage():
    result = rebin_histogram(
        [0, 1, 2], [4, -2], [-1, 0.5, 1.5], source_variance=[4, 9], coverage="partial"
    )

    np.testing.assert_allclose(result.counts, [2, 1])
    np.testing.assert_allclose(result.coverage, [1 / 3, 1])
    assert result.discarded_source_counts == pytest.approx(-1)


def test_strict_rejects_partial_coverage():
    with pytest.raises(ValueError, match="strict coverage"):
        rebin_histogram([0, 1], [1], [-0.1, 0.5])


def test_signed_counts_require_explicit_uncertainty():
    with pytest.raises(ValueError, match="signed source_counts"):
        rebin_histogram([0, 1], [-1], [0, 1])


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        (([0, 0], [1], [0, 1]), {}, "strictly increasing"),
        (([0, 1], [1, 2], [0, 1]), {}, "shape"),
        (([0, 1], [1], [0, np.inf]), {}, "finite"),
        (([0, 1], [1], [0, 1]), {"source_variance": [-1]}, "nonnegative"),
        (([0, 1], [1], [0, 1]), {"coverage": "crop"}, "coverage must"),
    ],
)
def test_input_errors(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        rebin_histogram(*args, **kwargs)


def test_sparse_covariance_validation_and_propagation():
    covariance = sparse.csr_matrix([[4.0, 1.0], [1.0, 9.0]])
    result = rebin_histogram([0, 1, 2], [3, 5], [0, 2], source_covariance=covariance)
    np.testing.assert_allclose(result.covariance.toarray(), [[15]])

    with pytest.raises(ValueError, match="symmetric"):
        rebin_histogram(
            [0, 1, 2], [3, 5], [0, 2], source_covariance=sparse.csr_matrix([[1, 1], [0, 1]])
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        rebin_histogram(
            [0, 1, 2], [3, 5], [0, 2],
            source_covariance=sparse.csr_matrix([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_singular_positive_semidefinite_covariance_is_accepted():
    covariance = sparse.csr_matrix([[1.0, 1.0], [1.0, 1.0]])
    result = rebin_histogram([0, 1, 2], [2, 3], [0, 2], source_covariance=covariance)
    np.testing.assert_allclose(result.covariance.toarray(), [[4]])


def test_rejects_nonfinite_widths_and_transformed_outputs():
    with pytest.raises(ValueError, match="widths must be finite"):
        rebin_histogram([-1e308, 1e308], [1], [0, 1])
    with pytest.raises(ValueError, match="rebinned counts are nonfinite"):
        rebin_histogram([0, 1, 2], [1e308, 1e308], [0, 2])
    with pytest.raises(ValueError, match="rebinned covariance is nonfinite"):
        rebin_histogram(
            [0, 1, 2], [1, 1], [0, 2],
            source_covariance=sparse.diags([1e308, 1e308], format="csr"),
        )


def test_8192_bin_shifted_grid_has_sparse_split_bin_covariance():
    size = 8192
    edges = np.arange(size + 1, dtype=float)
    shifted_edges = np.arange(size, dtype=float) + 0.5
    result = rebin_histogram(
        edges,
        np.ones(size),
        shifted_edges,
        source_covariance=sparse.eye(size, format="csr"),
    )

    assert sparse.isspmatrix_csr(result.overlap)
    assert sparse.isspmatrix_csr(result.covariance)
    assert result.overlap.shape == (size - 1, size)
    assert result.overlap.nnz == 2 * (size - 1)
    np.testing.assert_allclose(result.covariance.diagonal(), 0.5)
    np.testing.assert_allclose(result.covariance.diagonal(1), 0.25)
    np.testing.assert_allclose(result.covariance.diagonal(-1), 0.25)
    assert result.covariance.nnz == 3 * (size - 1) - 2
