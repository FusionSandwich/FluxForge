import numpy as np
import pytest
from scipy import sparse

from fluxforge.io.spe import GammaSpectrum
from fluxforge.analysis import spectrum_math as sm


def sample_pair():
    # Source edges [0,2,4]; target edges [0.5,1.5,2.5,3.5].
    return (GammaSpectrum(counts=[20, 30, 40], energies=np.array([1., 2., 3.]), live_time=2),
            GammaSpectrum(counts=[8, 12], energies=np.array([1., 3.]), live_time=1))


def test_hand_calculated_overlap_and_subtraction():
    sample, background = sample_pair()
    before = sample.to_dict(), background.to_dict()
    aligned, changed = sm._conservative_background(sample, background)
    assert changed
    # W=[[1/2,0],[1/4,1/4],[0,1/2]], with 1/4 each source bin cropped.
    expected_bg = np.array([[2, 1, 0], [1, 1.25, 1.5], [0, 1.5, 3]])
    np.testing.assert_allclose(aligned.counts, [4, 5, 6])
    np.testing.assert_allclose(aligned.counts_covariance.toarray(), expected_bg)
    assert aligned.metadata['rebin']['discarded_source_counts'] == 5
    net = sm.subtract_measured_background(sample, background, negative_policy='preserve')
    np.testing.assert_allclose(net.counts, [12, 20, 28])
    np.testing.assert_allclose(net.counts_covariance.toarray(), np.diag([20,30,40]) + 4*expected_bg)
    assert (sample.to_dict(), background.to_dict()) == before


def test_same_grid_correlated_input_counts_covariance():
    sample = GammaSpectrum(counts=[2, 3], counts_covariance=sparse.csr_matrix([[4, 1],[1, 9]]))
    background = GammaSpectrum(counts=[1, 2])
    result = sm.subtract_measured_background(sample, background, mode='manual', manual_scale=2)
    np.testing.assert_allclose(result.counts, [0, -1])
    np.testing.assert_allclose(result.counts_covariance.toarray(), [[8, 1], [1, 17]])
    with pytest.raises(ValueError, match='Clipping'):
        sm.subtract_measured_background(sample, background, mode='manual', manual_scale=2, negative_policy='clip')


@pytest.mark.parametrize('energies', [[1,1], [2,1], [1,np.nan], [1]])
def test_invalid_edges(energies):
    with pytest.raises(ValueError, match='strictly increasing'):
        sm.spectrum_bin_edges(GammaSpectrum(counts=np.ones(len(energies)), energies=np.array(energies)))


def test_strict_coverage_rejects():
    sample, background = sample_pair()
    sample.energies += 10
    with pytest.raises(ValueError, match='strict coverage'):
        sm._conservative_background(sample, background)
