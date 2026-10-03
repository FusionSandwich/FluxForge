import numpy as np
import pytest
from fluxforge.analysis.flux_wire_analysis import _qg_adjacent_linear_continuum_counts,_qg_style_linear_continuum_counts


def test_adjacent_anchor_integrated_covariance():
    counts=np.full(13,100.0)
    sigma=np.ones(13)
    sigma[[0,12]]=10.0
    n=11
    _,reported,_,_= _qg_adjacent_linear_continuum_counts(counts,1,11,spectrum_uncertainty=sigma)
    expected=np.sqrt(n+(n/2.0)**2*(sigma[0]**2+sigma[12]**2))
    assert reported == pytest.approx(expected)


def test_roi_edge_anchor_preserves_sigma_and_shared_bin_covariance():
    counts=np.full(13,100.0)
    sigma=np.ones(13)
    sigma[[1,11]]=3.0
    n=11
    _,reported,_,_= _qg_style_linear_continuum_counts(counts,1,11,spectrum_uncertainty=sigma)
    expected=np.sqrt((n-2)+(1.0-n/2.0)**2*(sigma[1]**2+sigma[11]**2))
    assert reported == pytest.approx(expected)
