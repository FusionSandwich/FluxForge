"""Direct GammaSpectrum uncertainty and covariance validation contracts."""

import numpy as np
import pytest
from scipy import sparse

from fluxforge.io.spe import GammaSpectrum


def test_direct_spectrum_rejects_signed_counts_without_uncertainty():
    with pytest.raises(ValueError, match="Signed counts require"):
        GammaSpectrum(counts=[-1.0, 2.0])


@pytest.mark.parametrize("uncertainty", [[-1.0, 2.0], [1.0, np.nan], [1.0, np.inf]])
def test_direct_spectrum_rejects_invalid_uncertainty(uncertainty):
    with pytest.raises(ValueError, match="finite and non-negative"):
        GammaSpectrum(counts=[1.0, 2.0], counts_uncertainty=uncertainty)


def test_covariance_diagonal_comparison_rejects_shape_before_broadcasting():
    covariance = sparse.diags([1.0, 4.0], format="csr")
    with pytest.raises(ValueError, match="same shape as counts"):
        GammaSpectrum(
            counts=[1.0, 2.0],
            counts_uncertainty=[1.0],
            counts_covariance=covariance,
        )


def test_covariance_diagonal_comparison_rejects_invalid_uncertainty():
    covariance = sparse.diags([1.0, 4.0], format="csr")
    with pytest.raises(ValueError, match="finite and non-negative"):
        GammaSpectrum(
            counts=[1.0, 2.0],
            counts_uncertainty=[1.0, np.nan],
            counts_covariance=covariance,
        )
