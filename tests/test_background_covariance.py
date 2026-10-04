"""Counterexamples for shared counts, signed sidebands, and serialization."""

import json

import numpy as np
import pytest

from fluxforge.analysis.flux_wire_analysis import (
    estimate_peak_area,
    estimate_peak_area_local_background,
)
from fluxforge.analysis.spectrum_math import (
    add_spectra,
    moving_average,
    subtract_measured_background,
    subtract_spectra,
)
from fluxforge.io.spe import GammaSpectrum


def fractional_grid_pair():
    sample = GammaSpectrum(counts=[10, 20], energies=np.array([0.5, 1.5]))
    background = GammaSpectrum(counts=[4, 9, 16], energies=np.array([0, 1, 2]))
    return sample, background


def test_shared_interpolation_counts_raise_sum_variance_without_changing_counts():
    sample, background = fractional_grid_pair()
    corrected = subtract_measured_background(
        sample, background, mode="manual", manual_scale=2
    )
    np.testing.assert_allclose(corrected.counts, [-3, -5])
    # W=[[.5,.5,0],[0,.5,.5]]: source channel 1 contributes to both outputs.
    expected = np.array([[23, 9], [9, 45]])
    np.testing.assert_allclose(corrected.counts_covariance.toarray(), expected)
    assert corrected.weighted_counts_variance(np.ones(2)) == pytest.approx(86)
    # Diagonal-only propagation gives 68 and would pass a per-bin-only test.
    assert np.sum(corrected.counts_uncertainty**2) == pytest.approx(68)
    assert corrected.counts_in_range(0, 1, use_energy=False)[1] == pytest.approx(
        np.sqrt(86)
    )
    _, uncertainty, _ = estimate_peak_area(
        corrected.counts,
        0,
        np.zeros(2),
        fwhm_channels=1,
        spectrum_data=corrected,
    )
    assert uncertainty == pytest.approx(np.sqrt(86))


def test_sideband_cross_terms_are_included_with_their_negative_sign():
    # Shared-mode covariance is cancelled by a continuum-subtracted count sum.
    covariance = np.eye(7) * 4 + np.ones((7, 7)) * 9
    spectrum = GammaSpectrum(
        counts=[10, 10, 20, 40, 20, 10, 10], counts_covariance=covariance
    )
    net, uncertainty, gross, background, bounds = estimate_peak_area_local_background(
        spectrum.counts,
        3,
        1,
        roi_width_fwhm=2,
        spectrum_data=spectrum,
    )
    assert (net, gross, background, bounds) == (50, 80, 30, (2, 4))
    # Net weights [0,-1.5,1,1,1,-1.5,0], sum=0; variance=4*(3+2*2.25)=30.
    assert uncertainty**2 == pytest.approx(30)
    # Dropping covariance, or summing ROI and sideband variances separately,
    # would incorrectly retain the shared background mode.
    assert uncertainty**2 != pytest.approx(13 * 7.5)


def test_covariance_serialization_and_missing_background_preserve_off_diagonal():
    sample, background = fractional_grid_pair()
    corrected = subtract_measured_background(
        sample, background, mode="manual", manual_scale=2
    )
    restored = GammaSpectrum.from_dict(json.loads(json.dumps(corrected.to_dict())))
    clone = subtract_measured_background(restored, None, warn_missing=False)
    np.testing.assert_allclose(clone.counts_covariance.toarray(), [[23, 9], [9, 45]])
    assert clone.weighted_counts_variance(np.ones(2)) == pytest.approx(86)
    legacy = corrected.to_dict()
    legacy.pop("counts_covariance")
    assert GammaSpectrum.from_dict(legacy).weighted_counts_variance(
        np.ones(2)
    ) == pytest.approx(68)


@pytest.mark.parametrize("operation", [add_spectra, subtract_spectra])
def test_arithmetic_retains_covariance_and_pads_shorter_spectrum(operation):
    left = GammaSpectrum(counts=[5, 6], counts_covariance=[[4, 2], [2, 9]])
    right = GammaSpectrum(counts=[1])
    result = operation(left, right)
    np.testing.assert_allclose(result.counts_covariance.toarray(), [[5, 2], [2, 9]])
    assert result.weighted_counts_variance(np.ones(2)) == pytest.approx(18)


def test_smoothing_propagates_overlapping_window_covariance():
    sample = GammaSpectrum(counts=np.ones(4) * 9)
    result = moving_average(sample, width=1)
    np.testing.assert_allclose(result.counts_covariance.toarray(), [[3, 2], [2, 3]])
    assert result.weighted_counts_variance(np.ones(2)) == pytest.approx(10)


@pytest.mark.parametrize(
    "covariance",
    [
        [[1]],
        [[1, 2], [0, 1]],
        [[-1, 0], [0, 1]],
        [[np.nan, 0], [0, 1]],
    ],
)
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError, match="counts_covariance"):
        GammaSpectrum(counts=[1, 2], counts_covariance=covariance)


def test_inconsistent_covariance_diagonal_is_rejected():
    with pytest.raises(ValueError, match="diagonal must match"):
        GammaSpectrum(
            counts=[1, 2], counts_uncertainty=[1, 1], counts_covariance=np.eye(2) * 2
        )


def test_negative_weighted_variance_is_not_reported_as_zero():
    spectrum = GammaSpectrum(counts=[1, 1], counts_covariance=[[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="negative variance"):
        spectrum.weighted_counts_variance(np.array([1, -1]))


def test_covariance_clone_does_not_alias_source_matrix():
    from scipy.sparse import csr_matrix

    covariance = csr_matrix([[4.0, 2.0], [2.0, 9.0]])
    sample = GammaSpectrum(counts=[5, 6], counts_covariance=covariance)
    clone = subtract_measured_background(sample, None, warn_missing=False)
    clone.counts_covariance.data[0] = 100
    np.testing.assert_allclose(sample.counts_covariance.toarray(), [[4, 2], [2, 9]])
    sample.counts_covariance.data[0] = 200
    np.testing.assert_allclose(covariance.toarray(), [[4, 2], [2, 9]])


@pytest.mark.parametrize("uncertainty", [[-1, 1], [np.nan, 1], [np.inf, 1]])
def test_invalid_count_uncertainty_is_rejected(uncertainty):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GammaSpectrum(counts=[5, 6], counts_uncertainty=uncertainty)


def test_covariance_cannot_be_taken_from_a_different_count_array():
    data = GammaSpectrum(counts=[1, 2])
    with pytest.raises(ValueError, match="counts must match"):
        estimate_peak_area([2, 3], 0, np.zeros(2), spectrum_data=data)
    with pytest.raises(ValueError, match="counts must match"):
        estimate_peak_area_local_background([2, 3], 0, 1, spectrum_data=data)
