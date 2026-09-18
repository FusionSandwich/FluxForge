"""Focused independent oracles for background, ROI, and joint-fit corrections."""

import numpy as np
import pytest

from fluxforge.analysis.peakfit import fit_multiple_peaks
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.core.analysis_workspace import (
    analyze_roi_region,
    background_adjusted_spectrum,
)
from fluxforge.io.spe import GammaSpectrum


def test_gui_background_preserves_signed_counts_and_poisson_variance():
    foreground = GammaSpectrum(
        counts=np.array([1.0, 10.0, 20.0]),
        counts_uncertainty=np.array([2.0, 3.0, 4.0]),
        live_time=100.0,
    )
    background = GammaSpectrum(
        counts=np.array([3.0, 2.0, 1.0]),
        counts_uncertainty=np.array([1.0, 2.0, 3.0]),
        live_time=50.0,
    )

    result = background_adjusted_spectrum(
        foreground, background, mode="statistical"
    )

    np.testing.assert_array_equal(result.counts, [-5.0, 6.0, 18.0])
    np.testing.assert_allclose(result.counts_uncertainty**2, [8.0, 25.0, 52.0])
    np.testing.assert_array_equal(foreground.counts, [1.0, 10.0, 20.0])


@pytest.mark.parametrize(
    "sample,background",
    [
        (
            GammaSpectrum(counts=np.ones(4)),
            GammaSpectrum(counts=np.ones(2)),
        ),
        (
            GammaSpectrum(
                counts=np.ones(2), calibration={"energy": [1.0, 2.0]}
            ),
            GammaSpectrum(
                counts=np.ones(4), calibration={"energy": [0.5, 1.0]}
            ),
        ),
        (
            GammaSpectrum(
                counts=np.ones(3), calibration={"energy": [0.0, 1.0]}
            ),
            GammaSpectrum(
                counts=np.ones(3), calibration={"energy": [1.0, 1.0]}
            ),
        ),
    ],
)
def test_background_rejects_grids_that_need_padding_or_rebinning(
    sample, background
):
    with pytest.raises(ValueError, match="identical .*grid"):
        subtract_measured_background(sample, background)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -1.0])
def test_manual_background_scale_must_be_finite_and_nonnegative(scale):
    spectrum = GammaSpectrum(counts=np.ones(4))
    with pytest.raises(ValueError, match="finite and non-negative"):
        subtract_measured_background(
            spectrum, spectrum, mode="manual", manual_scale=scale
        )


def test_roi_sideband_variance_uses_full_linear_estimator_weights():
    spectrum = GammaSpectrum(
        counts=np.full(100, 10.0), calibration={"energy": [0.0, 1.0]}
    )

    result = analyze_roi_region(
        spectrum, roi_bounds_keV=(40.0, 50.0), sideband_width_keV=5.0
    )

    roi_weight = np.zeros(100)
    roi_weight[40:51] = 1.0
    background_weight = np.zeros(100)
    background_weight[35:41] += 5.5 / 6.0
    background_weight[50:56] += 5.5 / 6.0
    independent_variance = np.sum((roi_weight - background_weight) ** 2 * 10.0)
    assert result.net_counts == pytest.approx(0.0)
    assert result.net_counts_uncertainty**2 == pytest.approx(
        independent_variance
    )


@pytest.mark.parametrize("background_model", ["linear", "constant"])
def test_shared_sigma_doublet_uses_the_sigma_parameter_for_each_background(
    background_model,
):
    channels = np.arange(200, dtype=float)
    counts = (
        10.0
        + 500.0 * np.exp(-0.5 * ((channels - 90.0) / 3.0) ** 2)
        + 300.0 * np.exp(-0.5 * ((channels - 101.0) / 3.0) ** 2)
    )

    fitted = fit_multiple_peaks(
        channels,
        counts,
        [90, 101],
        fit_width=18,
        background_model=background_model,
        share_sigma=True,
    )
    fitted = sorted(fitted, key=lambda item: item.peak.centroid)

    np.testing.assert_allclose(
        [item.peak.centroid for item in fitted], [90.0, 101.0], atol=0.1
    )
    np.testing.assert_allclose(
        [item.peak.area for item in fitted],
        np.array([500.0, 300.0]) * 3.0 * np.sqrt(2.0 * np.pi),
        rtol=0.01,
    )
