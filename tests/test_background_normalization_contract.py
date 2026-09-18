"""Measured-background scaling and energy-axis preservation contracts."""

import numpy as np
import pytest

from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.core.analysis_workspace import background_adjusted_spectrum
from fluxforge.io.spe import GammaSpectrum


@pytest.mark.parametrize("mode", ["live", "real"])
@pytest.mark.parametrize("invalid_side", ["sample", "background"])
@pytest.mark.parametrize("time", [0.0, -1.0, float("nan"), float("inf")])
def test_automatic_scaling_requires_positive_finite_selected_times(
    mode, invalid_side, time
):
    sample = GammaSpectrum(counts=[10.0, 20.0], live_time=4.0, real_time=5.0)
    background = GammaSpectrum(counts=[2.0, 3.0], live_time=2.0, real_time=2.5)
    invalid = sample if invalid_side == "sample" else background
    setattr(invalid, f"{mode}_time", time)

    with pytest.raises(ValueError, match=f"{invalid_side} {mode} time.*finite and positive"):
        subtract_measured_background(sample, background, mode=mode)
    np.testing.assert_array_equal(sample.counts, [10.0, 20.0])
    np.testing.assert_array_equal(background.counts, [2.0, 3.0])


def test_manual_scaling_does_not_require_count_times():
    sample = GammaSpectrum(counts=[1.0, 8.0], counts_uncertainty=[3.0, 4.0])
    background = GammaSpectrum(counts=[6.0, 2.0], counts_uncertainty=[2.0, 5.0])
    result = subtract_measured_background(sample, background, mode="manual", manual_scale=0.5)
    np.testing.assert_array_equal(result.counts, [-2.0, 7.0])
    np.testing.assert_allclose(result.counts_uncertainty**2, [10.0, 22.25])


def test_statistical_workspace_scaling_keeps_subsecond_ratio():
    sample = GammaSpectrum(counts=[2.0, 8.0], live_time=0.2)
    background = GammaSpectrum(counts=[2.0, 4.0], live_time=0.1)
    result = background_adjusted_spectrum(sample, background, mode="statistical")
    np.testing.assert_array_equal(result.counts, [-2.0, 0.0])
    np.testing.assert_allclose(result.counts_uncertainty**2, [10.0, 24.0])


def test_statistical_workspace_rejects_unknown_background_time():
    sample = GammaSpectrum(counts=[2.0, 8.0], live_time=2.0)
    background = GammaSpectrum(counts=[2.0, 4.0])
    with pytest.raises(ValueError, match="background live time.*finite and positive"):
        background_adjusted_spectrum(sample, background, mode="statistical")


@pytest.mark.parametrize("calibration", [{}, {"energy": [0.0, 1.0]}])
def test_subtraction_preserves_explicit_energy_axis_without_aliasing(calibration):
    energies = np.array([100.0, 100.6, 101.4])
    sample = GammaSpectrum(counts=[10.0, 20.0, 30.0], energies=energies.copy(), calibration=calibration)
    background = GammaSpectrum(counts=[2.0, 3.0, 4.0], energies=energies.copy(), calibration=calibration)
    result = subtract_measured_background(sample, background, mode="manual", manual_scale=1.0)
    np.testing.assert_array_equal(result.energies, energies)
    np.testing.assert_array_equal(result.counts, [8.0, 17.0, 26.0])
    result.energies[0] = 0.0
    np.testing.assert_array_equal(sample.energies, energies)
    np.testing.assert_array_equal(background.energies, energies)


@pytest.mark.parametrize("mode", ["live", "real"])
def test_automatic_scaling_rejects_overflow(mode):
    sample = GammaSpectrum(counts=[10.0], live_time=1e300, real_time=1e300)
    background = GammaSpectrum(counts=[2.0], live_time=1e-300, real_time=1e-300)
    with pytest.raises(ValueError, match="scale factor.*finite and positive"):
        subtract_measured_background(sample, background, mode=mode)
