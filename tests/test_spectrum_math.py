import pytest
import numpy as np

from fluxforge.analysis.spectrum_math import (
    add_spectra,
    moving_average,
    nonnegative_counts_for_algorithm,
    subtract_measured_background,
    subtract_spectra,
)
from fluxforge.io.spe import GammaSpectrum


def test_add_subtract_spectra_alignment():
    spec_a = GammaSpectrum(
        counts=np.array([1.0, 2.0, 3.0]), channels=np.array([0, 1, 2])
    )
    spec_b = GammaSpectrum(counts=np.array([1.0, 1.0]), channels=np.array([0, 1]))

    summed = add_spectra(spec_a, spec_b)
    diffed = subtract_spectra(spec_a, spec_b)

    assert np.allclose(summed.counts, [2.0, 3.0, 3.0])
    assert np.allclose(diffed.counts, [0.0, 1.0, 3.0])


def test_moving_average_trim():
    spec = GammaSpectrum(counts=np.arange(10, dtype=float), channels=np.arange(10))
    smoothed = moving_average(spec, width=2)

    assert smoothed.counts.size == 6
    expected = np.convolve(np.arange(10, dtype=float), np.ones(5), mode="valid") / 5.0
    assert np.allclose(smoothed.counts, expected)


def test_subtract_measured_background_live_scaling_and_uncertainty():
    sample = GammaSpectrum(
        counts=np.array([10.0, 20.0, 30.0]),
        channels=np.array([0, 1, 2]),
        live_time=100.0,
        real_time=120.0,
        spectrum_id="sample",
    )
    background = GammaSpectrum(
        counts=np.array([1.0, 2.0, 3.0]),
        channels=np.array([0, 1, 2]),
        live_time=50.0,
        real_time=60.0,
        spectrum_id="background",
    )
    corrected = subtract_measured_background(sample, background, mode="live")

    assert np.allclose(corrected.counts, [8.0, 16.0, 24.0])
    expected_var = np.array([10.0, 20.0, 30.0]) + (2.0**2) * np.array([1.0, 2.0, 3.0])
    assert np.allclose(corrected.counts_uncertainty, np.sqrt(expected_var))
    assert corrected.metadata["background_subtraction"][
        "scale_factor"
    ] == pytest.approx(2.0)


def test_subtract_measured_background_real_and_manual_scaling():
    sample = GammaSpectrum(
        counts=np.array([10.0, 20.0]),
        channels=np.array([0, 1]),
        live_time=100.0,
        real_time=200.0,
    )
    background = GammaSpectrum(
        counts=np.array([2.0, 4.0]),
        channels=np.array([0, 1]),
        live_time=50.0,
        real_time=100.0,
    )

    corrected_real = subtract_measured_background(sample, background, mode="real")
    assert np.allclose(corrected_real.counts, [6.0, 12.0])
    assert corrected_real.metadata["background_subtraction"][
        "scale_factor"
    ] == pytest.approx(2.0)

    corrected_manual = subtract_measured_background(
        sample,
        background,
        mode="manual",
        manual_scale=0.5,
    )
    assert np.allclose(corrected_manual.counts, [9.0, 18.0])
    expected_var = np.array([10.0, 20.0]) + (0.5**2) * np.array([2.0, 4.0])
    assert np.allclose(corrected_manual.counts_uncertainty, np.sqrt(expected_var))


def test_subtract_measured_background_missing_warns_and_returns_raw():
    sample = GammaSpectrum(
        counts=np.array([5.0, 7.0]),
        channels=np.array([0, 1]),
        live_time=10.0,
        real_time=10.0,
    )
    with pytest.warns(RuntimeWarning, match="no background spectrum was provided"):
        corrected = subtract_measured_background(sample, None, warn_missing=True)
    assert np.allclose(corrected.counts, sample.counts)
    assert np.allclose(corrected.counts_uncertainty, sample.counts_uncertainty)


def test_subtract_measured_background_negative_policy_and_clipping_warning():
    sample = GammaSpectrum(
        counts=np.array([1.0, 1.0]),
        channels=np.array([0, 1]),
        live_time=10.0,
        real_time=10.0,
    )
    background = GammaSpectrum(
        counts=np.array([2.0, 0.0]),
        channels=np.array([0, 1]),
        live_time=10.0,
        real_time=10.0,
    )

    corrected_hybrid = subtract_measured_background(
        sample, background, negative_policy="hybrid"
    )
    assert corrected_hybrid.counts[0] < 0.0

    with pytest.warns(RuntimeWarning, match="clipping to zero"):
        corrected_clip = subtract_measured_background(
            sample, background, negative_policy="clip"
        )
    assert corrected_clip.counts[0] == 0.0


def test_subtract_measured_background_resamples_background_to_sample_energy_grid():
    sample = GammaSpectrum(
        counts=np.array([100.0, 100.0, 100.0]),
        channels=np.array([0, 1, 2]),
        energies=np.array([0.0, 1.0, 2.0]),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [0.0, 1.0]},
    )
    background = GammaSpectrum(
        counts=np.array([10.0, 20.0, 30.0]),
        channels=np.array([0, 1, 2]),
        energies=np.array([1.0, 2.0, 3.0]),
        live_time=10.0,
        real_time=10.0,
        calibration={"energy": [1.0, 1.0]},
    )

    corrected = subtract_measured_background(sample, background, mode="live")

    assert np.allclose(corrected.counts, [100.0, 90.0, 80.0])
    assert corrected.metadata["background_subtraction"]["energy_aligned"] is True


def test_nonnegative_counts_for_algorithm_warns_on_negative_bins():
    spectrum = GammaSpectrum(
        counts=np.array([4.0, -2.0, 1.0]),
        counts_uncertainty=np.array([2.0, 1.0, 1.0]),
        channels=np.array([0, 1, 2]),
    )
    with pytest.warns(RuntimeWarning, match="requires non-negative counts"):
        clipped = nonnegative_counts_for_algorithm(spectrum, algorithm_name="SNIP")
    assert np.allclose(clipped, [4.0, 0.0, 1.0])
