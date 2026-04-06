import numpy as np

from fluxforge.analysis.spectroscopy_tools import (
    prominence_peaks,
    fit_gaussian_baseline,
)


def test_prominence_peaks_simple():
    channels = np.arange(10)
    counts = np.array([0, 1, 0, 5, 0, 1, 0, 4, 0, 0], dtype=float)
    peaks = prominence_peaks(channels, counts, prominence=1.0)

    assert len(peaks) >= 2


def test_fit_gaussian_baseline_recovers_centroid():
    energies = np.linspace(0.0, 10.0, 200)
    centroid = 6.5
    sigma = 0.4
    amplitude = 100.0
    baseline = 2.0 + 0.1 * energies
    counts = amplitude * np.exp(-0.5 * ((energies - centroid) / sigma) ** 2) + baseline

    result = fit_gaussian_baseline(
        energies, counts, e_min=5.0, e_max=8.0, baseline_order=1
    )
    assert abs(result.parameters[1] - centroid) < 0.2
