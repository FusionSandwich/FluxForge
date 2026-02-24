import numpy as np

from fluxforge.analysis.spectrum_math import add_spectra, subtract_spectra, moving_average
from fluxforge.io.spe import GammaSpectrum


def test_add_subtract_spectra_alignment():
    spec_a = GammaSpectrum(counts=np.array([1.0, 2.0, 3.0]), channels=np.array([0, 1, 2]))
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
