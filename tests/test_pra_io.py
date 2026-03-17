import numpy as np
from pathlib import Path

from fluxforge.io.pra import read_pra_histogram, pra_to_gamma_spectrum, read_pra_as_spectrum

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_read_pra_histogram():
    path = TEST_DATA_DIR / "pygammaspec" / "utils" / "background.txt"
    hist = read_pra_histogram(path, live_time_s=25851)

    assert hist.channels.size > 10
    assert hist.counts.size == hist.channels.size
    assert hist.channels[0] == 0.0
    assert hist.counts[0] == 0.0


def test_pra_to_spectrum_roundtrip():
    path = TEST_DATA_DIR / "pygammaspec" / "utils" / "background.txt"
    hist = read_pra_histogram(path, live_time_s=25851)
    spectrum = pra_to_gamma_spectrum(hist, spectrum_id="background")

    assert spectrum.spectrum_id == "background"
    assert np.allclose(spectrum.channels, hist.channels)
    assert np.allclose(spectrum.counts, hist.counts)
    assert spectrum.live_time == 25851


def test_read_pra_as_spectrum():
    path = TEST_DATA_DIR / "pygammaspec" / "utils" / "background.txt"
    spectrum = read_pra_as_spectrum(path, live_time_s=25851)

    assert spectrum.counts.size > 10
    assert spectrum.live_time == 25851
