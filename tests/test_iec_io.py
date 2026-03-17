from pathlib import Path

import numpy as np
import pytest

from fluxforge.io.iec import read_iec_file
from fluxforge.io.hpge import read_hpge_spectrum

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_read_iec_file_parses_header():
    path = TEST_DATA_DIR / "spectrum_io" / "samples" / "hpge_dummy_test_01.iec"
    spectrum = read_iec_file(path)

    assert spectrum.counts.size == 2048
    assert np.isclose(spectrum.live_time, 3564.0, rtol=0, atol=0.1)
    assert np.isclose(spectrum.real_time, 3600.0, rtol=0, atol=0.1)
    assert spectrum.calibration["energy"][0] == pytest.approx(-1.55656e-2, rel=0, abs=1e-6)
    assert spectrum.calibration["energy"][1] == pytest.approx(0.8, rel=0, abs=1e-6)
    assert spectrum.detector_id.strip() == "HPGE"
    assert spectrum.metadata.get("format") == "IEC"


def test_read_hpge_spectrum_supports_iec():
    path = TEST_DATA_DIR / "spectrum_io" / "samples" / "hpge_dummy_test_01.iec"
    spectrum = read_hpge_spectrum(path)

    assert spectrum.counts.size == 2048
