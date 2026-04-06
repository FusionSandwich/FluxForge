from pathlib import Path

from fluxforge.io.calibration_text import (
    read_legacy_spectrum_calibration,
    legacy_spectrum_to_calibration,
)

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_read_legacy_spectrum_calibration():
    path = TEST_DATA_DIR / "legacy_spectrum" / "utils" / "calibration.txt"
    cal_data = read_legacy_spectrum_calibration(path)

    assert cal_data.order == 2
    assert len(cal_data.channels) == 5
    assert len(cal_data.coefficients_desc) == 3


def test_legacy_spectrum_conversion():
    path = TEST_DATA_DIR / "legacy_spectrum" / "utils" / "calibration.txt"
    cal_data = read_legacy_spectrum_calibration(path)
    calibration = legacy_spectrum_to_calibration(cal_data)

    energy = calibration(14.41)
    assert abs(energy - 609.312) < 1.0
