from pathlib import Path

from fluxforge.io.calibration_text import (
    read_pygammaspec_calibration,
    pygammaspec_to_calibration,
)

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_read_pygammaspec_calibration():
    path = TEST_DATA_DIR / "pygammaspec" / "utils" / "calibration.txt"
    cal_data = read_pygammaspec_calibration(path)

    assert cal_data.order == 2
    assert len(cal_data.channels) == 5
    assert len(cal_data.coefficients_desc) == 3


def test_pygammaspec_conversion():
    path = TEST_DATA_DIR / "pygammaspec" / "utils" / "calibration.txt"
    cal_data = read_pygammaspec_calibration(path)
    calibration = pygammaspec_to_calibration(cal_data)

    energy = calibration(14.41)
    assert abs(energy - 609.312) < 1.0
