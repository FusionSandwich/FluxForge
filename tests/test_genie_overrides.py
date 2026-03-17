from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc
from fluxforge.io.genie import read_genie_spectrum


TEST_DATA_ROOT = Path(__file__).resolve().parent / "data" / "flux_wires"
RAW_ASC = TEST_DATA_ROOT / "raw" / "Co-Cd-RAFM-1_25cm.ASC"
PROC_TXT = TEST_DATA_ROOT / "processed" / "Co-Cd-RAFM-1_25cm.txt"


def test_read_genie_spectrum_header_only_uses_file_calibration():
    spectrum = read_genie_spectrum(RAW_ASC)
    assert spectrum.calibration.get("energy")
    cal = spectrum.calibration["energy"]
    assert cal[0] == pytest.approx(0.541, rel=0, abs=1e-3)
    assert cal[1] == pytest.approx(0.498, rel=0, abs=1e-3)


def test_read_genie_spectrum_user_override_precedence():
    energy_override = [1.0, 2.0, 0.0]
    efficiency_override = {"C1": 1.0, "C2": 2.0, "C3": 3.0, "C4": 4.0, "DetModel": 0.5}
    spectrum = read_genie_spectrum(
        RAW_ASC,
        energy_calibration_override=energy_override,
        efficiency_override=efficiency_override,
    )
    assert spectrum.calibration["energy"] == energy_override
    assert np.allclose(spectrum.energies[:3], [1.0, 3.0, 5.0])
    for key, value in efficiency_override.items():
        assert spectrum.metadata["efficiency"][key] == pytest.approx(value)


def test_read_genie_spectrum_user_only_override(tmp_path: Path):
    asc_file = tmp_path / "no_calibration.ASC"
    asc_file.write_text(
        "\n".join(
            [
                "ID: test",
                "Elapsed Real Time: 10",
                "Elapsed Live Time: 10",
                "Channel     Contents",
                "0 5",
                "1 7",
                "2 9",
            ]
        ),
        encoding="utf-8",
    )

    spectrum = read_genie_spectrum(
        asc_file,
        energy_calibration_override=[0.5, 0.25, 0.0],
        efficiency_override={"C1": -1.0, "C2": 0.5, "C3": 0.0, "C4": 0.0},
    )
    assert spectrum.calibration["energy"] == [0.5, 0.25, 0.0]
    assert np.allclose(spectrum.energies, [0.5, 0.75, 1.0])
    assert spectrum.metadata["efficiency"]["C1"] == pytest.approx(-1.0)


def test_flux_wire_raw_override_precedence():
    data = read_raw_asc(
        RAW_ASC,
        energy_calibration_override=[2.0, 0.1, 0.0],
        efficiency_override={"C1": 0.1, "C2": 0.2, "C3": 0.3, "C4": 0.4, "DetModel": 0.5},
    )
    assert data.energy_calibration == [2.0, 0.1, 0.0]
    assert data.efficiency is not None
    assert data.efficiency.C1 == pytest.approx(0.1)
    assert data.efficiency.geometry_factor_A == pytest.approx(0.5)


def test_flux_wire_processed_override_precedence():
    data = read_processed_txt(
        PROC_TXT,
        energy_calibration_override=[3.0, 0.25, 0.0],
        efficiency_override={"C1": -2.0, "C2": 0.6, "C3": -0.1, "C4": 0.01, "DetModel": 0.01},
    )
    assert data.energy_calibration == [3.0, 0.25, 0.0]
    assert data.efficiency is not None
    assert data.efficiency.C1 == pytest.approx(-2.0)
    assert data.efficiency.C2 == pytest.approx(0.6)
    assert data.efficiency.geometry_factor_A == pytest.approx(0.01)


def test_flux_wire_raw_profile_keeps_file_energy_and_fills_defaults():
    data = read_raw_asc(RAW_ASC, profile_name="rafm_25cm")
    assert data.energy_calibration[0] == pytest.approx(0.541, rel=0, abs=1e-3)
    assert data.energy_calibration[1] == pytest.approx(0.498, rel=0, abs=1e-3)
    assert data.efficiency is not None
    assert data.efficiency.C1 == pytest.approx(-20.26)
    assert data.efficiency.al_window_T1_um == pytest.approx(1000.0)
    assert data.efficiency.detector_thickness_DI_cm == pytest.approx(6.45)
    assert data.efficiency.dead_layer_DL_um == pytest.approx(700.0)
    assert data.efficiency.incident_angle_AI_deg == pytest.approx(0.0)
    assert data.source_file.endswith("Co-Cd-RAFM-1_25cm.ASC")
    assert data.resolution == pytest.approx([1.389, 7.8e-04, -4.072e-08])


def test_flux_wire_processed_profile_keeps_file_energy():
    data = read_processed_txt(PROC_TXT, profile_name="rafm_25cm")
    assert data.energy_calibration == pytest.approx([-1.694, 0.4996, 6.71e-08])
    assert data.efficiency is not None
    assert data.efficiency.C1 == pytest.approx(-20.26)
