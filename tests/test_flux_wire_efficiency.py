from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fluxforge.data.xcom import get_attenuation_data
from fluxforge.io.flux_wire import EfficiencyCalibration


ROOT = Path(__file__).resolve().parents[1]


def test_detector_model_uses_linear_attenuation_in_beer_lambert_exponents() -> None:
    energy_keV = 661.657
    aluminum_density_g_cm3 = 2.699
    germanium_density_g_cm3 = 5.323
    aluminum_window_cm = 0.12
    germanium_dead_layer_cm = 0.08
    germanium_active_depth_cm = 0.50

    aluminum = get_attenuation_data("Aluminum")
    germanium = get_attenuation_data("Germanium")
    assert aluminum.density == pytest.approx(aluminum_density_g_cm3)
    assert germanium.density == pytest.approx(germanium_density_g_cm3)

    mu_al_cm_inverse = float(aluminum.get_mu_rho(energy_keV)[0]) * aluminum_density_g_cm3
    mu_ge_cm_inverse = float(germanium.get_mu_rho(energy_keV)[0]) * germanium_density_g_cm3
    expected = np.exp(
        -(
            aluminum_window_cm * mu_al_cm_inverse
            + germanium_dead_layer_cm * mu_ge_cm_inverse
        )
    ) * (1.0 - np.exp(-germanium_active_depth_cm * mu_ge_cm_inverse))

    calibration = EfficiencyCalibration(
        C1=1.0,
        geometry_factor_A=1.0,
        al_window_T1_um=aluminum_window_cm * 1.0e4,
        dead_layer_DL_um=germanium_dead_layer_cm * 1.0e4,
        detector_thickness_DI_cm=germanium_active_depth_cm,
    )

    assert calibration.efficiency(energy_keV) == pytest.approx(expected, rel=1.0e-12)

    mass_coefficient_only = np.exp(
        -(
            aluminum_window_cm * float(aluminum.get_mu_rho(energy_keV)[0])
            + germanium_dead_layer_cm * float(germanium.get_mu_rho(energy_keV)[0])
        )
    ) * (
        1.0
        - np.exp(
            -germanium_active_depth_cm
            * float(germanium.get_mu_rho(energy_keV)[0])
        )
    )
    assert expected != pytest.approx(mass_coefficient_only, rel=0.1)


def test_rafm_profile_efficiency_regression_at_co60_energies() -> None:
    profiles = json.loads(
        (ROOT / "src" / "fluxforge" / "data" / "rafm_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    calibration = EfficiencyCalibration(**profiles["rafm_25cm"]["efficiency"])

    efficiencies = calibration.efficiency(np.asarray([1173.228, 1332.492]))

    assert efficiencies == pytest.approx(
        np.asarray([0.0010656903, 0.0010246541]),
        rel=1.0e-7,
    )


def test_exported_co60_point_is_consistent_without_qualifying_curve() -> None:
    calibration = EfficiencyCalibration(
        C1=-20.2569,
        C2=10.2872,
        C3=-1.65505,
        C4=0.0866563,
        geometry_factor_A=0.00347696,
        al_window_T1_um=1000.0,
        detector_thickness_DI_cm=1.39,
        dead_layer_DL_um=700.0,
    )

    calculated_percent = 100.0 * calibration.efficiency(1332.0)

    # The untouched local export records 0.035443 with no explicit unit.  This
    # checks the candidate percent interpretation, not an independent curve
    # validation.
    assert calculated_percent == pytest.approx(0.035443, rel=0.01)
