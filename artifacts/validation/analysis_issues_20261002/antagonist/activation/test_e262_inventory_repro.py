"""Known-truth probe for existing E262 target-atom normalization omission."""

import math

import pytest

from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan
from fluxforge.physics.activation import AVOGADRO


def test_e262_recovers_true_flux_from_measured_counts_and_sample_inventory():
    flux, mass_g, molar_mass = 1e10, 0.01, 58.9332
    half_life, duration, count_time = 166337000.0, 7200.0, 1000.0
    efficiency, yield_, sigma_barn = 0.05, 0.9998, 37.18
    lam = math.log(2) / half_life
    n_atoms = mass_g / molar_mass * AVOGADRO
    activity_eoi = n_atoms * flux * sigma_barn * 1e-24 * -math.expm1(-lam * duration)
    counts = activity_eoi * efficiency * yield_ * -math.expm1(-lam * count_time) / lam
    result = analyze_astm_e262_plan({
        "irradiation": {"duration_s": duration},
        "measurements": [{
            "reaction_id": "Co-59(n,g)Co-60", "product_isotope": "Co60",
            "net_counts": counts, "live_time_s": count_time,
            "efficiency": efficiency, "gamma_intensity": yield_, "half_life_s": half_life,
            "sample_mass_g": mass_g, "atomic_mass_g_mol": molar_mass,
            "sigma_0_barn": sigma_barn,
        }],
    })
    recovered = result["measurements"][0]["equivalent_2200ms_fluence_rate_cm2_s"]
    assert recovered == pytest.approx(flux, rel=1e-10, abs=0)
