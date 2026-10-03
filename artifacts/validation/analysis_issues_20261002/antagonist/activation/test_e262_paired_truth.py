"""Independent production/count integration checks with unequal monitors."""

import math

import pytest

from fluxforge.analysis.astm_e262 import analyze_astm_e262_plan
from fluxforge.physics.activation import AVOGADRO


HALF_LIFE = 232848.0
MOLAR_MASS = 196.96657
DURATION = 7200.0
SIGMA = 98.65e-24


def measurement(prefix, mass, production_per_atom, live, real, cooling):
    """Generate measured counts directly from the decay ODE and acceptance."""
    atoms = mass / MOLAR_MASS * AVOGADRO
    lam = math.log(2) / HALF_LIFE
    activity_eoi = atoms * production_per_atom * -math.expm1(-lam * DURATION)
    efficiency, yield_ = 0.04, 0.955
    counts = efficiency * yield_ * (live / real) * activity_eoi * math.exp(-lam * cooling) * -math.expm1(-lam * real) / lam
    return {
        prefix + "net_counts": counts,
        prefix + "live_time_s": live,
        prefix + "real_time_s": real,
        prefix + "cooling_time_s": cooling,
        prefix + "half_life_s": HALF_LIFE,
        prefix + "efficiency": efficiency,
        prefix + "gamma_intensity": yield_,
        prefix + "sample_mass_g": mass,
        prefix + "atomic_mass_g_mol": MOLAR_MASS,
    }, atoms, counts


@pytest.mark.parametrize("explicit_atoms", [False, True])
def test_cd_pair_unequal_mass_clock_and_cooling_recovers_thermal_flux(explicit_atoms):
    thermal_flux, epithermal_equivalent_flux = 7e9, 3e9
    westcott, shielding = 1.1, 0.9
    thermal_rate = thermal_flux * SIGMA * westcott * shielding
    epi_rate = epithermal_equivalent_flux * SIGMA
    bare, bare_atoms, bare_counts = measurement("", 0.010, thermal_rate + epi_rate, 480, 600, 60)
    cd, cd_atoms, cd_counts = measurement("cd_", 0.050, epi_rate, 1200, 1500, 1800)
    row = {
        "reaction_id": "Au-197(n,g)Au-198", "product_isotope": "Au198",
        "sigma_0_barn": SIGMA * 1e24,
        "westcott_g": westcott, "thermal_self_shielding_factor": shielding,
        **bare, **cd,
    }
    if explicit_atoms:
        for prefix, atoms in [("", bare_atoms), ("cd_", cd_atoms)]:
            row.pop(prefix + "sample_mass_g")
            row.pop(prefix + "atomic_mass_g_mol")
            row[prefix + "target_atoms"] = atoms
    result = analyze_astm_e262_plan({"irradiation": {"duration_s": DURATION}, "measurements": [row]})["measurements"][0]
    assert result["equivalent_2200ms_fluence_rate_cm2_s"] == pytest.approx(thermal_flux, rel=1e-12, abs=0)
    assert result["thermal_reaction_rate_per_atom_s"] == pytest.approx(thermal_rate, rel=1e-12, abs=0)
    assert result["cadmium_ratio"] == pytest.approx((thermal_rate + epi_rate) / epi_rate, rel=1e-12, abs=0)
    expected_sigma = math.sqrt((thermal_rate + epi_rate)**2 / bare_counts + epi_rate**2 / cd_counts) / (SIGMA * westcott * shielding)
    assert result["equivalent_2200ms_fluence_rate_unc_cm2_s"] == pytest.approx(expected_sigma, rel=1e-12, abs=0)


@pytest.mark.parametrize("explicit_atoms", [False, True])
def test_standard_comparison_unequal_mass_clock_and_cooling_recovers_flux(explicit_atoms):
    unknown_flux, reference_flux = 3e10, 1e10
    unknown, unknown_atoms, _ = measurement("unknown_", 0.002, unknown_flux * SIGMA, 900, 1200, 600)
    standard, standard_atoms, _ = measurement("standard_", 0.030, reference_flux * SIGMA, 100, 200, 7200)
    row = {
        "reaction_id": "Au-197(n,g)Au-198", "mode": "standard_comparison",
        "known_reference_fluence_rate_cm2_s": reference_flux,
        **unknown, **standard,
    }
    if explicit_atoms:
        for prefix, atoms in [("unknown_", unknown_atoms), ("standard_", standard_atoms)]:
            row.pop(prefix + "sample_mass_g")
            row.pop(prefix + "atomic_mass_g_mol")
            row[prefix + "target_atoms"] = atoms
    result = analyze_astm_e262_plan({"irradiation": {"duration_s": DURATION}, "measurements": [row]})["measurements"][0]
    assert result["equivalent_2200ms_fluence_rate_cm2_s"] == pytest.approx(unknown_flux, rel=1e-12, abs=0)
    assert result["unknown_reaction_rate_per_atom_s"] == pytest.approx(unknown_flux * SIGMA, rel=1e-12, abs=0)
    assert result["standard_reaction_rate_per_atom_s"] == pytest.approx(reference_flux * SIGMA, rel=1e-12, abs=0)
