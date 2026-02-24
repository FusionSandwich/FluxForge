"""Tests for k0-NAA analysis module."""

import pytest
from fluxforge.analysis.k0_naa import calculate_k0_parameters, K0Parameters

def test_calculate_k0_parameters_ideal():
    """Test k0 calculation with ideal data (f=0, alpha=0)."""
    # Ideal case: R_cd = 1 + f * Q0 (if alpha=0)
    # Let's set f=10, alpha=0
    # R_cd = 1 + 10 * Q0
    # A_bare = R_cd * A_cd
    
    # Q0 values from module:
    # sc46: 0.43, co60: 1.99, cu64: 0.975, fe59: 0.45
    
    f_target = 10.0
    
    bare_activities = {}
    cd_activities = {}
    
    # Setup data for Sc46
    q0_sc = 0.43
    r_cd_sc = 1 + f_target * q0_sc
    cd_activities['sc46'] = 100.0
    bare_activities['sc46'] = cd_activities['sc46'] * r_cd_sc
    
    # Setup data for Co60
    q0_co = 1.99
    r_cd_co = 1 + f_target * q0_co
    cd_activities['co60'] = 100.0  # Same activity to ensure slope ~ 0 for alpha check
    bare_activities['co60'] = cd_activities['co60'] * r_cd_co
    
    params = calculate_k0_parameters(bare_activities, cd_activities)
    
    assert params.f == pytest.approx(f_target, rel=1e-5)
    # Alpha should be 0 because we didn't vary activity with E_res
    assert params.alpha == pytest.approx(0.0, abs=0.1)

def test_calculate_k0_parameters_missing_data():
    """Test handling of missing isotopes."""
    bare = {'sc46': 100.0}
    cd = {'sc46': 10.0} # R=10 -> f = (9)/0.43 = 20.93
    
    params = calculate_k0_parameters(bare, cd)
    
    assert params.f > 0
    assert params.alpha == 0.0 # Not enough points for alpha

def test_calculate_k0_parameters_empty():
    """Test handling of empty input."""
    params = calculate_k0_parameters({}, {})
    assert params.f == 0.0
    assert params.alpha == 0.0


def test_k0_correction_factors():
    """Correction factors should scale concentrations as expected."""
    from fluxforge.analysis.k0_naa import K0Calculator, K0Measurement, K0Parameters
    
    flux_params = K0Parameters(f=20.0, alpha=0.0)
    
    au_meas = K0Measurement(
        product_isotope='Au-198',
        net_peak_area=1e6,
        peak_area_unc=1e4,
        efficiency=0.01,
        t_irr=3600,
        t_decay=3600,
        t_count=3600,
        sample_mass=0.001,
        g_th=1.0,
        g_ep=1.0,
        cd_factor=1.0,
    )
    
    calc = K0Calculator(flux_params, au_meas)
    
    meas = K0Measurement(
        product_isotope='Co-60',
        net_peak_area=5e4,
        peak_area_unc=5e2,
        efficiency=0.005,
        t_irr=3600,
        t_decay=3600,
        t_count=3600,
        sample_mass=0.1,
        g_th=1.0,
        g_ep=1.0,
        cd_factor=1.0,
    )
    
    baseline = calc.calculate_concentration(meas)
    
    meas_cd = K0Measurement(
        product_isotope='Co-60',
        net_peak_area=5e4,
        peak_area_unc=5e2,
        efficiency=0.005,
        t_irr=3600,
        t_decay=3600,
        t_count=3600,
        sample_mass=0.1,
        g_th=1.0,
        g_ep=1.0,
        cd_factor=2.0,
    )
    scaled = calc.calculate_concentration(meas_cd)
    
    assert scaled.concentration_ug_g == pytest.approx(baseline.concentration_ug_g / 2.0, rel=1e-6)
