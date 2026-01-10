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
