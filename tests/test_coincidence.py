"""Tests for Coincidence Summing Correction."""

import pytest
from fluxforge.corrections.coincidence import CoincidenceCorrector

class MockEfficiencyCurve:
    def evaluate(self, energy):
        # Simple mock efficiency: 1% everywhere
        return 0.01

def test_coincidence_corrector_default():
    corrector = CoincidenceCorrector()
    corr = corrector.calculate_correction('Cs137', 661.7)
    assert corr.factor == 1.0
    assert corr.uncertainty == 0.0

def test_coincidence_corrector_co60():
    eff_curve = MockEfficiencyCurve()
    corrector = CoincidenceCorrector(eff_curve)
    
    # Test 1173 keV (coincident with 1332)
    # Factor = 1 / (1 - eff(1332)) = 1 / (1 - 0.01) = 1 / 0.99 = 1.0101
    corr = corrector.calculate_correction('Co60', 1173.2)
    assert corr.factor == pytest.approx(1.0101, rel=1e-4)
    
    # Test 1332 keV (coincident with 1173)
    corr2 = corrector.calculate_correction('Co60', 1332.5)
    assert corr2.factor == pytest.approx(1.0101, rel=1e-4)
    
    # Test non-coincident energy for Co60 (hypothetical)
    corr3 = corrector.calculate_correction('Co60', 500.0)
    assert corr3.factor == 1.0

def test_coincidence_corrector_y88():
    eff_curve = MockEfficiencyCurve()
    corrector = CoincidenceCorrector(eff_curve)
    
    # Test 898 keV (coincident with 1836)
    corr = corrector.calculate_correction('Y88', 898.0)
    assert corr.factor == pytest.approx(1.0101, rel=1e-4)
