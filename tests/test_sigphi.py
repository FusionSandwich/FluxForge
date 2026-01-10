"""
Unit tests for SigPhi saturation rate calculations.

Tests the STAYSL-parity SigPhi module for saturation activity corrections.
"""

import math
import pytest
import numpy as np

import sys
sys.path.insert(0, '/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src')

from fluxforge.physics.sigphi import (
    IrradiationHistory,
    MonitorMeasurement,
    SaturationRateResult,
    NeutronBurnupModel,
    calculate_saturation_rate,
    saturation_factor,
    decay_factor,
    counting_factor,
    flux_history_correction_factor,
)


class TestSaturationFactor:
    """Test saturation factor calculations."""
    
    def test_short_irradiation(self):
        """Short irradiation should give S → λt."""
        decay_const = 1e-5  # s⁻¹
        irr_time = 100  # s
        
        S = saturation_factor(decay_const, irr_time)
        expected = decay_const * irr_time  # First-order approximation
        
        assert S == pytest.approx(expected, rel=0.01)
    
    def test_long_irradiation(self):
        """Long irradiation should give S → 1."""
        decay_const = 1e-3  # s⁻¹ (t½ ~ 700s)
        irr_time = 10000  # s (many half-lives)
        
        S = saturation_factor(decay_const, irr_time)
        
        assert S == pytest.approx(1.0, rel=0.01)
    
    def test_zero_decay_constant(self):
        """Zero decay constant (stable) should return 0."""
        S = saturation_factor(0.0, 1000)
        
        assert S == 0.0
    
    def test_half_life_irradiation(self):
        """Irradiation for one half-life should give S = 0.5."""
        half_life = 1000  # s
        decay_const = math.log(2) / half_life
        
        S = saturation_factor(decay_const, half_life)
        
        assert S == pytest.approx(0.5, rel=0.001)


class TestDecayFactor:
    """Test decay factor calculations."""
    
    def test_no_decay(self):
        """Zero delay should give D = 1."""
        D = decay_factor(1e-5, 0)
        
        assert D == pytest.approx(1.0)
    
    def test_one_half_life_decay(self):
        """Decay for one half-life should give D = 0.5."""
        half_life = 3600  # s
        decay_const = math.log(2) / half_life
        
        D = decay_factor(decay_const, half_life)
        
        assert D == pytest.approx(0.5, rel=0.001)
    
    def test_long_decay(self):
        """Long decay should give D → 0."""
        decay_const = 1e-3
        D = decay_factor(decay_const, 100000)
        
        assert D < 0.001


class TestCountingFactor:
    """Test counting factor calculations."""
    
    def test_short_counting(self):
        """Short counting should give C → 1."""
        decay_const = 1e-6  # Very long half-life
        count_time = 100
        
        C = counting_factor(decay_const, count_time)
        
        assert C == pytest.approx(1.0, rel=0.01)
    
    def test_long_counting(self):
        """Long counting relative to half-life."""
        half_life = 100
        decay_const = math.log(2) / half_life
        count_time = 1000  # 10 half-lives
        
        C = counting_factor(decay_const, count_time)
        
        # For long counting: C → 1/(λt) = half_life/(t * ln2)
        assert C < 1.0


class TestIrradiationHistory:
    """Test irradiation history handling."""
    
    def test_single_segment(self):
        """Test single irradiation segment."""
        history = IrradiationHistory(
            segments=[(3600, 1.0)],  # (duration_s, relative_power)
        )
        
        assert history.total_duration_s == 3600
        assert len(history.segments) == 1
    
    def test_multiple_segments(self):
        """Test multi-segment history."""
        history = IrradiationHistory(
            segments=[
                (3600, 1.0),   # Beam on
                (1800, 0.0),   # Beam off
                (3600, 1.0),   # Beam on again
            ]
        )
        
        assert history.total_duration_s == 9000
        assert len(history.segments) == 3


class TestFluxHistoryCorrection:
    """Test flux history correction factor (BCF-style)."""
    
    def test_constant_flux(self):
        """Constant flux should give correction factor based on saturation."""
        history = IrradiationHistory(
            segments=[(3600, 1.0)],  # (duration_s, relative_power)
        )
        decay_const = 1e-5
        
        F = flux_history_correction_factor(history, decay_const)
        
        # For single segment, F should equal saturation factor
        assert F == pytest.approx(saturation_factor(decay_const, 3600), rel=0.001)
    
    def test_varying_flux(self):
        """Varying flux should give flux-weighted correction factor."""
        history = IrradiationHistory(
            segments=[
                (1800, 2.0),  # High power first
                (1800, 1.0),  # Lower power second
            ]
        )
        decay_const = 1e-4  # Moderate decay
        
        F = flux_history_correction_factor(history, decay_const)
        
        # Correction should account for early high flux contribution
        assert F > 0


class TestCalculateSaturationRate:
    """Test full saturation rate calculation."""
    
    def test_simple_case(self):
        """Test basic saturation rate calculation."""
        half_life = 5.27 * 365.25 * 24 * 3600  # 5.27 years in seconds
        
        history = IrradiationHistory(
            segments=[(3600, 1.0)],
            cooling_time_s=1800,
            counting_time_s=3600,
        )
        
        measurement = MonitorMeasurement(
            reaction_id="Co-59_ng_Co-60",
            activity_bq=10000.0,
            activity_uncertainty=500.0,
            half_life_s=half_life,
            target_atoms=1e18,
            irradiation_history=history,
        )
        
        result = calculate_saturation_rate(measurement=measurement)
        
        assert isinstance(result, SaturationRateResult)
        assert result.R_sat > 0
        assert result.uncertainty >= 0
    
    def test_result_fields(self):
        """Test that result contains all required fields."""
        half_life = 2.6943 * 24 * 3600  # 2.6943 days
        
        history = IrradiationHistory(
            segments=[(7200, 1.0)],
            cooling_time_s=600,
            counting_time_s=3600,
        )
        
        measurement = MonitorMeasurement(
            reaction_id="Au-197_ng_Au-198",
            activity_bq=50000.0,
            activity_uncertainty=224.0,
            half_life_s=half_life,
            target_atoms=1e17,
            irradiation_history=history,
        )
        
        result = calculate_saturation_rate(measurement=measurement)
        
        assert hasattr(result, 'R_sat')
        assert hasattr(result, 'uncertainty')
        assert hasattr(result, 'saturation_factor')
        assert hasattr(result, 'decay_factor')
        assert hasattr(result, 'counting_factor')


class TestBurnupCorrection:
    """Tests for neutron burnup/transmutation correction (SigPhi parity)."""

    def test_no_burnup_when_sigma_zero(self):
        """If sigma_target_destruction is zero, burnup correction should be ~1."""
        half_life = 2.6943 * 24 * 3600  # Au-198
        history = IrradiationHistory(
            segments=[(7200, 1.0)],
            cooling_time_s=0,
            counting_time_s=0,
        )
        measurement = MonitorMeasurement(
            reaction_id="Au-197_ng_Au-198",
            activity_bq=50000.0,
            activity_uncertainty=200.0,
            half_life_s=half_life,
            target_atoms=1e17,
            irradiation_history=history,
        )

        burn = NeutronBurnupModel(
            flux_cm2_s_at_rel_power_1=1e12,
            sigma_target_destruction_barns=0.0,
        )

        result = calculate_saturation_rate(
            measurement=measurement,
            apply_burnup=True,
            burnup_model=burn,
        )
        assert result.burnup_correction == pytest.approx(1.0, rel=1e-6)

    def test_burnup_increases_inferred_rate(self):
        """With target depletion, inferred saturation rate should increase (correction > 1)."""
        half_life = 2.6943 * 24 * 3600
        history = IrradiationHistory(
            segments=[(2 * 24 * 3600, 1.0)],  # 2 days irradiation
            cooling_time_s=0,
            counting_time_s=0,
        )
        measurement = MonitorMeasurement(
            reaction_id="Au-197_ng_Au-198",
            activity_bq=1e6,
            activity_uncertainty=1e4,
            half_life_s=half_life,
            target_atoms=1e17,
            irradiation_history=history,
        )

        no_burn = calculate_saturation_rate(measurement=measurement)

        burn = NeutronBurnupModel(
            flux_cm2_s_at_rel_power_1=5e13,
            sigma_target_destruction_barns=1000.0,
        )
        with_burn = calculate_saturation_rate(
            measurement=measurement,
            apply_burnup=True,
            burnup_model=burn,
        )

        assert with_burn.burnup_correction >= 1.0
        assert with_burn.R_sat >= no_burn.R_sat

    def test_burnup_guardrail_warning(self):
        """Very aggressive depletion should trigger a warning string."""
        half_life = 2.6943 * 24 * 3600
        history = IrradiationHistory(
            segments=[(5 * 24 * 3600, 1.0)],
        )
        measurement = MonitorMeasurement(
            reaction_id="Au-197_ng_Au-198",
            activity_bq=1e6,
            activity_uncertainty=1e4,
            half_life_s=half_life,
            target_atoms=1e17,
            irradiation_history=history,
        )
        burn = NeutronBurnupModel(
            flux_cm2_s_at_rel_power_1=1e14,
            sigma_target_destruction_barns=1e6,
            max_correction_factor=1.01,
        )
        result = calculate_saturation_rate(
            measurement=measurement,
            apply_burnup=True,
            burnup_model=burn,
        )
        assert result.burnup_warning is not None


class TestSaturationRatesResult:
    """Test result structure."""
    
    def test_result_serialization(self):
        """Test result can be created with all fields."""
        result = SaturationRateResult(
            R_sat=1e-10,
            R_eoi=0.8e-10,
            A_sat=1e4,
            A_eoi=0.8e4,
            A_count=0.7e4,
            flux_history_factor=0.95,
            saturation_factor=0.8,
            decay_factor=0.9,
            counting_factor=0.99,
        )
        
        assert result.R_sat == 1e-10
        assert result.saturation_factor == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
