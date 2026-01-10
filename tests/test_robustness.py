"""
Tests for wire set robustness diagnostics module.

Tests conditioning analysis, energy coverage, and stability metrics.
"""

import pytest
import numpy as np

from fluxforge.analysis.robustness import (
    RobustnessLevel,
    ConditioningMetrics,
    EnergyCoverage,
    LeaveOneOutResult,
    WireSetDiagnostics,
    calculate_condition_metrics,
    analyze_energy_coverage,
    leave_one_out_analysis,
    diagnose_wire_set,
    quick_condition_check,
    estimate_optimal_wire_count,
)


class TestConditioningMetrics:
    """Tests for ConditioningMetrics dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        metrics = ConditioningMetrics(
            condition_number=1000.0,
            log_condition=3.0,
            effective_rank=5,
            truncation_threshold=1e-10,
            singular_values=np.array([100, 50, 25, 10, 1]),
            explained_variance=np.array([0.5, 0.8, 0.95, 0.99, 1.0]),
        )
        
        assert metrics.condition_number == 1000.0
        assert metrics.log_condition == 3.0
        assert metrics.effective_rank == 5
    
    def test_robustness_level_excellent(self):
        """Test excellent robustness classification."""
        metrics = ConditioningMetrics(
            condition_number=10,
            log_condition=1.0,
            effective_rank=5,
            truncation_threshold=1e-10,
            singular_values=np.array([10, 9, 8, 7, 1]),
            explained_variance=np.array([0.3, 0.5, 0.7, 0.9, 1.0]),
        )
        
        assert metrics.robustness_level == RobustnessLevel.EXCELLENT
    
    def test_robustness_level_poor(self):
        """Test poor robustness classification."""
        metrics = ConditioningMetrics(
            condition_number=1e7,
            log_condition=7.0,
            effective_rank=2,
            truncation_threshold=1e-10,
            singular_values=np.array([1e7, 1]),
            explained_variance=np.array([0.99, 1.0]),
        )
        
        assert metrics.robustness_level == RobustnessLevel.POOR


class TestCalculateConditionMetrics:
    """Tests for calculate_condition_metrics function."""
    
    def test_well_conditioned_matrix(self):
        """Test with well-conditioned matrix."""
        # Orthogonal-ish matrix (well conditioned)
        R = np.eye(5)
        
        metrics = calculate_condition_metrics(R)
        
        assert metrics.condition_number == pytest.approx(1.0)
        assert metrics.log_condition == pytest.approx(0.0)
        assert metrics.effective_rank == 5
    
    def test_ill_conditioned_matrix(self):
        """Test with ill-conditioned matrix."""
        # Create rank-deficient matrix
        R = np.array([
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10],  # Linear dependent
            [1, 1, 1, 1, 1],
        ])
        
        metrics = calculate_condition_metrics(R)
        
        # Should detect rank deficiency
        assert metrics.effective_rank < 3
        assert metrics.log_condition > 10  # Very ill-conditioned
    
    def test_singular_values_ordered(self):
        """Test that singular values are in descending order."""
        R = np.random.rand(10, 20)
        
        metrics = calculate_condition_metrics(R)
        
        # SVs should be descending
        diffs = np.diff(metrics.singular_values)
        assert np.all(diffs <= 0)
    
    def test_explained_variance_cumulative(self):
        """Test that explained variance is cumulative."""
        R = np.random.rand(5, 10)
        
        metrics = calculate_condition_metrics(R)
        
        # Should be monotonically increasing
        diffs = np.diff(metrics.explained_variance)
        assert np.all(diffs >= 0)
        
        # Last value should be 1
        assert metrics.explained_variance[-1] == pytest.approx(1.0)


class TestEnergyCoverage:
    """Tests for EnergyCoverage dataclass."""
    
    def test_creation(self):
        """Test basic creation."""
        coverage = EnergyCoverage(
            energy_bins=np.logspace(-4, 1, 11) * 1e6,
            coverage_per_group=np.ones(10) * 0.8,
            gaps=[],
            peak_sensitivity_groups=[0, 1, 2],
            thermal_coverage=0.9,
            epithermal_coverage=0.7,
            fast_coverage=0.6,
        )
        
        assert coverage.thermal_coverage == 0.9
        assert len(coverage.coverage_per_group) == 10


class TestAnalyzeEnergyCoverage:
    """Tests for analyze_energy_coverage function."""
    
    def test_uniform_coverage(self):
        """Test with uniform coverage."""
        n_wires = 5
        n_groups = 20
        energy_bins = np.logspace(-4, 1, n_groups + 1) * 1e6  # eV
        
        # Equal response in all groups
        R = np.ones((n_wires, n_groups))
        
        coverage = analyze_energy_coverage(R, energy_bins)
        
        # All groups should be covered
        assert all(c > 0.5 for c in coverage.coverage_per_group)
        assert len(coverage.gaps) == 0
    
    def test_gap_detection(self):
        """Test gap detection."""
        n_wires = 3
        n_groups = 20
        energy_bins = np.logspace(-4, 1, n_groups + 1) * 1e6
        
        # Create gap in middle
        R = np.ones((n_wires, n_groups))
        R[:, 8:12] = 0  # Gap in groups 8-11
        
        coverage = analyze_energy_coverage(R, energy_bins)
        
        assert len(coverage.gaps) >= 1
    
    def test_regional_coverage(self):
        """Test thermal/epithermal/fast coverage."""
        n_wires = 5
        n_groups = 30
        energy_bins = np.logspace(-5, 1, n_groups + 1) * 1e6  # eV
        
        R = np.random.rand(n_wires, n_groups)
        
        coverage = analyze_energy_coverage(R, energy_bins)
        
        # All coverages should be between 0 and 1
        assert 0 <= coverage.thermal_coverage <= 1
        assert 0 <= coverage.epithermal_coverage <= 1
        assert 0 <= coverage.fast_coverage <= 1


class TestLeaveOneOutAnalysis:
    """Tests for leave_one_out_analysis function."""
    
    def test_basic_analysis(self):
        """Test basic leave-one-out analysis."""
        R = np.random.rand(5, 10)
        
        results = leave_one_out_analysis(R)
        
        assert len(results) == 5
        assert all(isinstance(r, LeaveOneOutResult) for r in results)
    
    def test_wire_names(self):
        """Test custom wire names."""
        R = np.random.rand(3, 5)
        names = ["Au-197", "Co-59", "Ni-58"]
        
        results = leave_one_out_analysis(R, wire_names=names)
        
        assert results[0].wire_name == "Au-197"
        assert results[1].wire_name == "Co-59"
        assert results[2].wire_name == "Ni-58"
    
    def test_importance_scores(self):
        """Test importance scores are valid."""
        R = np.random.rand(5, 10)
        
        results = leave_one_out_analysis(R)
        
        for r in results:
            assert 0 <= r.importance_score <= 1
    
    def test_critical_detection(self):
        """Test critical wire detection."""
        # Create matrix where one wire provides unique coverage
        R = np.zeros((3, 5))
        R[0, :] = [1, 0.1, 0.1, 0.1, 0.1]  # Only one covering group 0 well
        R[1, :] = [0.1, 1, 1, 0.1, 0.1]    # Covers groups 1-2
        R[2, :] = [0.1, 0.1, 0.1, 1, 1]    # Covers groups 3-4
        
        results = leave_one_out_analysis(R)
        
        # All wires should have some importance since each covers unique region
        assert all(r.importance_score >= 0 for r in results)


class TestDiagnoseWireSet:
    """Tests for diagnose_wire_set function."""
    
    def test_basic_diagnosis(self):
        """Test basic wire set diagnosis."""
        n_wires = 8
        n_groups = 50
        R = np.random.rand(n_wires, n_groups)
        energy_bins = np.logspace(-4, 1, n_groups + 1) * 1e6
        
        diag = diagnose_wire_set(R, energy_bins)
        
        assert isinstance(diag, WireSetDiagnostics)
        assert diag.n_wires == n_wires
        assert diag.n_groups == n_groups
    
    def test_with_wire_names(self):
        """Test diagnosis with wire names."""
        R = np.random.rand(4, 20)
        energy_bins = np.logspace(-4, 1, 21) * 1e6
        names = ["Au-197", "Co-59", "Fe-54", "Ni-58"]
        
        diag = diagnose_wire_set(R, energy_bins, wire_names=names)
        
        assert len(diag.leave_one_out) == 4
        assert diag.leave_one_out[0].wire_name == "Au-197"
    
    def test_summary_generation(self):
        """Test summary text generation."""
        R = np.random.rand(6, 30)
        energy_bins = np.logspace(-4, 1, 31) * 1e6
        
        diag = diagnose_wire_set(R, energy_bins)
        summary = diag.summary()
        
        assert isinstance(summary, str)
        assert "WIRE SET ROBUSTNESS DIAGNOSTICS" in summary
        assert "Condition Number" in summary
    
    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        R = np.random.rand(5, 25)
        energy_bins = np.logspace(-4, 1, 26) * 1e6
        
        diag = diagnose_wire_set(R, energy_bins)
        
        assert isinstance(diag.recommendations, list)
        # Should have at least one recommendation
        assert len(diag.recommendations) >= 1
    
    def test_redundancy_factor(self):
        """Test redundancy factor calculation."""
        n_wires = 10
        R = np.eye(n_wires, 20)  # Rank = 10
        energy_bins = np.logspace(-4, 1, 21) * 1e6
        
        diag = diagnose_wire_set(R, energy_bins)
        
        # Redundancy = n_wires / effective_rank
        # With identity-like matrix, should be close to 1
        assert diag.redundancy_factor >= 0.8


class TestQuickConditionCheck:
    """Tests for quick_condition_check function."""
    
    def test_returns_tuple(self):
        """Test return value format."""
        R = np.random.rand(5, 10)
        
        result = quick_condition_check(R)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)  # log condition
        assert isinstance(result[1], str)  # assessment
    
    def test_well_conditioned_assessment(self):
        """Test assessment for well-conditioned matrix."""
        R = np.eye(5)
        
        log_cond, assessment = quick_condition_check(R)
        
        assert log_cond < 1
        assert "Excellent" in assessment or "well" in assessment.lower()
    
    def test_ill_conditioned_assessment(self):
        """Test assessment for ill-conditioned matrix."""
        # Near-singular matrix
        R = np.array([
            [1, 2, 3],
            [1, 2, 3.0001],  # Almost dependent
            [0, 0, 1],
        ])
        
        log_cond, assessment = quick_condition_check(R)
        
        # Should detect ill-conditioning
        assert log_cond > 3


class TestEstimateOptimalWireCount:
    """Tests for estimate_optimal_wire_count function."""
    
    def test_basic_estimate(self):
        """Test basic wire count estimation."""
        n_groups = 100
        
        recommended = estimate_optimal_wire_count(n_groups)
        
        # Should recommend reasonable number
        assert 5 <= recommended <= 30
    
    def test_scales_with_groups(self):
        """Test that recommendation scales with group count."""
        rec_small = estimate_optimal_wire_count(25)
        rec_large = estimate_optimal_wire_count(400)
        
        assert rec_large >= rec_small
    
    def test_respects_redundancy(self):
        """Test redundancy parameter."""
        rec_low = estimate_optimal_wire_count(100, target_redundancy=1.5)
        rec_high = estimate_optimal_wire_count(100, target_redundancy=3.0)
        
        assert rec_high > rec_low


class TestIntegration:
    """Integration tests for robustness analysis."""
    
    def test_realistic_wire_set(self):
        """Test with realistic-ish wire set."""
        # Simulate 10 wires with different energy sensitivities
        n_wires = 10
        n_groups = 52  # Typical group structure
        
        energy_bins = np.logspace(-4, 1, n_groups + 1) * 1e6  # 0.1 eV to 20 MeV
        
        # Create response matrix with peaks at different energies
        R = np.zeros((n_wires, n_groups))
        peak_groups = np.linspace(5, n_groups - 5, n_wires).astype(int)
        
        for i, peak in enumerate(peak_groups):
            # Gaussian-ish response
            for g in range(n_groups):
                R[i, g] = np.exp(-0.5 * ((g - peak) / 5) ** 2)
        
        # Run full diagnostics
        diag = diagnose_wire_set(R, energy_bins)
        
        # Check all components
        assert diag.n_wires == 10
        assert diag.conditioning is not None
        assert diag.coverage is not None
        assert len(diag.leave_one_out) == 10
        
        # Print summary (for manual verification)
        summary = diag.summary()
        assert len(summary) > 100  # Should be substantial
