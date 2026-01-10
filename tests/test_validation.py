"""
Tests for the validation module.

Tests C/E table generation, closure metrics, and ValidationBundle functionality.
"""

import numpy as np
import pytest
from datetime import datetime

from fluxforge.core.validation import (
    CEEntry,
    CETable,
    ClosureMetrics,
    ValidationBundle,
    ValidationStatus,
    calculate_ce_table,
    calculate_closure_metrics,
    create_validation_bundle,
)


class TestCEEntry:
    """Tests for CEEntry dataclass."""
    
    def test_ce_ratio_calculation(self):
        """Test C/E ratio is calculated correctly."""
        entry = CEEntry(
            identifier="Test",
            calculated=1.05,
            experimental=1.00,
        )
        assert np.isclose(entry.ce_ratio, 1.05, rtol=1e-10)
    
    def test_ce_ratio_with_uncertainties(self):
        """Test C/E uncertainty propagation."""
        entry = CEEntry(
            identifier="Test",
            calculated=100.0,
            experimental=100.0,
            c_uncertainty=5.0,  # 5%
            e_uncertainty=3.0,  # 3%
        )
        # Combined uncertainty ~ sqrt(0.05^2 + 0.03^2) ~ 0.058
        assert entry.ce_ratio == 1.0
        assert np.isclose(entry.ce_uncertainty, 0.0583, rtol=0.01)
    
    def test_pull_calculation(self):
        """Test pull (standardized residual) calculation."""
        entry = CEEntry(
            identifier="Test",
            calculated=1.10,
            experimental=1.00,
            c_uncertainty=0.05,
            e_uncertainty=0.05,
        )
        # Pull = (1.10 - 1.00) / sqrt(0.05^2 + 0.05^2) = 0.1 / 0.0707 ~ 1.41
        expected_pull = 0.10 / np.sqrt(0.05**2 + 0.05**2)
        assert np.isclose(entry.pull, expected_pull, rtol=0.01)
    
    def test_within_tolerance_true(self):
        """Test tolerance check passes when within bounds."""
        entry = CEEntry(
            identifier="Test",
            calculated=1.00,
            experimental=1.00,
            c_uncertainty=0.05,
            e_uncertainty=0.05,
        )
        assert entry.within_tolerance == True
    
    def test_within_tolerance_false(self):
        """Test tolerance check fails when outside bounds."""
        entry = CEEntry(
            identifier="Test",
            calculated=1.50,  # 50% high
            experimental=1.00,
            c_uncertainty=0.01,  # Small uncertainty
            e_uncertainty=0.01,
        )
        assert entry.within_tolerance == False
    
    def test_zero_experimental(self):
        """Test handling of zero experimental value."""
        entry = CEEntry(
            identifier="Test",
            calculated=1.0,
            experimental=0.0,
        )
        assert entry.ce_ratio == 0.0  # Division by zero protected


class TestCETable:
    """Tests for CETable dataclass."""
    
    def test_n_entries(self):
        """Test entry count."""
        entries = [
            CEEntry("A", 1.0, 1.0),
            CEEntry("B", 1.0, 1.0),
            CEEntry("C", 1.0, 1.0),
        ]
        table = CETable(entries=entries)
        assert table.n_entries == 3
    
    def test_mean_ce(self):
        """Test mean C/E calculation."""
        entries = [
            CEEntry("A", 1.10, 1.0),  # C/E = 1.1
            CEEntry("B", 0.90, 1.0),  # C/E = 0.9
            CEEntry("C", 1.00, 1.0),  # C/E = 1.0
        ]
        table = CETable(entries=entries)
        assert np.isclose(table.mean_ce, 1.0, rtol=1e-10)
    
    def test_std_ce(self):
        """Test standard deviation of C/E."""
        entries = [
            CEEntry("A", 1.10, 1.0),
            CEEntry("B", 0.90, 1.0),
            CEEntry("C", 1.00, 1.0),
        ]
        table = CETable(entries=entries)
        expected_std = np.std([1.1, 0.9, 1.0], ddof=1)
        assert np.isclose(table.std_ce, expected_std, rtol=1e-10)
    
    def test_fraction_within_tolerance(self):
        """Test fraction within tolerance calculation."""
        entries = [
            CEEntry("A", 1.0, 1.0, 0.05, 0.05),  # Within
            CEEntry("B", 1.0, 1.0, 0.05, 0.05),  # Within
            CEEntry("C", 2.0, 1.0, 0.01, 0.01),  # Outside (C/E = 2)
        ]
        table = CETable(entries=entries)
        assert np.isclose(table.fraction_within_tolerance, 2/3, rtol=0.01)
    
    def test_to_markdown(self):
        """Test markdown table generation."""
        entries = [
            CEEntry("Reaction_1", 1.05, 1.0, 0.05, 0.03),
        ]
        table = CETable(entries=entries, description="Test comparison")
        md = table.to_markdown()
        
        assert "Reaction_1" in md
        assert "C/E" in md
        assert "Mean C/E" in md
    
    def test_empty_table(self):
        """Test empty table handling."""
        table = CETable()
        assert table.n_entries == 0
        assert table.mean_ce == 0.0
        assert table.std_ce == 0.0
        assert table.fraction_within_tolerance == 0.0


class TestClosureMetrics:
    """Tests for ClosureMetrics dataclass."""
    
    def test_chi2_acceptable_true(self):
        """Test chi-square acceptability when p-value > 0.05."""
        metrics = ClosureMetrics(p_value=0.15)
        assert metrics.chi2_acceptable is True
    
    def test_chi2_acceptable_false(self):
        """Test chi-square unacceptable when p-value < 0.05."""
        metrics = ClosureMetrics(p_value=0.01)
        assert metrics.chi2_acceptable is False
    
    def test_pulls_normal_true(self):
        """Test pull normality when K-S p > 0.05."""
        metrics = ClosureMetrics(ks_pvalue=0.20)
        assert metrics.pulls_normal is True
    
    def test_pulls_normal_false(self):
        """Test pull normality failure."""
        metrics = ClosureMetrics(ks_pvalue=0.01)
        assert metrics.pulls_normal is False
    
    def test_summary(self):
        """Test summary string generation."""
        metrics = ClosureMetrics(
            chi_square=5.5,
            dof=10,
            reduced_chi2=0.55,
            p_value=0.85,
            rms_residual=0.05,
            max_pull=1.2,
            ks_pvalue=0.45,
        )
        summary = metrics.summary()
        assert "Chi-square: 5.50" in summary
        assert "dof=10" in summary
        assert "✓" in summary  # P-value acceptable


class TestValidationBundle:
    """Tests for ValidationBundle dataclass."""
    
    def test_determine_status_passed(self):
        """Test status determination for passing validation."""
        ce_entries = [CEEntry(f"R{i}", 1.0, 1.0, 0.05, 0.05) for i in range(10)]
        ce_table = CETable(entries=ce_entries)
        
        closure = ClosureMetrics(
            p_value=0.5,  # Good
            max_pull=1.0,  # Good
        )
        
        bundle = ValidationBundle(ce_table=ce_table, closure=closure)
        assert bundle.determine_status() == ValidationStatus.PASSED
    
    def test_determine_status_failed(self):
        """Test status determination for failing validation."""
        ce_entries = [CEEntry(f"R{i}", 2.0, 1.0, 0.01, 0.01) for i in range(10)]
        ce_table = CETable(entries=ce_entries)
        
        closure = ClosureMetrics(
            p_value=0.001,  # Bad
            max_pull=5.0,   # Bad
        )
        
        bundle = ValidationBundle(ce_table=ce_table, closure=closure)
        assert bundle.determine_status() == ValidationStatus.FAILED
    
    def test_determine_status_incomplete(self):
        """Test status for incomplete validation."""
        bundle = ValidationBundle()
        assert bundle.determine_status() == ValidationStatus.INCOMPLETE
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        bundle = ValidationBundle(
            reference_type="MCNP",
            reference_label="Reference flux",
            test_label="Unfolded flux",
            status=ValidationStatus.PASSED,
        )
        
        d = bundle.to_dict()
        assert d['schema'] == 'fluxforge://validation_bundle/v1'
        assert d['reference_type'] == 'MCNP'
        assert d['status'] == 'passed'
    
    def test_full_report(self):
        """Test full report generation."""
        ce_entries = [CEEntry("Au-197(n,g)", 1.02, 1.0, 0.03, 0.02)]
        bundle = ValidationBundle(
            ce_table=CETable(entries=ce_entries),
            closure=ClosureMetrics(chi_square=1.5, dof=1, p_value=0.22),
            reference_type="MCNP",
            reference_label="MCNP tallies",
            test_label="GRAVEL unfolded",
            status=ValidationStatus.PASSED,
            notes="Test validation run",
        )
        
        report = bundle.full_report()
        assert "VALIDATION REPORT" in report
        assert "MCNP" in report
        assert "Au-197(n,g)" in report
        assert "Test validation run" in report


class TestCalculateCETable:
    """Tests for calculate_ce_table function."""
    
    def test_basic_calculation(self):
        """Test basic C/E table calculation."""
        calc = np.array([1.05, 0.95, 1.02])
        expt = np.array([1.00, 1.00, 1.00])
        
        table = calculate_ce_table(calc, expt)
        
        assert table.n_entries == 3
        assert np.isclose(table.entries[0].ce_ratio, 1.05)
        assert np.isclose(table.entries[1].ce_ratio, 0.95)
    
    def test_with_uncertainties(self):
        """Test with uncertainty propagation."""
        calc = np.array([100.0, 100.0])
        expt = np.array([100.0, 100.0])
        c_unc = np.array([5.0, 10.0])
        e_unc = np.array([3.0, 5.0])
        
        table = calculate_ce_table(calc, expt, c_unc, e_unc)
        
        # First entry: 5% C unc, 3% E unc -> ~5.83% combined
        assert table.entries[0].ce_uncertainty > 0
    
    def test_with_identifiers(self):
        """Test with custom identifiers."""
        calc = np.array([1.0, 1.0])
        expt = np.array([1.0, 1.0])
        ids = ["Au-197(n,g)", "Co-59(n,g)"]
        
        table = calculate_ce_table(calc, expt, identifiers=ids)
        
        assert table.entries[0].identifier == "Au-197(n,g)"
        assert table.entries[1].identifier == "Co-59(n,g)"
    
    def test_custom_tolerance(self):
        """Test custom tolerance setting."""
        calc = np.array([1.05])
        expt = np.array([1.00])
        
        table_strict = calculate_ce_table(calc, expt, tolerance=0.01)
        table_relaxed = calculate_ce_table(calc, expt, tolerance=0.10)
        
        assert table_strict.entries[0].within_tolerance is False
        assert table_relaxed.entries[0].within_tolerance is True


class TestCalculateClosureMetrics:
    """Tests for calculate_closure_metrics function."""
    
    def test_perfect_match(self):
        """Test metrics for perfect match."""
        calc = np.array([1.0, 2.0, 3.0])
        expt = np.array([1.0, 2.0, 3.0])
        unc = np.array([0.1, 0.1, 0.1])
        
        metrics = calculate_closure_metrics(calc, expt, c_uncertainties=unc, e_uncertainties=unc)
        
        assert metrics.chi_square < 0.01
        assert metrics.rms_residual < 1e-10
        assert metrics.max_pull < 1e-10
    
    def test_with_residuals(self):
        """Test metrics with residuals present."""
        calc = np.array([1.1, 2.2, 2.8])
        expt = np.array([1.0, 2.0, 3.0])
        unc = np.array([0.1, 0.1, 0.1])
        
        metrics = calculate_closure_metrics(calc, expt, c_uncertainties=unc, e_uncertainties=unc)
        
        assert metrics.chi_square > 0
        assert metrics.rms_residual > 0
        assert metrics.max_pull > 0
    
    def test_with_covariance_matrix(self):
        """Test with full covariance matrix."""
        calc = np.array([1.1, 2.1])
        expt = np.array([1.0, 2.0])
        cov = np.array([[0.01, 0.005], [0.005, 0.01]])
        
        metrics = calculate_closure_metrics(calc, expt, covariance=cov)
        
        assert metrics.chi_square > 0
        assert metrics.dof == 2
    
    def test_reduced_chi2(self):
        """Test reduced chi-square calculation."""
        calc = np.array([1.0, 1.0, 1.0, 1.0])
        expt = np.array([1.0, 1.0, 1.0, 1.0])
        
        metrics = calculate_closure_metrics(calc, expt)
        
        assert metrics.dof == 4


class TestCreateValidationBundle:
    """Tests for create_validation_bundle function."""
    
    def test_complete_bundle_creation(self):
        """Test creating complete validation bundle."""
        calc = np.array([1.02, 0.98, 1.01, 0.99])
        expt = np.array([1.00, 1.00, 1.00, 1.00])
        c_unc = np.array([0.05, 0.05, 0.05, 0.05])
        e_unc = np.array([0.03, 0.03, 0.03, 0.03])
        ids = ["R1", "R2", "R3", "R4"]
        
        bundle = create_validation_bundle(
            calc, expt,
            c_uncertainties=c_unc,
            e_uncertainties=e_unc,
            identifiers=ids,
            reference_type="MCNP",
            reference_label="MCNP flux tallies",
            test_label="GRAVEL unfolded",
        )
        
        assert bundle.ce_table.n_entries == 4
        assert bundle.closure.dof == 4
        assert bundle.reference_type == "MCNP"
        assert bundle.status in [ValidationStatus.PASSED, ValidationStatus.MARGINAL]
    
    def test_provenance_populated(self):
        """Test provenance information is populated."""
        calc = np.array([1.0, 1.0])
        expt = np.array([1.0, 1.0])
        
        bundle = create_validation_bundle(calc, expt, tolerance=0.15)
        
        assert 'created_at' in bundle.provenance
        assert bundle.provenance['n_entries'] == 2
        assert bundle.provenance['tolerance'] == 0.15
    
    def test_status_auto_determined(self):
        """Test status is automatically determined."""
        # Good match
        calc = np.array([1.0, 1.0, 1.0])
        expt = np.array([1.0, 1.0, 1.0])
        c_unc = np.array([0.05, 0.05, 0.05])
        e_unc = np.array([0.05, 0.05, 0.05])
        
        bundle = create_validation_bundle(calc, expt, c_unc, e_unc)
        
        assert bundle.status == ValidationStatus.PASSED
    
    def test_failed_validation(self):
        """Test bundle with failing validation."""
        calc = np.array([2.0, 0.5, 3.0])  # Large discrepancies
        expt = np.array([1.0, 1.0, 1.0])
        c_unc = np.array([0.01, 0.01, 0.01])  # Small uncertainties
        e_unc = np.array([0.01, 0.01, 0.01])
        
        bundle = create_validation_bundle(calc, expt, c_unc, e_unc)
        
        assert bundle.status == ValidationStatus.FAILED


class TestValidationIntegration:
    """Integration tests for validation workflow."""
    
    def test_full_workflow(self):
        """Test complete validation workflow."""
        # Simulate comparing unfolded spectrum to reference
        np.random.seed(42)
        n_groups = 10
        
        # Reference spectrum
        reference = np.logspace(10, 6, n_groups)  # Decreasing flux
        ref_unc = reference * 0.03  # 3% uncertainty
        
        # Unfolded spectrum with some scatter
        unfolded = reference * (1 + np.random.normal(0, 0.02, n_groups))
        unf_unc = unfolded * 0.05  # 5% uncertainty
        
        # Energy group identifiers
        groups = [f"Group_{i+1}" for i in range(n_groups)]
        
        bundle = create_validation_bundle(
            unfolded, reference,
            c_uncertainties=unf_unc,
            e_uncertainties=ref_unc,
            identifiers=groups,
            reference_type="Transport",
            reference_label="MCNP reference spectrum",
            test_label="GRAVEL unfolded spectrum",
        )
        
        # Check bundle is complete
        assert bundle.ce_table.n_entries == n_groups
        assert bundle.closure.chi_square >= 0
        assert bundle.status != ValidationStatus.INCOMPLETE
        
        # Check report generation
        report = bundle.full_report()
        assert "VALIDATION REPORT" in report
        assert "MCNP reference spectrum" in report
    
    def test_reaction_rate_validation(self):
        """Test validation of reaction rates."""
        # Simulated dosimetry reaction rates
        reactions = [
            "Au-197(n,g)",
            "In-115(n,n')",
            "Ni-58(n,p)",
            "Fe-54(n,p)",
            "Al-27(n,a)",
            "Nb-93(n,2n)",
        ]
        
        # Measured rates (normalized)
        measured = np.array([1.23e-4, 4.56e-5, 2.34e-5, 1.12e-5, 5.67e-6, 8.90e-7])
        meas_unc = measured * 0.05  # 5% measurement uncertainty
        
        # Calculated from unfolded spectrum (slightly different)
        calculated = measured * np.array([1.02, 0.97, 1.03, 0.98, 1.01, 0.99])
        calc_unc = calculated * 0.03  # 3% calculation uncertainty
        
        bundle = create_validation_bundle(
            calculated, measured,
            c_uncertainties=calc_unc,
            e_uncertainties=meas_unc,
            identifiers=reactions,
            reference_type="Measurement",
            reference_label="Activation measurements",
            test_label="Calculated from unfolded flux",
        )
        
        # This should pass (small differences, reasonable uncertainties)
        assert bundle.ce_table.fraction_within_tolerance > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
