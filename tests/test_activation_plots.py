"""
Tests for Activation Plotting Module

Tests the thesis-quality activation comparison plotting functions.
"""

import pytest
import numpy as np

from fluxforge.plots.activation import (
    ActivityMeasurement,
    ComparisonResult,
    plot_activity_comparison_bar,
    plot_ce_ratio_summary,
    plot_material_comparison_grid,
    plot_decay_curves,
    plot_flux_wire_validation,
    plot_cd_ratio_analysis,
    plot_validation_summary_table,
    apply_thesis_style,
    MATERIAL_COLORS,
    COOLING_COLORS,
    SOURCE_COLORS,
)


# =============================================================================
# Test Data Classes
# =============================================================================

class TestActivityMeasurement:
    """Tests for ActivityMeasurement dataclass."""
    
    def test_create_measurement(self):
        """Test creating an activity measurement."""
        meas = ActivityMeasurement(
            isotope='Mn56',
            activity_Bq=1.5e6,
            uncertainty_Bq=1.5e5,
            cooling_time='300s',
            material='EUROFER97',
            source='experimental',
        )
        
        assert meas.isotope == 'Mn56'
        assert meas.activity_Bq == 1.5e6
        assert meas.uncertainty_Bq == 1.5e5
        assert meas.relative_uncertainty == pytest.approx(0.1)
    
    def test_relative_uncertainty_zero_activity(self):
        """Test relative uncertainty with zero activity."""
        meas = ActivityMeasurement(
            isotope='Mn56',
            activity_Bq=0.0,
            uncertainty_Bq=100.0,
        )
        assert meas.relative_uncertainty == 0.0


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""
    
    def test_create_comparison(self):
        """Test creating a comparison result."""
        comp = ComparisonResult(
            isotope='Co60',
            experimental_Bq=1e5,
            experimental_unc_Bq=1e4,
            simulated_Bq=1.1e5,
            simulated_unc_Bq=5e3,
            cooling_time='15d',
            material='CNA',
        )
        
        assert comp.isotope == 'Co60'
        assert comp.ratio_CE == pytest.approx(1.1)
        assert comp.difference_percent == pytest.approx(10.0)
    
    def test_ratio_uncertainty_propagation(self):
        """Test that C/E uncertainty is properly propagated."""
        comp = ComparisonResult(
            isotope='Co60',
            experimental_Bq=100.0,
            experimental_unc_Bq=10.0,  # 10%
            simulated_Bq=100.0,
            simulated_unc_Bq=10.0,  # 10%
        )
        
        # Expected relative uncertainty: sqrt(0.1^2 + 0.1^2) = sqrt(0.02) ≈ 0.141
        expected_rel_unc = np.sqrt(0.1**2 + 0.1**2)
        expected_abs_unc = 1.0 * expected_rel_unc
        
        assert comp.ratio_CE == pytest.approx(1.0)
        assert comp.ratio_CE_uncertainty == pytest.approx(expected_abs_unc, rel=0.01)
    
    def test_zero_experimental_handling(self):
        """Test handling of zero experimental value."""
        comp = ComparisonResult(
            isotope='Co60',
            experimental_Bq=0.0,
            experimental_unc_Bq=0.0,
            simulated_Bq=100.0,
            simulated_unc_Bq=10.0,
        )
        
        assert np.isnan(comp.ratio_CE)
        assert np.isnan(comp.difference_percent)


# =============================================================================
# Test Plotting Functions (Output Validation)
# =============================================================================

class TestPlotActivityComparisonBar:
    """Tests for activity comparison bar chart."""
    
    @pytest.fixture
    def sample_comparisons(self):
        """Create sample comparison data."""
        return [
            ComparisonResult('Mn56', 1e6, 1e5, 1.1e6, 5e4, '300s', 'CNA'),
            ComparisonResult('V52', 5e5, 5e4, 4.8e5, 3e4, '300s', 'CNA'),
            ComparisonResult('W187', 2e4, 2e3, 2.2e4, 1e3, '300s', 'CNA'),
        ]
    
    def test_plot_creates_figure(self, sample_comparisons, tmp_path):
        """Test that plotting creates a figure."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, ax = plot_activity_comparison_bar(
            sample_comparisons,
            save_path=tmp_path / 'test.png',
        )
        
        assert fig is not None
        assert ax is not None
        assert (tmp_path / 'test.png').exists()
    
    def test_empty_comparisons_raises(self):
        """Test that empty comparisons raises error."""
        with pytest.raises(ValueError):
            plot_activity_comparison_bar([])


class TestPlotCERatioSummary:
    """Tests for C/E ratio summary plot."""
    
    @pytest.fixture
    def sample_comparisons(self):
        """Create sample comparison data with various C/E values."""
        return [
            ComparisonResult('Mn56', 1e6, 1e5, 1.05e6, 5e4, '300s', 'CNA'),  # C/E = 1.05 (good)
            ComparisonResult('V52', 5e5, 5e4, 7.5e5, 3e4, '300s', 'CNA'),   # C/E = 1.5 (fair)
            ComparisonResult('W187', 2e4, 2e3, 0.5e4, 1e3, '300s', 'CNA'),  # C/E = 0.25 (poor)
        ]
    
    def test_plot_creates_figure(self, sample_comparisons, tmp_path):
        """Test that plotting creates a figure."""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ax = plot_ce_ratio_summary(
            sample_comparisons,
            save_path=tmp_path / 'ce_ratio.png',
        )
        
        assert fig is not None
        assert (tmp_path / 'ce_ratio.png').exists()


class TestPlotDecayCurves:
    """Tests for decay curve plotting."""
    
    @pytest.fixture
    def sample_decay_data(self):
        """Create sample decay curve data."""
        return {
            'Mn56': [
                (300, 1e6, 1e5),
                (7200, 5e5, 5e4),
                (86400, 1e4, 1e3),
            ],
            'Co60': [
                (300, 1e5, 1e4),
                (7200, 9.9e4, 9.9e3),
                (86400, 9.5e4, 9.5e3),
            ],
        }
    
    def test_plot_creates_figure(self, sample_decay_data, tmp_path):
        """Test that plotting creates a figure."""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ax = plot_decay_curves(
            sample_decay_data,
            save_path=tmp_path / 'decay.png',
        )
        
        assert fig is not None
        assert (tmp_path / 'decay.png').exists()


class TestPlotCdRatioAnalysis:
    """Tests for Cd-ratio analysis plotting."""
    
    @pytest.fixture
    def sample_cd_data(self):
        """Create sample Cd-ratio data."""
        return {
            'Co': {'bare_activity': 1e5, 'cd_activity': 4e4, 'cd_ratio': 2.5},
            'Sc': {'bare_activity': 5e4, 'cd_activity': 1e4, 'cd_ratio': 5.0},
            'Cu': {'bare_activity': 8e4, 'cd_activity': 6e4, 'cd_ratio': 1.33},
        }
    
    def test_plot_creates_figure(self, sample_cd_data, tmp_path):
        """Test that plotting creates a figure."""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plot_cd_ratio_analysis(
            sample_cd_data,
            save_path=tmp_path / 'cd_ratio.png',
        )
        
        assert fig is not None
        assert (tmp_path / 'cd_ratio.png').exists()


class TestPlotValidationSummaryTable:
    """Tests for validation summary table."""
    
    @pytest.fixture
    def sample_comparisons(self):
        """Create sample comparison data."""
        return [
            ComparisonResult('Mn56', 1e6, 1e5, 1.1e6, 5e4),
            ComparisonResult('V52', 5e5, 5e4, 5.5e5, 3e4),
            ComparisonResult('W187', 2e4, 2e3, 1.8e4, 1e3),
            ComparisonResult('Co60', 1e5, 1e4, 9e4, 5e3),
        ]
    
    def test_plot_creates_figure(self, sample_comparisons, tmp_path):
        """Test that plotting creates a figure."""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ax = plot_validation_summary_table(
            sample_comparisons,
            save_path=tmp_path / 'summary.png',
        )
        
        assert fig is not None
        assert (tmp_path / 'summary.png').exists()


# =============================================================================
# Test Style and Colors
# =============================================================================

class TestStyleConfiguration:
    """Tests for style configuration."""
    
    def test_material_colors_defined(self):
        """Test that material colors are defined."""
        assert 'CNA' in MATERIAL_COLORS
        assert 'EUROFER97' in MATERIAL_COLORS
    
    def test_cooling_colors_defined(self):
        """Test that cooling time colors are defined."""
        assert '300s' in COOLING_COLORS
        assert '15d' in COOLING_COLORS
    
    def test_source_colors_defined(self):
        """Test that source colors are defined."""
        assert 'experimental' in SOURCE_COLORS
        assert 'alara' in SOURCE_COLORS
    
    def test_apply_thesis_style(self):
        """Test that thesis style can be applied."""
        import matplotlib
        matplotlib.use('Agg')
        
        # Should not raise
        apply_thesis_style()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
