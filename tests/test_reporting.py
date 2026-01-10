"""Tests for the reporting module (STAYSL-class output)."""

import numpy as np
import pytest

from fluxforge.reporting import (
    FluxTableEntry,
    DifferentialFluxTable,
    SpectralReactionRate,
    ReactionRateTable,
    CorrelationMatrix,
    StepwiseSpectrum,
    UnfoldingReport,
    create_unfolding_report,
)


class TestDifferentialFluxTable:
    """Tests for DifferentialFluxTable."""

    def test_from_spectrum(self):
        flux = np.array([1e10, 5e9, 1e8])
        energy_bounds = np.array([1e-3, 1e-1, 1e1, 1e3])  # eV
        
        table = DifferentialFluxTable.from_spectrum(
            flux, energy_bounds, label="Test"
        )
        
        assert len(table.entries) == 3
        assert table.entries[0].group == 1
        assert table.entries[0].energy_low_eV == pytest.approx(1e-3)
        assert table.entries[0].flux == pytest.approx(1e10)
        assert table.label == "Test"
        assert table.total_fluence > 0

    def test_to_staysl_format(self):
        flux = np.array([1e10, 1e9])
        energy_bounds = np.array([1e-3, 1.0, 1e6])
        uncertainty = np.array([1e9, 1e8])
        
        table = DifferentialFluxTable.from_spectrum(
            flux, energy_bounds, uncertainty, label="Prior"
        )
        
        text = table.to_text("staysl")
        assert "DIFFERENTIAL FLUX TABLE" in text
        assert "Prior" in text
        assert "E_low" in text

    def test_to_csv_format(self):
        flux = np.array([1e10])
        energy_bounds = np.array([1.0, 10.0])
        
        table = DifferentialFluxTable.from_spectrum(flux, energy_bounds)
        csv = table.to_text("csv")
        
        assert "group,E_low_eV" in csv
        lines = csv.strip().split("\n")
        assert len(lines) == 2  # Header + 1 data row

    def test_to_markdown_format(self):
        flux = np.array([1e10])
        energy_bounds = np.array([1.0, 10.0])
        
        table = DifferentialFluxTable.from_spectrum(
            flux, energy_bounds, label="Test"
        )
        md = table.to_text("markdown")
        
        assert "| Grp |" in md
        assert "**Total fluence:**" in md

    def test_energy_weighted_average(self):
        # Uniform flux in log space
        flux = np.array([1.0, 1.0])
        energy_bounds = np.array([1.0, 10.0, 100.0])
        
        table = DifferentialFluxTable.from_spectrum(flux, energy_bounds)
        
        # Energy-weighted average should be between geometric means
        E_mid_1 = np.sqrt(1.0 * 10.0)
        E_mid_2 = np.sqrt(10.0 * 100.0)
        assert E_mid_1 < table.energy_weighted_avg < E_mid_2


class TestReactionRateTable:
    """Tests for ReactionRateTable."""

    def test_from_cross_sections(self):
        flux = np.array([1e10, 1e9])
        flux_unc = np.array([1e9, 1e8])
        energy_bounds = np.array([1e-3, 1.0, 1e6])
        
        cross_sections = {
            "Au197_ng": np.array([100.0, 0.1]),  # barns
            "Fe58_ng": np.array([1.0, 0.01]),
        }
        
        table = ReactionRateTable.from_cross_sections(
            flux, flux_unc, cross_sections, energy_bounds, "Adjusted"
        )
        
        assert len(table.rates) == 2
        assert table.flux_label == "Adjusted"
        
        # Find Au197 rate
        au_rate = next(r for r in table.rates if r.reaction_id == "Au197_ng")
        assert au_rate.rate > 0
        assert au_rate.rate_uncertainty > 0

    def test_to_staysl_format(self):
        rates = [
            SpectralReactionRate("Au197_ng", 1e15, 1e14, 50.0),
            SpectralReactionRate("Fe58_ng", 1e13, 1e12, 0.5),
        ]
        table = ReactionRateTable(rates, "Adjusted")
        
        text = table.to_text("staysl")
        assert "SPECTRAL-AVERAGED REACTION RATES" in text
        assert "Au197_ng" in text


class TestCorrelationMatrix:
    """Tests for CorrelationMatrix."""

    def test_from_covariance(self):
        # Create a simple covariance matrix
        cov = np.array([
            [1.0, 0.5],
            [0.5, 4.0],
        ])
        labels = ["A", "B"]
        
        corr = CorrelationMatrix.from_covariance(cov, labels, "Test")
        
        # Diagonal should be 1.0
        assert corr.matrix[0, 0] == pytest.approx(1.0)
        assert corr.matrix[1, 1] == pytest.approx(1.0)
        
        # Off-diagonal: 0.5 / sqrt(1*4) = 0.25
        assert corr.matrix[0, 1] == pytest.approx(0.25)
        assert corr.matrix[1, 0] == pytest.approx(0.25)

    def test_to_staysl_format(self):
        cov = np.eye(3)
        labels = ["G1", "G2", "G3"]
        
        corr = CorrelationMatrix.from_covariance(cov, labels, "Test")
        text = corr.to_text("staysl")
        
        assert "CORRELATION MATRIX" in text
        assert "1.000" in text

    def test_to_csv_format(self):
        cov = np.eye(2)
        labels = ["A", "B"]
        
        corr = CorrelationMatrix.from_covariance(cov, labels)
        csv = corr.to_text("csv")
        
        lines = csv.strip().split("\n")
        assert len(lines) == 3  # Header + 2 data rows


class TestStepwiseSpectrum:
    """Tests for StepwiseSpectrum."""

    def test_from_histogram(self):
        flux = np.array([1e10, 1e9])
        energy_bounds = np.array([1.0, 10.0, 100.0])
        
        step = StepwiseSpectrum.from_histogram(flux, energy_bounds, "Test")
        
        # Should have 2*n points for step plot
        assert len(step.energy_eV) == 4
        assert len(step.flux) == 4
        
        # Check step structure
        assert step.energy_eV[0] == pytest.approx(1.0)
        assert step.energy_eV[1] == pytest.approx(10.0)
        assert step.flux[0] == pytest.approx(1e10)
        assert step.flux[1] == pytest.approx(1e10)

    def test_to_csv(self):
        flux = np.array([1e10])
        energy_bounds = np.array([1.0, 10.0])
        
        step = StepwiseSpectrum.from_histogram(flux, energy_bounds)
        csv = step.to_csv()
        
        assert "energy_eV,flux" in csv
        lines = csv.strip().split("\n")
        assert len(lines) == 3  # Header + 2 points


class TestUnfoldingReport:
    """Tests for UnfoldingReport."""

    def test_create_unfolding_report(self):
        prior = np.array([1e10, 5e9, 1e8])
        adjusted = np.array([1.1e10, 4.8e9, 1.05e8])
        energy_bounds = np.array([1e-3, 1e-1, 1e1, 1e3])
        prior_unc = prior * 0.1
        adj_unc = adjusted * 0.05
        
        report = create_unfolding_report(
            prior, adjusted, energy_bounds,
            prior_uncertainty=prior_unc,
            adjusted_uncertainty=adj_unc,
            chi_squared=2.5,
            degrees_of_freedom=2,
        )
        
        assert report.chi_squared == pytest.approx(2.5)
        assert report.degrees_of_freedom == 2
        assert len(report.prior_flux_table.entries) == 3
        assert len(report.adjusted_flux_table.entries) == 3

    def test_generate_full_report_staysl(self):
        prior = np.array([1e10, 1e9])
        adjusted = np.array([1.1e10, 0.9e9])
        energy_bounds = np.array([1.0, 10.0, 100.0])
        
        report = create_unfolding_report(
            prior, adjusted, energy_bounds,
            chi_squared=1.5, degrees_of_freedom=1
        )
        
        text = report.generate_full_report("staysl")
        
        assert "FLUXFORGE SPECTRAL ADJUSTMENT REPORT" in text
        assert "Chi-squared" in text
        assert "Prior Flux" in text
        assert "Adjusted Flux" in text

    def test_generate_full_report_markdown(self):
        prior = np.array([1e10])
        adjusted = np.array([1.1e10])
        energy_bounds = np.array([1.0, 10.0])
        
        report = create_unfolding_report(prior, adjusted, energy_bounds)
        
        md = report.generate_full_report("markdown")
        assert "# FluxForge Spectral Adjustment Report" in md

    def test_get_stepwise_spectra(self):
        prior = np.array([1e10, 1e9])
        adjusted = np.array([1.1e10, 0.9e9])
        energy_bounds = np.array([1.0, 10.0, 100.0])
        
        report = create_unfolding_report(prior, adjusted, energy_bounds)
        
        prior_step, adj_step = report.get_stepwise_spectra()
        
        assert prior_step.label == "Prior"
        assert adj_step.label == "Adjusted"
        assert len(prior_step.energy_eV) == 4

    def test_with_correlation_matrices(self):
        prior = np.array([1e10, 1e9])
        adjusted = np.array([1.1e10, 0.9e9])
        energy_bounds = np.array([1.0, 10.0, 100.0])
        
        adj_cov = np.array([[1e18, 1e17], [1e17, 1e16]])
        dos_cov = np.array([[1e6, 1e5], [1e5, 1e6]])
        
        report = create_unfolding_report(
            prior, adjusted, energy_bounds,
            adjusted_covariance=adj_cov,
            dosimetry_covariance=dos_cov,
            reaction_labels=["Au197", "Fe58"],
        )
        
        assert report.input_correlation is not None
        assert report.output_correlation is not None
        
        text = report.generate_full_report("staysl")
        assert "CORRELATION MATRIX" in text
