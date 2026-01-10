#!/usr/bin/env python
"""
STAYSL-Class Reporting Demo
============================

Demonstrates the new reporting capabilities for STAYSL-parity output:
- Differential flux tables
- Correlation matrices
- Spectral-averaged reaction rates
- Stepwise spectrum plots
- Full unfolding reports
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# FluxForge imports
from fluxforge.reporting import (
    DifferentialFluxTable,
    ReactionRateTable,
    CorrelationMatrix,
    StepwiseSpectrum,
    UnfoldingReport,
    create_unfolding_report,
)
from fluxforge.solvers.gls import gls_adjust


def create_example_data():
    """Generate example flux and cross section data."""
    # 10-group energy structure (thermal to 20 MeV)
    energy_bounds_eV = np.array([
        1e-5, 0.4, 0.5, 1.0, 10.0, 1e3, 1e5, 
        1e6, 2e6, 10e6, 20e6
    ])
    n_groups = len(energy_bounds_eV) - 1
    
    # Prior flux (1/E spectrum-like)
    prior_flux = np.zeros(n_groups)
    for g in range(n_groups):
        E_mid = np.sqrt(energy_bounds_eV[g] * energy_bounds_eV[g+1])
        lethargy = np.log(energy_bounds_eV[g+1] / energy_bounds_eV[g])
        prior_flux[g] = 1e10 * lethargy / (1 + E_mid / 1e6)
    
    # Prior uncertainty (25% relative)
    prior_uncertainty = 0.25 * prior_flux
    prior_covariance = np.diag(prior_uncertainty**2)
    
    # Example reactions (simplified cross sections)
    cross_sections = {
        "Au197_ng": np.array([100, 80, 50, 20, 5, 1, 0.1, 0.01, 0.001, 0.0001]),
        "Fe58_ng": np.array([2.0, 1.5, 1.0, 0.5, 0.2, 0.05, 0.01, 0.001, 0.0001, 0.00001]),
        "In115_ng": np.array([200, 150, 100, 50, 20, 5, 1, 0.1, 0.01, 0.001]),
        "Ni58_np": np.array([0, 0, 0, 0, 0.1, 1, 10, 50, 100, 150]),
        "Al27_na": np.array([0, 0, 0, 0, 0, 0.1, 5, 30, 80, 120]),
    }
    
    return energy_bounds_eV, prior_flux, prior_uncertainty, prior_covariance, cross_sections


def demo_flux_table():
    """Demonstrate DifferentialFluxTable generation."""
    print("\n" + "="*70)
    print("DEMO: Differential Flux Table")
    print("="*70)
    
    energy_bounds_eV, prior_flux, prior_uncertainty, _, _ = create_example_data()
    
    # Create flux table
    table = DifferentialFluxTable.from_spectrum(
        prior_flux,
        energy_bounds_eV,
        prior_uncertainty,
        label="Prior Flux (1/E Approximation)",
    )
    
    # Print in different formats
    print("\n--- STAYSL Format ---")
    print(table.to_text("staysl"))
    
    print("\n--- Markdown Format ---")
    print(table.to_text("markdown"))
    
    print("\n--- CSV Format ---")
    print(table.to_text("csv"))
    
    return table


def demo_reaction_rates():
    """Demonstrate ReactionRateTable calculation."""
    print("\n" + "="*70)
    print("DEMO: Spectral-Averaged Reaction Rates")
    print("="*70)
    
    energy_bounds_eV, prior_flux, prior_uncertainty, _, cross_sections = create_example_data()
    
    # Calculate reaction rates
    table = ReactionRateTable.from_cross_sections(
        prior_flux,
        prior_uncertainty,
        cross_sections,
        energy_bounds_eV,
        flux_label="Prior Flux",
    )
    
    print("\n--- Reaction Rate Table ---")
    print(table.to_text("staysl"))
    
    return table


def demo_correlation_matrix():
    """Demonstrate CorrelationMatrix generation."""
    print("\n" + "="*70)
    print("DEMO: Correlation Matrix")
    print("="*70)
    
    # Create example covariance matrix with correlations
    n = 5
    labels = ["Au197", "Fe58", "In115", "Ni58", "Al27"]
    
    # Build a covariance with off-diagonal correlations
    variances = np.array([1e-28, 1e-32, 1e-27, 1e-31, 1e-30])
    correlation_values = [
        [1.0, 0.3, 0.5, 0.0, 0.0],
        [0.3, 1.0, 0.2, 0.0, 0.0],
        [0.5, 0.2, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 0.4, 1.0],
    ]
    
    # Convert to covariance
    sigma = np.sqrt(variances)
    cov = np.outer(sigma, sigma) * np.array(correlation_values)
    
    # Create correlation matrix
    corr = CorrelationMatrix.from_covariance(cov, labels, "Dosimetry Input")
    
    print("\n--- Correlation Matrix ---")
    print(corr.to_text("staysl"))
    
    return corr


def demo_stepwise_spectrum():
    """Demonstrate StepwiseSpectrum for plotting."""
    print("\n" + "="*70)
    print("DEMO: Stepwise Spectrum")
    print("="*70)
    
    energy_bounds_eV, prior_flux, _, _, _ = create_example_data()
    
    # Create stepwise spectrum
    step = StepwiseSpectrum.from_histogram(prior_flux, energy_bounds_eV, "Prior")
    
    print(f"\nStepwise spectrum has {len(step.energy_eV)} points for step plotting")
    print("\n--- CSV Export ---")
    print(step.to_csv()[:500] + "...")
    
    # Plot if matplotlib available
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(step.energy_eV, step.flux, 'b-', linewidth=1.5, label=step.label)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Flux (n/cm²/s)")
        ax.set_title("Stepwise Spectrum Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / "stepwise_spectrum.png", dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {output_dir / 'stepwise_spectrum.png'}")
        plt.close(fig)
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    return step


def demo_full_report():
    """Demonstrate complete UnfoldingReport generation."""
    print("\n" + "="*70)
    print("DEMO: Full Unfolding Report")
    print("="*70)
    
    # Get example data
    energy_bounds_eV, prior_flux, prior_uncertainty, prior_covariance, cross_sections = create_example_data()
    n_groups = len(prior_flux)
    n_reactions = len(cross_sections)
    
    # Build response matrix
    response = np.array([cross_sections[rxn] for rxn in cross_sections.keys()])
    reaction_ids = list(cross_sections.keys())
    
    # Simulate measurements from prior flux
    lethargy = np.log(energy_bounds_eV[1:] / energy_bounds_eV[:-1])
    true_rates = response @ (prior_flux * lethargy)
    
    # Add noise for "measured" rates
    meas_unc = 0.05 * true_rates  # 5% uncertainty
    np.random.seed(42)
    measured_rates = true_rates + np.random.normal(0, meas_unc)
    meas_cov = np.diag(meas_unc**2)
    
    # Perform GLS adjustment
    result = gls_adjust(
        prior_flux=prior_flux,
        prior_cov=prior_covariance,
        measured_rates=measured_rates,
        measurement_cov=meas_cov,
        response_matrix=response,
    )
    
    adjusted_flux = result.flux
    adjusted_uncertainty = np.sqrt(np.diag(result.covariance))
    
    # Create full report
    report = create_unfolding_report(
        prior_flux=prior_flux,
        adjusted_flux=adjusted_flux,
        energy_bounds_eV=energy_bounds_eV,
        prior_uncertainty=prior_uncertainty,
        adjusted_uncertainty=adjusted_uncertainty,
        adjusted_covariance=result.covariance,
        dosimetry_covariance=meas_cov,
        reaction_labels=reaction_ids,
        cross_sections=cross_sections,
        chi_squared=result.chi2,
        degrees_of_freedom=n_reactions,
    )
    
    # Generate reports in different formats
    print("\n--- Full Report (STAYSL Format) ---")
    full_report = report.generate_full_report("staysl")
    print(full_report[:3000] + "...")  # Truncate for demo
    
    # Save report files
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    report.save(output_dir / "unfolding_report.txt", "staysl")
    report.save(output_dir / "unfolding_report.md", "markdown")
    
    print(f"\nReports saved to:")
    print(f"  - {output_dir / 'unfolding_report.txt'}")
    print(f"  - {output_dir / 'unfolding_report.md'}")
    
    # Create comparison plot
    try:
        prior_step, adj_step = report.get_stepwise_spectra()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(prior_step.energy_eV, prior_step.flux, 'b-', 
                  linewidth=1.5, label=prior_step.label)
        ax.loglog(adj_step.energy_eV, adj_step.flux, 'r--', 
                  linewidth=1.5, label=adj_step.label)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Flux (n/cm²/s)")
        ax.set_title("Prior vs Adjusted Flux")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.savefig(output_dir / "prior_vs_adjusted.png", dpi=150, bbox_inches="tight")
        print(f"  - {output_dir / 'prior_vs_adjusted.png'}")
        plt.close(fig)
    except Exception as e:
        print(f"\nCould not create comparison plot: {e}")
    
    return report


def main():
    """Run all demos."""
    print("FluxForge STAYSL-Class Reporting Demo")
    print("=====================================\n")
    
    # Run demos
    demo_flux_table()
    demo_reaction_rates()
    demo_correlation_matrix()
    demo_stepwise_spectrum()
    demo_full_report()
    
    print("\n" + "="*70)
    print("All demos completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
