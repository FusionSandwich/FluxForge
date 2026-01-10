#!/usr/bin/env python3
"""
Generate Publication-Quality Plots from Experimental Data

This script produces the full set of plots required by the FluxForge capability
specification, using the experimental Fe-Cd-RAFM-1 data:

1. Unfolded spectrum vs energy with uncertainty bands (Fig 10 style)
2. Prior vs posterior spectrum overlay
3. Reaction rate residuals/pulls per monitor
4. Covariance and correlation matrix heatmaps
5. Predicted vs measured reaction rates (parity plot)
6. Solver comparison (GLS, GRAVEL, MLEM)
7. Response matrix visualization

Usage:
    python examples/generate_plots.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add FluxForge to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    print("ERROR: matplotlib is required for plotting")
    sys.exit(1)

# Import FluxForge modules
from fluxforge.core.response import (
    EnergyGroupStructure,
    ReactionCrossSection,
    build_response_matrix,
)
from fluxforge.physics.activation import (
    GammaLineMeasurement,
    IrradiationSegment,
    weighted_activity,
    reaction_rate_from_activity,
)
from fluxforge.solvers.gls import gls_adjust
from fluxforge.solvers.iterative import gravel, mlem, gradient_descent


# =============================================================================
# Plot Style Configuration
# =============================================================================

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "prior": "#1f77b4",       # Blue
    "gls": "#d62728",         # Red
    "gravel": "#ff7f0e",      # Orange
    "mlem": "#2ca02c",        # Green
    "experiment": "#9467bd",   # Purple
    "model": "#17becf",       # Cyan
}


# =============================================================================
# Data Loading
# =============================================================================

def load_example_data() -> Dict:
    """Load all Fe-Cd-RAFM-1 experimental data."""
    example_dir = repo_root / "src" / "fluxforge" / "examples" / "fe_cd_rafm_1"
    
    return {
        "boundaries": json.loads((example_dir / "boundaries.json").read_text()),
        "cross_sections": json.loads((example_dir / "cross_sections.json").read_text()),
        "number_densities": json.loads((example_dir / "number_densities.json").read_text()),
        "measurements": json.loads((example_dir / "measurements.json").read_text()),
        "prior_flux": json.loads((example_dir / "prior_flux.json").read_text()),
    }


def run_unfolding(data: Dict) -> Tuple[Dict, Dict, Dict, List, List]:
    """Run all three solvers and return results."""
    boundaries = data["boundaries"]
    cross_sections = data["cross_sections"]
    number_densities = data["number_densities"]
    measurements = data["measurements"]
    prior_flux = data["prior_flux"]
    
    # Build response matrix
    groups = EnergyGroupStructure(boundaries)
    reactions = [
        ReactionCrossSection(reaction_id=r_id, sigma_g=sigma)
        for r_id, sigma in cross_sections.items()
    ]
    nd_values = [number_densities[rx.reaction_id] for rx in reactions]
    response = build_response_matrix(reactions, groups, nd_values)
    
    # Calculate reaction rates
    segments = [IrradiationSegment(**seg) for seg in measurements["segments"]]
    measured_rates = []
    rate_uncertainties = []
    
    for reaction in measurements["reactions"]:
        gamma_lines = [GammaLineMeasurement(**gl) for gl in reaction["gamma_lines"]]
        activity, _ = weighted_activity(gamma_lines)
        rate_estimate = reaction_rate_from_activity(activity, segments, reaction["half_life_s"])
        measured_rates.append(rate_estimate.rate)
        rate_uncertainties.append(rate_estimate.uncertainty)
    
    # Build covariance matrices
    measurement_cov = [
        [(rate_uncertainties[i] ** 2) if i == j else 0.0
         for j in range(len(measured_rates))]
        for i in range(len(measured_rates))
    ]
    prior_cov = [
        [(0.25 * val) ** 2 if i == j else 0.0
         for j, val in enumerate(prior_flux)]
        for i, _ in enumerate(prior_flux)
    ]
    
    # Run solvers
    gls_result = gls_adjust(response.matrix, measured_rates, measurement_cov, prior_flux, prior_cov)
    gravel_result = gravel(response.matrix, measured_rates, initial_flux=prior_flux, max_iters=500)
    mlem_result = mlem(response.matrix, measured_rates, initial_flux=prior_flux, max_iters=200)
    gd_result = gradient_descent(response.matrix, measured_rates, initial_flux=prior_flux, max_iters=5000)
    
    return (
        {"flux": gls_result.flux, "covariance": gls_result.covariance, 
         "chi2": gls_result.chi2, "residuals": gls_result.residuals},
        {"flux": gravel_result.flux, "history": gravel_result.history},
        {"flux": mlem_result.flux, "history": mlem_result.history},
        {"flux": gd_result.flux, "history": gd_result.history},
        measured_rates,
        response.matrix,
    )


# =============================================================================
# Plot 1: Unfolded Spectrum with Uncertainty Bands
# =============================================================================

def plot_spectrum_uncertainty_bands(boundaries, prior, gls, output_dir):
    """
    Fig 10 style: Unfolded spectrum with 1-sigma uncertainty bands.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    n_groups = len(prior)
    x = np.arange(n_groups)
    
    # Calculate midpoints for plotting
    midpoints = [(boundaries[g] + boundaries[g+1]) / 2 for g in range(n_groups)]
    widths = [boundaries[g+1] - boundaries[g] for g in range(n_groups)]
    
    # Prior spectrum
    ax.bar(x - 0.2, prior, width=0.35, label='Prior (Initial Guess)', 
           color=COLORS['prior'], alpha=0.7, edgecolor='black')
    
    # GLS posterior with error bars
    gls_flux = gls["flux"]
    gls_unc = [np.sqrt(gls["covariance"][g][g]) for g in range(n_groups)]
    
    ax.bar(x + 0.2, gls_flux, width=0.35, label='GLS Posterior', 
           color=COLORS['gls'], alpha=0.7, edgecolor='black')
    ax.errorbar(x + 0.2, gls_flux, yerr=gls_unc, fmt='none', 
                color='black', capsize=5, capthick=1.5, linewidth=1.5)
    
    # Labels and styling
    ax.set_xlabel('Energy Group')
    ax.set_ylabel('Flux (arbitrary units)')
    ax.set_title('Neutron Flux Spectrum: Prior vs GLS Adjusted\n(Fe-Cd-RAFM-1 Experimental Data)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{midpoints[g]/1e3:.1f} keV' for g in range(n_groups)], rotation=45)
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    
    # Add energy range annotations
    for g in range(n_groups):
        ax.annotate(f'{boundaries[g]:.1f}-{boundaries[g+1]:.0f} eV',
                   (x[g], max(prior[g], gls_flux[g]) * 1.05),
                   ha='center', fontsize=9, rotation=45)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'spectrum_with_uncertainty.png')
    fig.savefig(output_dir / 'spectrum_with_uncertainty.pdf')
    plt.close(fig)
    print(f"  Saved: spectrum_with_uncertainty.png/pdf")


# =============================================================================
# Plot 2: Prior vs Posterior Overlay (Log Scale)
# =============================================================================

def plot_prior_posterior_overlay(boundaries, prior, gls, gravel, mlem, output_dir):
    """
    Prior vs posterior spectrum overlay with all three methods.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    n_groups = len(prior)
    midpoints = np.array([(boundaries[g] + boundaries[g+1]) / 2 for g in range(n_groups)])
    
    # Plot as step functions
    for g in range(n_groups):
        e_low, e_high = boundaries[g], boundaries[g+1]
        
        # Prior
        ax.plot([e_low, e_high], [prior[g], prior[g]], 
                color=COLORS['prior'], linewidth=2, label='Prior' if g == 0 else '')
        
        # GLS
        if gls['flux'][g] > 0:
            ax.plot([e_low, e_high], [gls['flux'][g], gls['flux'][g]], 
                    color=COLORS['gls'], linewidth=2, linestyle='--', 
                    label='GLS Adjusted' if g == 0 else '')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Flux (arbitrary units)')
    ax.set_title('Prior vs Adjusted Spectrum (Log-Log Scale)\n(Fe-Cd-RAFM-1 Experimental Data)')
    ax.legend(loc='best')
    ax.set_xlim(boundaries[0] * 0.5, boundaries[-1] * 2)
    ax.set_ylim(bottom=1)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'prior_posterior_overlay.png')
    fig.savefig(output_dir / 'prior_posterior_overlay.pdf')
    plt.close(fig)
    print(f"  Saved: prior_posterior_overlay.png/pdf")


# =============================================================================
# Plot 3: Residuals / Pulls Per Monitor
# =============================================================================

def plot_residuals(gls, reaction_ids, output_dir):
    """
    Reaction rate residuals/pulls per monitor with chi-square contributions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    residuals = gls["residuals"]
    chi2 = gls["chi2"]
    n_monitors = len(residuals)
    
    # Left: Residual bar chart
    x = np.arange(n_monitors)
    colors = ['green' if abs(r) < 2 else 'orange' if abs(r) < 3 else 'red' for r in residuals]
    ax1.bar(x, residuals, color=colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=1)
    ax1.axhline(y=2, color='orange', linestyle='--', linewidth=1, label='±2σ')
    ax1.axhline(y=-2, color='orange', linestyle='--', linewidth=1)
    ax1.axhline(y=3, color='red', linestyle='--', linewidth=1, label='±3σ')
    ax1.axhline(y=-3, color='red', linestyle='--', linewidth=1)
    
    ax1.set_xlabel('Monitor Reaction')
    ax1.set_ylabel('Normalized Residual (σ)')
    ax1.set_title(f'Residuals per Monitor (χ² = {chi2:.2f})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(reaction_ids, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    
    # Right: Chi-square contribution
    chi2_contrib = [r**2 for r in residuals]
    ax2.bar(x, chi2_contrib, color=COLORS['experiment'], edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Monitor Reaction')
    ax2.set_ylabel('χ² Contribution')
    ax2.set_title('Chi-Square Contribution per Monitor')
    ax2.set_xticks(x)
    ax2.set_xticklabels(reaction_ids, rotation=45, ha='right')
    
    # Add total chi2 annotation
    ax2.axhline(y=chi2/n_monitors, color='red', linestyle='--', linewidth=1, 
                label=f'Average = {chi2/n_monitors:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'residuals_pulls.png')
    fig.savefig(output_dir / 'residuals_pulls.pdf')
    plt.close(fig)
    print(f"  Saved: residuals_pulls.png/pdf")


# =============================================================================
# Plot 4: Covariance and Correlation Matrix Heatmaps
# =============================================================================

def plot_covariance_correlation(gls, boundaries, output_dir):
    """
    Posterior covariance and correlation matrix heatmaps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cov = np.array(gls["covariance"])
    n = len(cov)
    
    # Correlation matrix
    std = np.sqrt(np.diag(cov))
    # Avoid division by zero
    std = np.where(std > 0, std, 1.0)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    
    # Energy labels
    labels = [f'{boundaries[g]:.1f}-{boundaries[g+1]:.0f} eV' for g in range(n)]
    
    # Left: Covariance
    im1 = ax1.imshow(cov, cmap='RdBu_r', aspect='auto')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_yticklabels(labels)
    ax1.set_title('Posterior Covariance Matrix')
    plt.colorbar(im1, ax=ax1, label='Covariance')
    
    # Annotate values
    for i in range(n):
        for j in range(n):
            text = f'{cov[i,j]:.1e}'
            ax1.text(j, i, text, ha='center', va='center', fontsize=8)
    
    # Right: Correlation
    im2 = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_yticklabels(labels)
    ax2.set_title('Posterior Correlation Matrix')
    plt.colorbar(im2, ax=ax2, label='Correlation')
    
    # Annotate values
    for i in range(n):
        for j in range(n):
            text = f'{corr[i,j]:.2f}'
            ax2.text(j, i, text, ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'covariance_correlation.png')
    fig.savefig(output_dir / 'covariance_correlation.pdf')
    plt.close(fig)
    print(f"  Saved: covariance_correlation.png/pdf")


# =============================================================================
# Plot 5: Predicted vs Measured (Parity Plot)
# =============================================================================

def plot_parity(measured_rates, gls, response_matrix, output_dir):
    """
    Predicted vs measured reaction rates (parity plot) with uncertainty bars.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate predicted rates from GLS flux
    flux = gls["flux"]
    predicted = []
    for i in range(len(measured_rates)):
        pred = sum(response_matrix[i][g] * flux[g] for g in range(len(flux)))
        predicted.append(pred)
    
    # Measurement uncertainties (assume 5%)
    measured_unc = [0.05 * m for m in measured_rates]
    
    # Plot data
    ax.errorbar(measured_rates, predicted, xerr=measured_unc, 
                fmt='o', markersize=10, color=COLORS['experiment'],
                capsize=5, capthick=2, linewidth=2, label='Monitor Reactions')
    
    # Perfect agreement line
    max_val = max(max(measured_rates), max(predicted)) * 1.1
    min_val = min(min(measured_rates), min(predicted)) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Agreement')
    
    # ±10% bands
    ax.fill_between([min_val, max_val], [0.9*min_val, 0.9*max_val], 
                   [1.1*min_val, 1.1*max_val], alpha=0.2, color='gray', label='±10%')
    
    ax.set_xlabel('Measured Reaction Rate (reactions/s)')
    ax.set_ylabel('Predicted Reaction Rate (reactions/s)')
    ax.set_title('Predicted vs Measured Reaction Rates\n(Fe-Cd-RAFM-1 Experimental Data)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'parity_plot.png')
    fig.savefig(output_dir / 'parity_plot.pdf')
    plt.close(fig)
    print(f"  Saved: parity_plot.png/pdf")


# =============================================================================
# Plot 6: Solver Comparison
# =============================================================================

def plot_solver_comparison(boundaries, prior, gls, gravel, mlem, gd, output_dir):
    """
    Compare GLS, GRAVEL, MLEM, and Gradient Descent solutions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    n_groups = len(prior)
    x = np.arange(n_groups)
    width = 0.16
    
    # Left: Flux comparison
    ax1.bar(x - 2*width, prior, width, label='Prior', color=COLORS['prior'], alpha=0.7)
    ax1.bar(x - width, gls['flux'], width, label='GLS', color=COLORS['gls'], alpha=0.7)
    ax1.bar(x, gravel['flux'], width, label='GRAVEL', color=COLORS['gravel'], alpha=0.7)
    ax1.bar(x + width, mlem['flux'], width, label='MLEM', color=COLORS['mlem'], alpha=0.7)
    ax1.bar(x + 2*width, gd['flux'], width, label='GD', color=COLORS['model'], alpha=0.7)
    
    ax1.set_xlabel('Energy Group')
    ax1.set_ylabel('Flux (arbitrary units)')
    ax1.set_title('Solver Comparison: Flux by Energy Group')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Group {g}' for g in range(n_groups)])
    ax1.legend(loc='best')
    ax1.set_yscale('symlog', linthresh=1)
    
    # Right: Relative change from prior
    gls_change = [(f - p) / p * 100 if p > 0 else 0 for f, p in zip(gls['flux'], prior)]
    gravel_change = [(f - p) / p * 100 if p > 0 else 0 for f, p in zip(gravel['flux'], prior)]
    mlem_change = [(f - p) / p * 100 if p > 0 else 0 for f, p in zip(mlem['flux'], prior)]
    gd_change = [(f - p) / p * 100 if p > 0 else 0 for f, p in zip(gd['flux'], prior)]
    
    width_bar = 0.2
    ax2.bar(x - 1.5*width_bar, gls_change, width_bar, label='GLS', color=COLORS['gls'], alpha=0.7)
    ax2.bar(x - 0.5*width_bar, gravel_change, width_bar, label='GRAVEL', color=COLORS['gravel'], alpha=0.7)
    ax2.bar(x + 0.5*width_bar, mlem_change, width_bar, label='MLEM', color=COLORS['mlem'], alpha=0.7)
    ax2.bar(x + 1.5*width_bar, gd_change, width_bar, label='GD', color=COLORS['model'], alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1)
    
    ax2.set_xlabel('Energy Group')
    ax2.set_ylabel('Change from Prior (%)')
    ax2.set_title('Relative Adjustment by Solver')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Group {g}' for g in range(n_groups)])
    ax2.legend(loc='best')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'solver_comparison.png')
    fig.savefig(output_dir / 'solver_comparison.pdf')
    plt.close(fig)
    print(f"  Saved: solver_comparison.png/pdf")


# =============================================================================
# Plot 7: Response Matrix Visualization
# =============================================================================

def plot_response_matrix(response_matrix, boundaries, reaction_ids, output_dir):
    """
    Response matrix R[i,g] visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    R = np.array(response_matrix)
    n_reactions, n_groups = R.shape
    
    # Normalize for visualization
    R_norm = np.log10(R + 1e-30)  # Log scale
    
    im = ax.imshow(R_norm, cmap='viridis', aspect='auto')
    
    # Labels
    group_labels = [f'{boundaries[g]:.1f}-{boundaries[g+1]:.0f} eV' for g in range(n_groups)]
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    ax.set_yticks(range(n_reactions))
    ax.set_yticklabels(reaction_ids)
    
    ax.set_xlabel('Energy Group')
    ax.set_ylabel('Monitor Reaction')
    ax.set_title('Response Matrix R[reaction, group]\n(log₁₀ scale)')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Response)')
    
    # Annotate with actual values
    for i in range(n_reactions):
        for j in range(n_groups):
            text = f'{R[i,j]:.1e}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color='white')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'response_matrix.png')
    fig.savefig(output_dir / 'response_matrix.pdf')
    plt.close(fig)
    print(f"  Saved: response_matrix.png/pdf")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("FLUXFORGE: Generating Publication-Quality Plots")
    print("=" * 80)
    
    # Create output directory
    output_dir = repo_root / "output" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data and run unfolding
    print("\nLoading experimental data...")
    data = load_example_data()
    
    print("Running unfolding analysis...")
    gls, gravel_res, mlem_res, gd_res, measured_rates, response_matrix = run_unfolding(data)
    
    boundaries = data["boundaries"]
    prior = data["prior_flux"]
    reaction_ids = list(data["cross_sections"].keys())
    
    print("\nGenerating plots...")
    
    # Generate all plots
    plot_spectrum_uncertainty_bands(boundaries, prior, gls, output_dir)
    plot_prior_posterior_overlay(boundaries, prior, gls, gravel_res, mlem_res, output_dir)
    plot_residuals(gls, reaction_ids, output_dir)
    plot_covariance_correlation(gls, boundaries, output_dir)
    plot_parity(measured_rates, gls, response_matrix, output_dir)
    plot_solver_comparison(boundaries, prior, gls, gravel_res, mlem_res, gd_res, output_dir)
    plot_response_matrix(response_matrix, boundaries, reaction_ids, output_dir)
    
    print("\n" + "=" * 80)
    print("PLOT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
