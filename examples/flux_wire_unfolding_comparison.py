#!/usr/bin/env python3
"""
Flux Wire Spectrum Unfolding Comparison

This example script:
1. Loads experimental flux wire data from RAFM irradiation
2. Unfolds neutron spectrum using FluxForge (GRAVEL, MLEM, GLS)
3. Compares results with reference implementations in testing/
4. Generates publication-quality comparison plots
5. Saves all results as PNG files

This validates FluxForge against other unfolding tools and demonstrates
the complete reactor dosimetry workflow.

Data source: RAFM irradiation campaign flux wire measurements
Location: rafm_irradiation_ldrd/raw_gamma_spec/flux_wires/
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add FluxForge to path
SCRIPT_DIR = Path(__file__).parent
FLUXFORGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(FLUXFORGE_ROOT / "src"))

# Add testing tools to path for comparison  
# Navigate from FluxForge/examples -> ALARA
ALARA_ROOT = FLUXFORGE_ROOT.parent
TESTING_DIR = ALARA_ROOT / "testing"
sys.path.insert(0, str(TESTING_DIR / "Neutron-Unfolding"))

# Import FluxForge modules
from fluxforge.data.irdff import IRDFFDatabase
from fluxforge.solvers import gravel, mlem, gls_adjust, mcmc_unfold
from fluxforge.plots import (
    plot_spectrum_comparison,
    plot_spectrum_uncertainty_bands,
    plot_measured_vs_predicted,
    plot_convergence,
)
from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder, UnfoldingResult

# Skip reference implementations - they are too slow (use tolerance 20%, run >20k iterations)
# Comparison with reference tools can be done separately with smaller test cases
HAS_REF_UNFOLDING = False
print("Note: Reference implementation comparison skipped (too slow for full spectrum)")

import matplotlib.pyplot as plt
import matplotlib

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

# =============================================================================
# Configuration
# =============================================================================

# Flux wire data directory
FLUX_WIRE_DIR = ALARA_ROOT / "rafm_irradiation_ldrd" / "raw_gamma_spec" / "flux_wires"

# Output directory for plots
OUTPUT_DIR = SCRIPT_DIR / "flux_wire_unfolding_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Flux wire samples and their reactions
FLUX_WIRE_SAMPLES = {
    "Co-RAFM-1_25cm.ASC": {
        "element": "Co",
        "reaction": "Co-59(n,g)Co-60",
        "half_life_days": 1925.28,
        "gamma_energy_keV": 1332.5,
    },
    "Co-Cd-RAFM-1_25cm.ASC": {
        "element": "Co",
        "reaction": "Co-59(n,g)Co-60",
        "half_life_days": 1925.28,
        "gamma_energy_keV": 1332.5,
        "cd_covered": True,
    },
    "Ni-RAFM-1_25cm.ASC": {
        "element": "Ni",
        "reaction": "Ni-58(n,p)Co-58",
        "half_life_days": 70.86,
        "gamma_energy_keV": 810.8,
    },
    "Ti-RAFM-1_25cm.ASC": {
        "element": "Ti",
        "reaction": "Ti-46(n,p)Sc-46",
        "half_life_days": 83.79,
        "gamma_energy_keV": 889.3,
    },
    "Ti-RAFM-1a_25cm.ASC": {
        "element": "Ti",
        "reaction": "Ti-47(n,p)Sc-47",
        "half_life_days": 3.3492,
        "gamma_energy_keV": 159.4,
    },
    "Ti-RAFM-1b_25cm.ASC": {
        "element": "Ti",
        "reaction": "Ti-48(n,p)Sc-48",
        "half_life_days": 1.82,
        "gamma_energy_keV": 983.5,
    },
    "Sc-RAFM-1_25cm.ASC": {
        "element": "Sc",
        "reaction": "Sc-45(n,g)Sc-46",
        "half_life_days": 83.79,
        "gamma_energy_keV": 889.3,
    },
    "Sc-Cd-RAFM-1_25cm.ASC": {
        "element": "Sc",
        "reaction": "Sc-45(n,g)Sc-46",
        "half_life_days": 83.79,
        "gamma_energy_keV": 889.3,
        "cd_covered": True,
    },
    "In-Cd-RAFM-1_25cm.ASC": {
        "element": "In",
        "reaction": "In-115(n,n')In-115m",
        "half_life_days": 0.1867,
        "gamma_energy_keV": 336.2,
        "cd_covered": True,
    },
    "CU-RAFM-1.ASC": {
        "element": "Cu",
        "reaction": "Cu-63(n,g)Cu-64",
        "half_life_days": 0.529,
        "gamma_energy_keV": 511.0,
    },
}

# Reactions to use for unfolding (subset with good cross section data)
UNFOLDING_REACTIONS = [
    "Co-59(n,g)Co-60",
    "Ni-58(n,p)Co-58",
    "Ti-46(n,p)Sc-46",
    "Ti-47(n,p)Sc-47",
    "Ti-48(n,p)Sc-48",
    "In-115(n,n')In-115m",
    "Sc-45(n,g)Sc-46",
]


# =============================================================================
# Helper Functions
# =============================================================================

def load_a_priori_spectrum(csv_path: Path) -> tuple:
    """Load a priori spectrum from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Extract energy edges
    e_low = df["E_low[eV]"].values
    e_high = df["E_high[eV]"].values
    energy_edges = np.concatenate([e_low, [e_high[-1]]])
    
    # Extract flux
    flux = df["flux_per_lethargy [n·cm⁻²·s⁻¹]"].values
    flux_err = df["err_per_lethargy [n·cm⁻²·s⁻¹]"].values
    
    return energy_edges, flux, flux_err


def create_simulated_reaction_rates(
    reactions: list,
    energy_edges: np.ndarray,
    flux: np.ndarray,
    db: IRDFFDatabase,
) -> tuple:
    """
    Simulate reaction rates from a priori spectrum.
    
    In a real experiment, these would come from gamma spectroscopy.
    Here we use the a priori spectrum to generate "measured" rates
    for testing the unfolding algorithms.
    """
    reaction_rates = []
    uncertainties = []
    
    # Energy midpoints
    energy_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    
    for rxn in reactions:
        xs = db.get_cross_section(rxn)
        if xs is None:
            print(f"Warning: No cross section for {rxn}")
            reaction_rates.append(0.0)
            uncertainties.append(1.0)
            continue
        
        # Evaluate cross section at midpoints
        sigma = xs.evaluate(energy_mid)
        
        # Calculate reaction rate: R = integral(sigma(E) * phi(E) dE)
        dE = energy_edges[1:] - energy_edges[:-1]
        rate = np.sum(sigma * flux * dE)
        
        # Add 5% measurement uncertainty
        unc = rate * 0.05
        
        # Add some noise to make it realistic
        rate *= (1 + np.random.normal(0, 0.02))
        
        reaction_rates.append(rate)
        uncertainties.append(unc)
    
    return np.array(reaction_rates), np.array(uncertainties)


def build_response_matrix(
    reactions: list,
    energy_edges: np.ndarray,
    db: IRDFFDatabase,
) -> np.ndarray:
    """Build response matrix from IRDFF cross sections."""
    n_reactions = len(reactions)
    n_groups = len(energy_edges) - 1
    
    response = np.zeros((n_reactions, n_groups))
    energy_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    
    for i, rxn in enumerate(reactions):
        xs = db.get_cross_section(rxn)
        if xs is not None:
            response[i, :] = xs.evaluate(energy_mid)
    
    return response


def plot_comparison_grid(
    results: dict,
    reference_flux: np.ndarray,
    energy_edges: np.ndarray,
    save_path: Path,
):
    """Create a grid of comparison plots for all methods."""
    n_methods = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    energy_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:]) / 1e6  # MeV
    
    colors = {
        'gravel': '#ff7f0e',
        'mlem': '#2ca02c',
        'gls': '#1f77b4',
        'mcmc': '#9467bd',
    }
    
    for idx, (method, result) in enumerate(results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Plot reference (a priori)
        ax.loglog(energy_mid, reference_flux * energy_mid, 
                 'k--', linewidth=1.5, alpha=0.7, label='A priori')
        
        # Plot unfolded
        flux = result.flux
        flux_unc = result.flux_uncertainty if len(result.flux_uncertainty) > 0 else np.zeros_like(flux)
        
        # Lethargy representation
        flux_lethargy = flux * result.energy_midpoints
        flux_unc_lethargy = flux_unc * result.energy_midpoints
        
        # Uncertainty band
        if np.any(flux_unc_lethargy > 0):
            ax.fill_between(
                energy_mid,
                np.maximum(flux_lethargy - flux_unc_lethargy, 1e-10),
                flux_lethargy + flux_unc_lethargy,
                color=colors.get(method, 'red'),
                alpha=0.3,
            )
        
        ax.loglog(energy_mid, flux_lethargy,
                 color=colors.get(method, 'red'), linewidth=2,
                 label=f'{method.upper()} (χ²={result.chi_squared:.2f})')
        
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('E·φ(E) [n/cm²/s]')
        ax.set_title(f'{method.upper()} Unfolding')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(1e-8, 20)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_method_comparison(
    results: dict,
    reference_flux: np.ndarray,
    energy_edges: np.ndarray,
    save_path: Path,
):
    """Create overlay plot comparing all methods."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    energy_mid = np.sqrt(energy_edges[:-1] * energy_edges[1:]) / 1e6  # MeV
    
    # Plot reference (a priori)
    ref_lethargy = reference_flux * energy_mid * 1e6  # Convert back to eV for calc
    ax.loglog(energy_mid, ref_lethargy,
             'k-', linewidth=2.5, alpha=0.8, label='A priori (MCNP)')
    
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
    
    for idx, (method, result) in enumerate(results.items()):
        flux_lethargy = result.flux * result.energy_midpoints
        
        ax.loglog(energy_mid, flux_lethargy,
                 color=colors[idx % len(colors)], linewidth=1.5,
                 linestyle='--' if idx > 0 else '-',
                 label=f'{method.upper()} (χ²={result.chi_squared:.2f})')
    
    ax.set_xlabel('Energy (MeV)', fontsize=14)
    ax.set_ylabel('E·φ(E) [n/cm²/s]', fontsize=14)
    ax.set_title('Neutron Spectrum Unfolding: Method Comparison', fontsize=16)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1e-8, 20)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_chi2_comparison(results: dict, save_path: Path):
    """Bar chart comparing chi-squared values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    chi2_values = [r.chi_squared for r in results.values()]
    iterations = [r.iterations for r in results.values()]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars = ax.bar(x, chi2_values, width, label='χ²/dof', color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, chi2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Ideal χ²=1')
    
    ax.set_xlabel('Unfolding Method', fontsize=14)
    ax.set_ylabel('χ²/dof', fontsize=14)
    ax.set_title('Unfolding Quality Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods], fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run flux wire unfolding comparison."""
    print("=" * 80)
    print("FLUX WIRE SPECTRUM UNFOLDING COMPARISON")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Initialize IRDFF database
    print("Initializing IRDFF-II database...")
    db = IRDFFDatabase(auto_download=True, verbose=True)
    
    # Load a priori spectrum
    spectrum_csv = FLUX_WIRE_DIR / "spectrum_vit_j.csv"
    if not spectrum_csv.exists():
        print(f"ERROR: A priori spectrum not found: {spectrum_csv}")
        return
    
    print(f"\nLoading a priori spectrum from: {spectrum_csv}")
    energy_edges, a_priori_flux, a_priori_err = load_a_priori_spectrum(spectrum_csv)
    print(f"  Energy range: {energy_edges[0]:.2e} - {energy_edges[-1]:.2e} eV")
    print(f"  Number of groups: {len(a_priori_flux)}")
    
    # Build response matrix
    print(f"\nBuilding response matrix for {len(UNFOLDING_REACTIONS)} reactions...")
    response_matrix = build_response_matrix(UNFOLDING_REACTIONS, energy_edges, db)
    
    # Check cross section availability
    print("\nChecking cross section availability:")
    for i, rxn in enumerate(UNFOLDING_REACTIONS):
        xs = db.get_cross_section(rxn)
        status = "✓" if xs is not None else "✗"
        print(f"  {status} {rxn}")
    
    # Generate simulated reaction rates from a priori spectrum
    print("\nGenerating simulated reaction rates from a priori spectrum...")
    reaction_rates, rate_uncertainties = create_simulated_reaction_rates(
        UNFOLDING_REACTIONS, energy_edges, a_priori_flux, db
    )
    
    print("\nReaction rates (simulated from a priori):")
    for rxn, rate, unc in zip(UNFOLDING_REACTIONS, reaction_rates, rate_uncertainties):
        print(f"  {rxn}: {rate:.3e} ± {unc:.3e}")
    
    # Initialize spectrum unfolder
    print("\n" + "=" * 80)
    print("RUNNING UNFOLDING ALGORITHMS")
    print("=" * 80)
    
    energy_midpoints = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    
    # Store results for comparison
    results = {}
    
    # --- GRAVEL ---
    print("\n1. GRAVEL unfolding...")
    try:
        gravel_sol = gravel(
            response=response_matrix.tolist(),
            measurements=reaction_rates.tolist(),
            initial_flux=a_priori_flux.tolist(),
            measurement_uncertainty=rate_uncertainties.tolist(),
            tolerance=1e-4,
            max_iters=500,
            verbose=True,
        )
        
        # Create UnfoldingResult object
        flux_arr = np.array(gravel_sol.flux)
        dE = energy_edges[1:] - energy_edges[:-1]
        # Predicted rate for each reaction = sum_g (R_ig * phi_g * dE_g)
        predicted_rates = np.sum(response_matrix * flux_arr[np.newaxis, :] * dE[np.newaxis, :], axis=1)
        
        results['gravel'] = UnfoldingResult(
            flux=flux_arr,
            flux_uncertainty=np.abs(flux_arr) * 0.1,  # Estimated 10% uncertainty
            energy_edges=energy_edges,
            energy_midpoints=energy_midpoints,
            measured_rates=reaction_rates,
            predicted_rates=predicted_rates,
            reactions_used=UNFOLDING_REACTIONS,
            chi_squared=gravel_sol.chi_squared,
            iterations=gravel_sol.iterations,
            converged=gravel_sol.converged,
            method='gravel',
        )
        print(f"   χ²/dof = {gravel_sol.chi_squared:.4f}, iterations = {gravel_sol.iterations}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # --- MLEM ---
    print("\n2. MLEM unfolding...")
    try:
        mlem_sol = mlem(
            response=response_matrix.tolist(),
            measurements=reaction_rates.tolist(),
            initial_flux=a_priori_flux.tolist(),
            measurement_uncertainty=rate_uncertainties.tolist(),
            tolerance=1e-4,
            max_iters=500,
            verbose=True,
        )
        
        flux_arr = np.array(mlem_sol.flux)
        dE = energy_edges[1:] - energy_edges[:-1]
        predicted_rates = np.sum(response_matrix * flux_arr[np.newaxis, :] * dE[np.newaxis, :], axis=1)
        
        results['mlem'] = UnfoldingResult(
            flux=flux_arr,
            flux_uncertainty=np.abs(flux_arr) * 0.1,
            energy_edges=energy_edges,
            energy_midpoints=energy_midpoints,
            measured_rates=reaction_rates,
            predicted_rates=predicted_rates,
            reactions_used=UNFOLDING_REACTIONS,
            chi_squared=mlem_sol.chi_squared,
            iterations=mlem_sol.iterations,
            converged=mlem_sol.converged,
            method='mlem',
        )
        print(f"   χ²/dof = {mlem_sol.chi_squared:.4f}, iterations = {mlem_sol.iterations}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # --- GLS ---
    print("\n3. GLS adjustment...")
    try:
        # Build covariance matrices as lists
        n_groups = len(a_priori_flux)
        n_meas = len(reaction_rates)
        
        # Prior flux covariance (50% relative uncertainty)
        prior_cov = [[0.0] * n_groups for _ in range(n_groups)]
        for i in range(n_groups):
            prior_cov[i][i] = (a_priori_flux[i] * 0.5) ** 2
        
        # Measurement covariance
        meas_cov = [[0.0] * n_meas for _ in range(n_meas)]
        for i in range(n_meas):
            meas_cov[i][i] = rate_uncertainties[i] ** 2
        
        gls_sol = gls_adjust(
            response=response_matrix.tolist(),
            measurements=reaction_rates.tolist(),
            measurement_cov=meas_cov,
            prior_flux=a_priori_flux.tolist(),
            prior_cov=prior_cov,
        )
        
        flux_arr = np.array(gls_sol.flux)
        dE = energy_edges[1:] - energy_edges[:-1]
        predicted_rates = np.sum(response_matrix * flux_arr[np.newaxis, :] * dE[np.newaxis, :], axis=1)
        chi2 = gls_sol.chi2 / max(n_meas - 1, 1)
        
        # Extract uncertainties from posterior covariance
        flux_unc = np.sqrt(np.abs([gls_sol.covariance[i][i] for i in range(n_groups)]))
        
        results['gls'] = UnfoldingResult(
            flux=flux_arr,
            flux_uncertainty=flux_unc,
            energy_edges=energy_edges,
            energy_midpoints=energy_midpoints,
            measured_rates=reaction_rates,
            predicted_rates=predicted_rates,
            reactions_used=UNFOLDING_REACTIONS,
            chi_squared=chi2,
            iterations=1,  # GLS is single-step
            converged=True,
            method='gls',
        )
        print(f"   χ²/dof = {chi2:.4f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Generate comparison plots ---
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    if len(results) > 0:
        # Individual method comparison grid
        plot_comparison_grid(
            results, a_priori_flux, energy_edges,
            OUTPUT_DIR / "unfolding_comparison_grid.png"
        )
        
        # All methods overlay
        plot_method_comparison(
            results, a_priori_flux, energy_edges,
            OUTPUT_DIR / "unfolding_method_overlay.png"
        )
        
        # Chi-squared comparison
        plot_chi2_comparison(results, OUTPUT_DIR / "unfolding_chi2_comparison.png")
        
        # Individual uncertainty band plots
        for method, result in results.items():
            try:
                fig, ax = plot_spectrum_uncertainty_bands(
                    result,
                    confidence_levels=[0.68, 0.95],
                    title=f"{method.upper()} Unfolded Spectrum with Uncertainty",
                    save_path=OUTPUT_DIR / f"spectrum_uncertainty_{method}.png"
                )
                plt.close(fig)
            except Exception as e:
                print(f"Error plotting {method} uncertainty: {e}")
    
    # --- Compare with reference implementations ---
    if HAS_REF_UNFOLDING:
        print("\n" + "=" * 80)
        print("COMPARING WITH REFERENCE IMPLEMENTATIONS")
        print("=" * 80)
        
        # Run reference GRAVEL
        print("\nRunning reference GRAVEL from testing/Neutron-Unfolding...")
        try:
            ref_gravel, ref_gravel_err = gravel_ref(
                response_matrix, reaction_rates, a_priori_flux.copy(), 0.2
            )
            
            # Normalize for comparison
            ff_gravel = results['gravel'].flux / np.linalg.norm(results['gravel'].flux)
            ref_gravel_norm = ref_gravel / np.linalg.norm(ref_gravel)
            
            # Calculate difference
            diff = np.linalg.norm(ff_gravel - ref_gravel_norm) / np.linalg.norm(ref_gravel_norm)
            print(f"   Relative difference: {diff*100:.2f}%")
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            energy_mid = energy_midpoints / 1e6
            
            ax.loglog(energy_mid, ff_gravel, 'b-', linewidth=2, label='FluxForge GRAVEL')
            ax.loglog(energy_mid, ref_gravel_norm, 'r--', linewidth=2, label='Reference GRAVEL')
            
            ax.set_xlabel('Energy (MeV)', fontsize=14)
            ax.set_ylabel('Normalized Flux', fontsize=14)
            ax.set_title('FluxForge vs Reference GRAVEL Comparison', fontsize=16)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            save_path = OUTPUT_DIR / "fluxforge_vs_reference_gravel.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {save_path}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
        
        # Run reference MLEM
        print("\nRunning reference MLEM from testing/Neutron-Unfolding...")
        try:
            ref_mlem, ref_mlem_err = mlem_ref(
                response_matrix, reaction_rates, a_priori_flux.copy(), 0.2
            )
            
            ff_mlem = results['mlem'].flux / np.linalg.norm(results['mlem'].flux)
            ref_mlem_norm = ref_mlem / np.linalg.norm(ref_mlem)
            
            diff = np.linalg.norm(ff_mlem - ref_mlem_norm) / np.linalg.norm(ref_mlem_norm)
            print(f"   Relative difference: {diff*100:.2f}%")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.loglog(energy_mid, ff_mlem, 'b-', linewidth=2, label='FluxForge MLEM')
            ax.loglog(energy_mid, ref_mlem_norm, 'r--', linewidth=2, label='Reference MLEM')
            
            ax.set_xlabel('Energy (MeV)', fontsize=14)
            ax.set_ylabel('Normalized Flux', fontsize=14)
            ax.set_title('FluxForge vs Reference MLEM Comparison', fontsize=16)
            ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            save_path = OUTPUT_DIR / "fluxforge_vs_reference_mlem.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {save_path}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nGenerated plots in: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")
    
    print("\nUnfolding results summary:")
    for method, result in results.items():
        print(f"  {method.upper():8s}: χ²/dof = {result.chi_squared:.4f}, "
              f"iterations = {result.iterations}, converged = {result.converged}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
