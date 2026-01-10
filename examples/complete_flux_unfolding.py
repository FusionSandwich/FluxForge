#!/usr/bin/env python
"""
Complete Flux Wire Spectrum Unfolding

Uses all available flux wire measurements (10-12 samples) to unfold the
neutron spectrum. Compares:
1. Model spectrum (from MCNP if available)
2. Discrete 10-bin unfolded spectrum
3. Continuous GLS-adjusted spectrum

Both raw ASC files and processed TXT files are analyzed and compared.
"""

from pathlib import Path
import numpy as np
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add FluxForge to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fluxforge.io.flux_wire import (
    read_processed_txt,
    read_raw_asc,
    FluxWireData,
)
from fluxforge.analysis.flux_wire_analysis import (
    analyze_flux_wire,
    get_sample_element,
    get_expected_isotopes,
    FLUX_WIRE_NUCLIDES,
)
from fluxforge.analysis.flux_unfold import (
    unfold_discrete_bins,
    unfold_gls,
    extract_reactions_from_processed,
    FluxWireReaction,
    THERMAL_CROSS_SECTIONS,
    REACTION_ENERGIES,
)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


# =============================================================================
# Data Loading
# =============================================================================

def load_all_flux_wires(proc_dir: Path, raw_dir: Optional[Path] = None) -> Dict[str, dict]:
    """
    Load all flux wire data from processed and optionally raw files.
    
    Returns dict mapping sample_id to {'processed': FluxWireData, 'raw': FluxWireData}
    """
    samples = {}
    
    # Load processed files
    for filepath in sorted(proc_dir.glob("*.txt")):
        try:
            data = read_processed_txt(filepath)
            sample_id = data.sample_id
            if sample_id not in samples:
                samples[sample_id] = {'processed': None, 'raw': None, 'filepath': filepath}
            samples[sample_id]['processed'] = data
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    # Load raw files if available
    if raw_dir and raw_dir.exists():
        for filepath in sorted(raw_dir.glob("*.ASC")):
            try:
                data = read_raw_asc(filepath)
                sample_id = data.sample_id
                # Find matching processed sample
                matched = False
                for sid in samples:
                    if sid.replace(" ", "").replace("@", "_") in sample_id.replace(" ", "").replace("@", "_") or \
                       sample_id.replace(" ", "").replace("@", "_") in sid.replace(" ", "").replace("@", "_"):
                        samples[sid]['raw'] = data
                        matched = True
                        break
                if not matched and sample_id not in samples:
                    samples[sample_id] = {'processed': None, 'raw': data, 'filepath': filepath}
            except Exception as e:
                print(f"Warning: Could not read {filepath.name}: {e}")
    
    return samples


def extract_all_reactions(
    samples: Dict[str, dict],
    irradiation_time_s: float = 8 * 3600,  # 8 hours default
    decay_time_s: float = 4 * 3600,  # 4 hours default
    use_raw: bool = False,
) -> Tuple[List[FluxWireReaction], List[FluxWireReaction]]:
    """
    Extract reactions from all flux wire samples.
    
    Returns (processed_reactions, raw_reactions)
    """
    proc_reactions = []
    raw_reactions = []
    
    for sample_id, data_dict in samples.items():
        # Process processed data
        if data_dict['processed'] is not None:
            try:
                rxns = extract_reactions_from_processed(
                    data_dict['processed'],
                    irradiation_time_s=irradiation_time_s,
                    decay_time_s=decay_time_s,
                    calculate_flux=True,
                )
                proc_reactions.extend(rxns)
            except Exception as e:
                print(f"Warning: Could not extract reactions from {sample_id}: {e}")
        
        # Process raw data (if available and analyzed)
        if use_raw and data_dict['raw'] is not None:
            try:
                # Would need to analyze raw spectrum first
                # For now, skip raw processing
                pass
            except Exception as e:
                print(f"Warning: Could not analyze raw {sample_id}: {e}")
    
    return proc_reactions, raw_reactions


# =============================================================================
# Model Spectrum Loading
# =============================================================================

def load_mcnp_spectrum(spectrum_file: Path) -> Optional[Dict]:
    """
    Load MCNP neutron spectrum from file.
    
    Supports CSV format with columns: energy_eV, flux
    """
    if not spectrum_file.exists():
        return None
    
    energy = []
    flux = []
    
    try:
        with open(spectrum_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        e = float(parts[0])
                        f_val = float(parts[1])
                        energy.append(e)
                        flux.append(f_val)
                    except ValueError:
                        continue
        
        if energy and flux:
            return {
                'energy_eV': np.array(energy),
                'flux': np.array(flux),
                'source': str(spectrum_file),
            }
    except Exception as e:
        print(f"Warning: Could not load MCNP spectrum: {e}")
    
    return None


def create_1_over_e_spectrum(
    e_min_eV: float = 0.0253,
    e_max_eV: float = 2e7,
    n_points: int = 100,
    thermal_peak: float = 1e13,
    epithermal_fraction: float = 0.1,
) -> Dict:
    """
    Create a synthetic 1/E spectrum with thermal peak for reference.
    """
    # Lethargy-spaced energies
    energies = np.logspace(np.log10(e_min_eV), np.log10(e_max_eV), n_points)
    
    # 1/E epithermal + Maxwell thermal
    thermal = thermal_peak * np.exp(-energies / 0.0253) * (energies / 0.0253)**0.5
    epithermal = epithermal_fraction * thermal_peak * (0.0253 / energies)
    
    flux = thermal + epithermal
    
    return {
        'energy_eV': energies,
        'flux': flux,
        'source': 'synthetic_1_over_E',
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_spectrum_comparison(
    discrete_result,
    gls_result,
    model_spectrum: Optional[Dict],
    output_file: Path,
    reactions: List[FluxWireReaction],
    title: str = "Neutron Flux Spectrum Unfolding Comparison",
):
    """
    Create a comprehensive plot comparing model, discrete, and GLS spectra.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors
    model_color = 'red'
    discrete_color = 'blue'
    gls_color = 'green'
    
    # === Top Left: Full Spectrum Comparison ===
    ax1 = axes[0, 0]
    
    # Discrete spectrum (step plot)
    ax1.step(discrete_result.energy_bounds_eV[:-1], discrete_result.flux, 
             where='post', label='Discrete 10-bin', color=discrete_color, 
             linewidth=2, alpha=0.8)
    ax1.fill_between(discrete_result.energy_bounds_eV[:-1],
                     np.maximum(discrete_result.flux - discrete_result.flux_unc, 1e5),
                     discrete_result.flux + discrete_result.flux_unc,
                     alpha=0.2, step='post', color=discrete_color)
    
    # GLS spectrum
    ax1.step(gls_result.energy_bounds_eV[:-1], gls_result.flux,
             where='post', label=f'GLS Continuous (χ²={gls_result.chi2:.1f})',
             color=gls_color, linewidth=2, alpha=0.8)
    ax1.fill_between(gls_result.energy_bounds_eV[:-1],
                     np.maximum(gls_result.flux - gls_result.flux_unc, 1e5),
                     gls_result.flux + gls_result.flux_unc,
                     alpha=0.2, step='post', color=gls_color)
    
    # Model spectrum
    if model_spectrum is not None:
        ax1.plot(model_spectrum['energy_eV'], model_spectrum['flux'],
                 label='Model (MCNP)', color=model_color, linewidth=2, 
                 linestyle='--', alpha=0.8)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Flux (n/cm²/s)')
    ax1.set_title('Spectrum Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(1e-2, 2e7)
    ax1.set_ylim(1e8, 1e15)
    
    # === Top Right: Reaction Sensitivity ===
    ax2 = axes[0, 1]
    
    # Plot reaction energies and measured fluxes
    reaction_energies = []
    reaction_fluxes = []
    reaction_labels = []
    reaction_types = []
    
    for rxn in reactions:
        e_char = REACTION_ENERGIES.get(rxn.reaction_id, None)
        flux = getattr(rxn, 'flux', 0)
        if e_char and flux > 0:
            reaction_energies.append(e_char)
            reaction_fluxes.append(flux)
            reaction_labels.append(rxn.isotope)
            reaction_types.append(getattr(rxn, 'flux_type', 'unknown'))
    
    # Color by reaction type
    thermal_mask = np.array([t == 'thermal' for t in reaction_types])
    fast_mask = np.array([t == 'fast' for t in reaction_types])
    
    if any(thermal_mask):
        ax2.scatter(np.array(reaction_energies)[thermal_mask], 
                   np.array(reaction_fluxes)[thermal_mask],
                   c='blue', s=100, marker='o', label='Thermal (n,γ)', alpha=0.7)
    if any(fast_mask):
        ax2.scatter(np.array(reaction_energies)[fast_mask],
                   np.array(reaction_fluxes)[fast_mask],
                   c='orange', s=100, marker='^', label='Fast (n,p)', alpha=0.7)
    
    # Add labels
    for i, (e, f, lbl) in enumerate(zip(reaction_energies, reaction_fluxes, reaction_labels)):
        ax2.annotate(lbl, (e, f), textcoords="offset points", xytext=(5, 5), 
                    fontsize=8, alpha=0.8)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Characteristic Energy (eV)')
    ax2.set_ylabel('Measured Flux (n/cm²/s)')
    ax2.set_title('Reaction Sensitivity Energies')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1e-2, 2e7)
    
    # === Bottom Left: Lethargy Plot ===
    ax3 = axes[1, 0]
    
    # Convert to lethargy (E * dφ/dE = constant for 1/E spectrum)
    disc_centers = np.sqrt(discrete_result.energy_bounds_eV[:-1] * discrete_result.energy_bounds_eV[1:])
    disc_widths = discrete_result.energy_bounds_eV[1:] - discrete_result.energy_bounds_eV[:-1]
    disc_lethargy = discrete_result.flux * disc_centers / disc_widths
    
    gls_centers = np.sqrt(gls_result.energy_bounds_eV[:-1] * gls_result.energy_bounds_eV[1:])
    gls_widths = gls_result.energy_bounds_eV[1:] - gls_result.energy_bounds_eV[:-1]
    gls_lethargy = gls_result.flux * gls_centers / gls_widths
    
    ax3.semilogx(disc_centers, disc_lethargy, 'o-', label='Discrete', 
                 color=discrete_color, markersize=8, linewidth=2)
    ax3.semilogx(gls_centers, gls_lethargy, 's-', label='GLS', 
                 color=gls_color, markersize=6, linewidth=2)
    
    if model_spectrum is not None:
        model_widths = np.diff(model_spectrum['energy_eV'])
        model_centers = (model_spectrum['energy_eV'][:-1] + model_spectrum['energy_eV'][1:]) / 2
        model_lethargy = model_spectrum['flux'][:-1] * model_centers / model_widths
        ax3.semilogx(model_centers, model_lethargy, '--', label='Model', 
                     color=model_color, linewidth=2)
    
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('E·dφ/dE (lethargy flux)')
    ax3.set_title('Lethargy Representation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1e-2, 2e7)
    
    # === Bottom Right: Flux Summary Table ===
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = [
        ['Flux Type', 'Discrete', 'GLS'],
        ['Thermal (<0.5 eV)', f'{discrete_result.flux[0]:.2e}', 
         f'{np.sum(gls_result.flux[:4]):.2e}'],
        ['Epithermal (0.5-100 keV)', f'{np.sum(discrete_result.flux[1:7]):.2e}',
         f'{np.sum(gls_result.flux[4:14]):.2e}'],
        ['Fast (>100 keV)', f'{np.sum(discrete_result.flux[7:]):.2e}',
         f'{np.sum(gls_result.flux[14:]):.2e}'],
        ['Total', f'{np.sum(discrete_result.flux):.2e}',
         f'{np.sum(gls_result.flux):.2e}'],
    ]
    
    # Also add reaction summary
    table_data.extend([
        ['', '', ''],
        ['Reactions Used', str(len(reactions)), ''],
        ['Thermal reactions', str(sum(1 for r in reactions if getattr(r, 'flux_type', '') == 'thermal')), ''],
        ['Fast reactions', str(sum(1 for r in reactions if getattr(r, 'flux_type', '') == 'fast')), ''],
    ])
    
    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', weight='bold')
    
    ax4.set_title('Flux Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def save_results_to_csv(
    discrete_result,
    gls_result,
    reactions: List[FluxWireReaction],
    output_dir: Path,
):
    """Save all results to CSV files."""
    
    # Save discrete spectrum
    with open(output_dir / "discrete_spectrum.csv", 'w') as f:
        f.write("# Discrete 10-bin Unfolded Spectrum\n")
        f.write("E_low_eV,E_high_eV,Flux_n_cm2_s,Flux_unc\n")
        for i in range(len(discrete_result.flux)):
            f.write(f"{discrete_result.energy_bounds_eV[i]:.6e},"
                    f"{discrete_result.energy_bounds_eV[i+1]:.6e},"
                    f"{discrete_result.flux[i]:.6e},"
                    f"{discrete_result.flux_unc[i]:.6e}\n")
    
    # Save GLS spectrum
    with open(output_dir / "gls_spectrum.csv", 'w') as f:
        f.write("# GLS Continuous Unfolded Spectrum\n")
        f.write("E_low_eV,E_high_eV,Flux_n_cm2_s,Flux_unc\n")
        for i in range(len(gls_result.flux)):
            f.write(f"{gls_result.energy_bounds_eV[i]:.6e},"
                    f"{gls_result.energy_bounds_eV[i+1]:.6e},"
                    f"{gls_result.flux[i]:.6e},"
                    f"{gls_result.flux_unc[i]:.6e}\n")
    
    # Save reaction data
    with open(output_dir / "reaction_data.csv", 'w') as f:
        f.write("# Flux Wire Reaction Data\n")
        f.write("Sample,Isotope,Reaction,Activity_Bq,Flux_n_cm2_s,Flux_type,E_char_eV\n")
        for rxn in reactions:
            e_char = REACTION_ENERGIES.get(rxn.reaction_id, 0)
            flux = getattr(rxn, 'flux', 0)
            flux_type = getattr(rxn, 'flux_type', 'unknown')
            f.write(f"{rxn.sample_id},{rxn.isotope},{rxn.reaction_id},"
                    f"{rxn.activity_bq:.6e},{flux:.6e},{flux_type},{e_char:.2e}\n")
    
    print(f"Results saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run complete flux wire spectrum unfolding."""
    print("\n" + "#" * 80)
    print("# COMPLETE FLUX WIRE NEUTRON SPECTRUM UNFOLDING")
    print("#" * 80)
    
    # === Setup Paths ===
    base_dir = Path("/filespace/s/smandych/CAE/projects/ALARA/rafm_irradiation_ldrd")
    proc_dir = base_dir / "irradiation_QG_processed/flux_wires"
    raw_dir = base_dir / "raw_gamma_spec/flux_wires"
    output_dir = Path("/filespace/s/smandych/CAE/projects/ALARA/FluxForge/flux_unfolding_results")
    output_dir.mkdir(exist_ok=True)
    
    # Optional model spectrum
    model_file = Path("/filespace/s/smandych/CAE/projects/ALARA/spectra_files/whale_spectrum.txt")
    
    # === Load All Flux Wire Data ===
    print("\n1. LOADING FLUX WIRE DATA")
    print("-" * 40)
    
    if not proc_dir.exists():
        print(f"ERROR: Processed directory not found: {proc_dir}")
        return
    
    samples = load_all_flux_wires(proc_dir, raw_dir)
    print(f"Loaded {len(samples)} flux wire samples:")
    
    for sample_id, data_dict in samples.items():
        proc_status = "✓" if data_dict['processed'] else "✗"
        raw_status = "✓" if data_dict['raw'] else "✗"
        element = get_sample_element(sample_id)
        nuclides = data_dict['processed'].nuclides if data_dict['processed'] else []
        isotopes = [n.isotope for n in nuclides]
        print(f"  {sample_id:<30} Proc:{proc_status} Raw:{raw_status} Element:{element:<3} Isotopes:{isotopes}")
    
    # === Extract Reactions ===
    print("\n2. EXTRACTING REACTIONS")
    print("-" * 40)
    
    # Use realistic irradiation parameters
    # MURR typical: 8 hour irradiation, ~4 hours decay before counting
    proc_reactions, raw_reactions = extract_all_reactions(
        samples,
        irradiation_time_s=8 * 3600,  # 8 hours
        decay_time_s=4 * 3600,  # 4 hours
        use_raw=False,  # Raw analysis not fully implemented yet
    )
    
    print(f"Extracted {len(proc_reactions)} reactions from processed data")
    
    print("\nReaction Summary:")
    print(f"{'Sample':<25} {'Isotope':<10} {'Flux (n/cm²/s)':<15} {'Type'}")
    print("-" * 65)
    for rxn in proc_reactions:
        flux = getattr(rxn, 'flux', 0)
        flux_type = getattr(rxn, 'flux_type', 'unknown')
        print(f"{rxn.sample_id[:24]:<25} {rxn.isotope:<10} {flux:<15.3e} {flux_type}")
    
    # === Load Model Spectrum ===
    print("\n3. LOADING MODEL SPECTRUM")
    print("-" * 40)
    
    model_spectrum = load_mcnp_spectrum(model_file)
    if model_spectrum is None:
        print("No MCNP model spectrum found, creating synthetic 1/E reference")
        # Estimate thermal flux from reactions
        thermal_fluxes = [getattr(r, 'flux', 0) for r in proc_reactions 
                         if getattr(r, 'flux_type', '') == 'thermal']
        thermal_peak = np.mean(thermal_fluxes) if thermal_fluxes else 1e12
        model_spectrum = create_1_over_e_spectrum(thermal_peak=thermal_peak)
    else:
        print(f"Loaded model spectrum from: {model_spectrum['source']}")
    
    # === Unfold Spectrum ===
    print("\n4. UNFOLDING NEUTRON SPECTRUM")
    print("-" * 40)
    
    # Discrete 10-bin unfolding
    print("\n4a. Discrete 10-bin unfolding...")
    discrete_result = unfold_discrete_bins(
        reactions=proc_reactions,
        n_bins=10,
        e_min_eV=0.0253,
        e_max_eV=2e7,
    )
    
    print("Discrete spectrum:")
    print(f"  {'Bin':<4} {'E_low (eV)':<12} {'E_high (eV)':<12} {'Flux (n/cm²/s)'}")
    for i in range(len(discrete_result.flux)):
        print(f"  {i+1:<4} {discrete_result.energy_bounds_eV[i]:<12.2e} "
              f"{discrete_result.energy_bounds_eV[i+1]:<12.2e} {discrete_result.flux[i]:.3e}")
    
    # GLS continuous unfolding
    print("\n4b. GLS continuous unfolding (20 groups)...")
    gls_result = unfold_gls(
        reactions=proc_reactions,
        n_groups=20,
        e_min_eV=0.0253,
        e_max_eV=2e7,
        prior_uncertainty=2.0,
    )
    print(f"GLS Chi-squared: {gls_result.chi2:.2f}")
    print(f"GLS Total flux: {np.sum(gls_result.flux):.3e} n/cm²/s")
    
    # === Save Results ===
    print("\n5. SAVING RESULTS")
    print("-" * 40)
    
    save_results_to_csv(discrete_result, gls_result, proc_reactions, output_dir)
    
    # === Generate Plot ===
    print("\n6. GENERATING COMPARISON PLOT")
    print("-" * 40)
    
    plot_spectrum_comparison(
        discrete_result=discrete_result,
        gls_result=gls_result,
        model_spectrum=model_spectrum,
        output_file=output_dir / "complete_spectrum_comparison.png",
        reactions=proc_reactions,
        title="MURR Whale Tube Neutron Spectrum - Flux Wire Unfolding",
    )
    
    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFlux wire samples analyzed: {len(samples)}")
    print(f"Reactions extracted: {len(proc_reactions)}")
    print(f"  Thermal reactions: {sum(1 for r in proc_reactions if getattr(r, 'flux_type', '') == 'thermal')}")
    print(f"  Fast reactions: {sum(1 for r in proc_reactions if getattr(r, 'flux_type', '') == 'fast')}")
    
    # Flux summary
    thermal_fluxes = [getattr(r, 'flux', 0) for r in proc_reactions 
                     if getattr(r, 'flux_type', '') == 'thermal' and getattr(r, 'flux', 0) > 0]
    fast_fluxes = [getattr(r, 'flux', 0) for r in proc_reactions 
                  if getattr(r, 'flux_type', '') == 'fast' and getattr(r, 'flux', 0) > 0]
    
    print(f"\nFlux estimates from individual reactions:")
    if thermal_fluxes:
        print(f"  Thermal (n,γ): {np.mean(thermal_fluxes):.2e} ± {np.std(thermal_fluxes):.2e} n/cm²/s")
    if fast_fluxes:
        print(f"  Fast (n,p):    {np.mean(fast_fluxes):.2e} ± {np.std(fast_fluxes):.2e} n/cm²/s")
    
    print(f"\nUnfolded spectrum total flux:")
    print(f"  Discrete 10-bin: {np.sum(discrete_result.flux):.2e} n/cm²/s")
    print(f"  GLS 20-group:    {np.sum(gls_result.flux):.2e} n/cm²/s")
    
    print(f"\nOutput files:")
    print(f"  {output_dir / 'complete_spectrum_comparison.png'}")
    print(f"  {output_dir / 'discrete_spectrum.csv'}")
    print(f"  {output_dir / 'gls_spectrum.csv'}")
    print(f"  {output_dir / 'reaction_data.csv'}")


if __name__ == "__main__":
    main()
