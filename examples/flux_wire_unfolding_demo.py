#!/usr/bin/env python
"""
Flux Wire Spectrum Unfolding Example

Demonstrates the complete workflow for:
1. Loading processed flux wire gamma spectroscopy data
2. Analyzing raw gamma spectra to extract activities
3. Converting activities to reaction rates
4. Unfolding neutron spectrum using discrete binning and GLS methods
5. Plotting modeled vs unfolded spectra

This example uses flux wire data from the RAFM irradiation experiment.
"""

from pathlib import Path
import numpy as np
import sys

# Add FluxForge to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fluxforge.io.flux_wire import (
    read_processed_txt,
    read_raw_asc,
    load_flux_wire_directory,
)
from fluxforge.analysis.flux_wire_analysis import (
    analyze_flux_wire,
    compare_raw_vs_processed,
    get_sample_element,
    get_expected_isotopes,
    FLUX_WIRE_NUCLIDES,
    ELEMENT_TO_ISOTOPES,
)
from fluxforge.analysis.flux_unfold import (
    unfold_flux_wires,
    unfold_discrete_bins,
    unfold_gls,
    extract_reactions_from_processed,
    FluxWireReaction,
)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def demonstrate_sample_isotope_mapping():
    """Show how flux wire samples map to expected isotopes."""
    print_header("1. SAMPLE TO ISOTOPE MAPPING")
    
    print("\nFlux wire materials and their activation products:\n")
    print(f"{'Element':<10} {'Expected Isotopes':<30} {'Primary Reaction'}")
    print("-" * 70)
    
    for element, isotopes in ELEMENT_TO_ISOTOPES.items():
        reactions = []
        for iso in isotopes:
            rxn = FLUX_WIRE_NUCLIDES.get(iso, {}).get('reaction', 'Unknown')
            reactions.append(rxn)
        print(f"{element:<10} {', '.join(isotopes):<30} {reactions[0] if reactions else ''}")
    
    print("\nExample sample name parsing:")
    test_samples = [
        "Co-Cd-RAFM-1_25cm",
        "Ti-RAFM-1_25cm",
        "Ni-RAFM-1_25cm",
    ]
    for sample in test_samples:
        element = get_sample_element(sample)
        isotopes = get_expected_isotopes(sample)
        print(f"  {sample} -> Element: {element}, Isotopes: {isotopes}")


def demonstrate_processed_file_analysis(proc_dir: Path):
    """Demonstrate analysis of processed flux wire files."""
    print_header("2. PROCESSED FILE ANALYSIS")
    
    files = sorted(proc_dir.glob("*.txt"))
    print(f"\nFound {len(files)} processed flux wire files\n")
    
    print(f"{'File':<30} {'Isotope':<10} {'Activity (uCi)':<15} {'Activity (Bq)':<15}")
    print("-" * 75)
    
    for filepath in files[:6]:  # Show first 6
        try:
            data = read_processed_txt(filepath)
            for nuclide in data.nuclides:
                print(f"{data.sample_id:<30} {nuclide.isotope:<10} "
                      f"{nuclide.activity:<15.4e} {nuclide.activity_bq:<15.3e}")
        except Exception as e:
            print(f"Error reading {filepath.name}: {e}")
    
    if len(files) > 6:
        print(f"... and {len(files) - 6} more files")


def demonstrate_raw_spectrum_analysis(raw_dir: Path, proc_dir: Path):
    """Compare raw spectrum analysis with commercial processed results."""
    print_header("3. RAW SPECTRUM ANALYSIS vs COMMERCIAL")
    
    raw_file = raw_dir / "Co-Cd-RAFM-1_25cm.ASC"
    proc_file = proc_dir / "Co-Cd-RAFM-1_25cm.txt"
    
    if not raw_file.exists() or not proc_file.exists():
        print("Test files not found, skipping this section")
        return
    
    # Load data
    raw_data = read_raw_asc(raw_file)
    proc_data = read_processed_txt(proc_file)
    
    print(f"\nSample: {raw_data.sample_id}")
    print(f"Live time: {raw_data.live_time:.1f} s")
    print(f"Expected isotope: {get_expected_isotopes(raw_data.sample_id)}")
    
    # Copy efficiency from processed to raw
    if proc_data.efficiency:
        raw_data.efficiency = proc_data.efficiency
    if proc_data.spectrum:
        raw_data.spectrum.calibration = proc_data.spectrum.calibration.copy()
        raw_data.spectrum._update_energies()
    
    # Analyze
    result = analyze_flux_wire(raw_data, reference_data=proc_data)
    
    print("\nComparison:")
    print(f"{'Isotope':<10} {'FluxForge (uCi)':<18} {'Commercial (uCi)':<18} {'Ratio'}")
    print("-" * 60)
    
    for isotope, calc_act in result.nuclide_activities.items():
        if isotope in result.reference_activities:
            ref_bq = result.reference_activities[isotope]
            calc_uci = calc_act['activity_uci']
            ref_uci = ref_bq / 3.7e4
            ratio = result.activity_ratios.get(isotope, 0)
            print(f"{isotope:<10} {calc_uci:<18.4e} {ref_uci:<18.4e} {ratio:.3f}")


def demonstrate_reaction_rate_extraction(proc_dir: Path):
    """Demonstrate extraction of reaction rates from activities."""
    print_header("4. REACTION RATE EXTRACTION AND FLUX CALCULATION")
    
    files = sorted(proc_dir.glob("*.txt"))
    
    print("\nConverting activities to reaction rates and calculating flux:")
    print("(Using actual sample masses from FLUX_WIRE_SAMPLES)\n")
    
    print(f"{'Sample':<25} {'Reaction':<20} {'Activity (Bq)':<15} {'Flux (n/cm²/s)':<15} {'Type'}")
    print("-" * 95)
    
    all_reactions = []
    
    for filepath in files:
        try:
            data = read_processed_txt(filepath)
            if not data.nuclides:
                print(f"{data.sample_id[:24]:<25} No nuclides detected")
                continue
                
            reactions = extract_reactions_from_processed(
                data,
                sample_mass_mg=None,  # Use default from FLUX_WIRE_SAMPLES
                irradiation_time_s=3600,
                calculate_flux=True,
            )
            all_reactions.extend(reactions)
            
            for rxn in reactions:
                flux = getattr(rxn, 'flux', 0.0)
                flux_type = getattr(rxn, 'flux_type', 'unknown')
                print(f"{rxn.sample_id[:24]:<25} {rxn.reaction_id[:19]:<20} "
                      f"{rxn.activity_bq:<15.3e} {flux:<15.3e} {flux_type}")
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
    
    # Summary of flux values
    if all_reactions:
        print("\n" + "-" * 95)
        print("FLUX SUMMARY BY TYPE:")
        thermal_fluxes = [getattr(r, 'flux', 0) for r in all_reactions if getattr(r, 'flux_type', '') == 'thermal']
        fast_fluxes = [getattr(r, 'flux', 0) for r in all_reactions if getattr(r, 'flux_type', '') == 'fast']
        
        if thermal_fluxes:
            print(f"  Thermal flux (average): {np.mean(thermal_fluxes):.3e} n/cm²/s")
        if fast_fluxes:
            print(f"  Fast flux (average):    {np.mean(fast_fluxes):.3e} n/cm²/s")
    
    return all_reactions


def demonstrate_discrete_unfolding(reactions: list):
    """Demonstrate discrete N-bin spectrum unfolding."""
    print_header("5. DISCRETE 10-BIN SPECTRUM UNFOLDING")
    
    result = unfold_discrete_bins(
        reactions=reactions,
        n_bins=10,
        e_min_eV=0.0253,
        e_max_eV=2e7,
    )
    
    print("\nUnfolded spectrum (10 equal-lethargy bins):\n")
    print(f"{'Bin':<4} {'E_low (eV)':<12} {'E_high (eV)':<12} {'Flux':<15} {'Reactions'}")
    print("-" * 60)
    
    for i in range(len(result.flux)):
        e_lo = result.energy_bounds_eV[i]
        e_hi = result.energy_bounds_eV[i+1]
        flux = result.flux[i]
        n_rxn = "+" if flux > 0 else ""
        print(f"{i+1:<4} {e_lo:<12.2e} {e_hi:<12.2e} {flux:<15.3e} {n_rxn}")
    
    # Summary
    print("\nSpectrum characteristics:")
    thermal_flux = result.flux[0] + result.flux[1]  # First two bins
    fast_flux = result.flux[-3:].sum()  # Last three bins
    print(f"  Thermal+epithermal flux: {thermal_flux:.3e}")
    print(f"  Fast flux (>100 keV): {fast_flux:.3e}")
    
    return result


def demonstrate_gls_unfolding(reactions: list):
    """Demonstrate GLS spectrum adjustment unfolding."""
    print_header("6. GLS CONTINUOUS SPECTRUM ADJUSTMENT")
    
    result = unfold_gls(
        reactions=reactions,
        n_groups=20,
        e_min_eV=0.0253,
        e_max_eV=2e7,
        prior_uncertainty=2.0,
    )
    
    print("\nGLS adjusted spectrum (20 groups):\n")
    print(f"Chi-squared: {result.chi2:.4f}")
    print(f"Total flux: {result.flux.sum():.3e}")
    
    # Show key features
    bin_centers = np.sqrt(result.energy_bounds_eV[:-1] * result.energy_bounds_eV[1:])
    
    print("\nSpectrum shape (selected bins):")
    print(f"{'Energy (eV)':<15} {'Flux':<15} {'Rel. Unc.'}")
    print("-" * 45)
    
    for i in [0, 4, 9, 14, 19]:  # Show 5 representative bins
        if i < len(result.flux):
            e = bin_centers[i]
            flux = result.flux[i]
            unc = result.flux_unc[i] / flux if flux > 0 else 0
            print(f"{e:<15.2e} {flux:<15.3e} {unc*100:.1f}%")
    
    return result


def save_spectrum_to_csv(discrete_result, gls_result, output_file: Path):
    """Save unfolded spectra to CSV file."""
    print_header("7. SAVING RESULTS")
    
    with open(output_file, 'w') as f:
        # Discrete spectrum
        f.write("# Discrete 10-bin spectrum\n")
        f.write("E_low_eV,E_high_eV,Flux,Flux_unc\n")
        for i in range(len(discrete_result.flux)):
            f.write(f"{discrete_result.energy_bounds_eV[i]:.6e},"
                    f"{discrete_result.energy_bounds_eV[i+1]:.6e},"
                    f"{discrete_result.flux[i]:.6e},"
                    f"{discrete_result.flux_unc[i]:.6e}\n")
        
        f.write("\n# GLS adjusted spectrum\n")
        f.write("E_low_eV,E_high_eV,Flux,Flux_unc\n")
        for i in range(len(gls_result.flux)):
            f.write(f"{gls_result.energy_bounds_eV[i]:.6e},"
                    f"{gls_result.energy_bounds_eV[i+1]:.6e},"
                    f"{gls_result.flux[i]:.6e},"
                    f"{gls_result.flux_unc[i]:.6e}\n")
    
    print(f"\nResults saved to: {output_file}")


def plot_unfolded_spectrum(discrete_result, gls_result, output_file: Path, model_spectrum=None):
    """Plot the unfolded spectra comparison."""
    print_header("8. PLOTTING SPECTRUM COMPARISON")
    
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate bin centers (geometric mean)
    disc_centers = np.sqrt(discrete_result.energy_bounds_eV[:-1] * discrete_result.energy_bounds_eV[1:])
    gls_centers = np.sqrt(gls_result.energy_bounds_eV[:-1] * gls_result.energy_bounds_eV[1:])
    
    # Left plot: Discrete 10-bin spectrum
    ax1.step(discrete_result.energy_bounds_eV[:-1], discrete_result.flux, where='post', 
             label='Discrete 10-bin', color='blue', linewidth=2)
    ax1.fill_between(discrete_result.energy_bounds_eV[:-1], 
                     discrete_result.flux - discrete_result.flux_unc,
                     discrete_result.flux + discrete_result.flux_unc,
                     alpha=0.3, step='post', color='blue')
    
    # Add model spectrum if provided
    if model_spectrum is not None:
        ax1.step(model_spectrum['energy_eV'], model_spectrum['flux'], where='post',
                 label='Model (MCNP)', color='red', linewidth=2, linestyle='--')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Flux (n/cm²/s/eV)')
    ax1.set_title('Discrete 10-bin Unfolding')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e-2, 2e7)
    
    # Right plot: GLS adjusted spectrum
    ax2.step(gls_result.energy_bounds_eV[:-1], gls_result.flux, where='post',
             label=f'GLS Adjusted (χ²={gls_result.chi2:.2f})', color='green', linewidth=2)
    ax2.fill_between(gls_result.energy_bounds_eV[:-1],
                     gls_result.flux - gls_result.flux_unc,
                     gls_result.flux + gls_result.flux_unc,
                     alpha=0.3, step='post', color='green')
    
    # Add model spectrum if provided
    if model_spectrum is not None:
        ax2.step(model_spectrum['energy_eV'], model_spectrum['flux'], where='post',
                 label='Model (MCNP)', color='red', linewidth=2, linestyle='--')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Flux (n/cm²/s/eV)')
    ax2.set_title('GLS Continuous Unfolding')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e-2, 2e7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def load_model_spectrum(model_file: Path):
    """Load a model spectrum for comparison (e.g., from MCNP)."""
    if not model_file.exists():
        return None
    
    # Try to parse MCNP tally or simple CSV format
    energy = []
    flux = []
    
    try:
        with open(model_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        e = float(parts[0])
                        f_val = float(parts[1])
                        energy.append(e)
                        flux.append(f_val)
                    except ValueError:
                        continue
        
        if energy and flux:
            return {'energy_eV': np.array(energy), 'flux': np.array(flux)}
    except Exception:
        pass
    
    return None


def main():
    """Run the complete flux wire unfolding demonstration."""
    print("\n" + "#" * 80)
    print("# FLUX WIRE NEUTRON SPECTRUM UNFOLDING DEMONSTRATION")
    print("#" * 80)
    
    # Data directories
    base_dir = Path("/filespace/s/smandych/CAE/projects/ALARA/rafm_irradiation_ldrd")
    proc_dir = base_dir / "irradiation_QG_processed/flux_wires"
    raw_dir = base_dir / "raw_gamma_spec/flux_wires"
    output_dir = Path("/filespace/s/smandych/CAE/projects/ALARA/FluxForge/flux_unfolding_results")
    output_dir.mkdir(exist_ok=True)
    
    # Optional: load a model spectrum from MCNP for comparison
    model_file = Path("/filespace/s/smandych/CAE/projects/ALARA/spectra_files/whale_spectrum.txt")
    model_spectrum = load_model_spectrum(model_file) if model_file.exists() else None
    
    # Run demonstrations
    demonstrate_sample_isotope_mapping()
    
    if proc_dir.exists():
        demonstrate_processed_file_analysis(proc_dir)
        
        if raw_dir.exists():
            demonstrate_raw_spectrum_analysis(raw_dir, proc_dir)
        
        reactions = demonstrate_reaction_rate_extraction(proc_dir)
        
        if reactions:
            discrete_result = demonstrate_discrete_unfolding(reactions)
            gls_result = demonstrate_gls_unfolding(reactions)
            
            # Save results
            save_spectrum_to_csv(
                discrete_result,
                gls_result,
                output_dir / "unfolded_spectrum.csv"
            )
            
            # Plot comparison
            plot_unfolded_spectrum(
                discrete_result,
                gls_result,
                output_dir / "unfolded_spectrum_comparison.png",
                model_spectrum=model_spectrum,
            )
    else:
        print(f"\nData directory not found: {proc_dir}")
        print("Please update the paths in this script.")
    
    print_header("DEMONSTRATION COMPLETE")
    print("\nThe unfolded neutron spectrum represents the flux distribution")
    print("that would produce the measured activation rates in the flux wires.")
    print("\nKey observations:")
    print("  - Thermal peak from Co, Cu, Sc (n,γ) reactions")
    print("  - Epithermal peak from In resonances")
    print("  - Fast flux from Ti, Ni threshold reactions")
    print("\nExpected flux in MURR whale tube: ~10^13 n/cm²/s (1 MW reactor)")


if __name__ == "__main__":
    main()
