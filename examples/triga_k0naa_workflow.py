#!/usr/bin/env python3
"""
TRIGA Reactor k0-NAA Workflow with UWNR Flux Wire Data

This example demonstrates the complete k0-standardization NAA workflow
specifically designed for TRIGA reactors using actual UWNR flux wire data.

Key TRIGA-specific aspects:
1. Cadmium-ratio method for f and α determination
2. Multiple flux monitors (Co, Sc, In) with bare/Cd-covered pairs
3. Epithermal flux characterization typical of TRIGA cores
4. Multi-energy bin flux spectrum unfolding

References:
- De Corte et al., "The updated NAA nuclear data library derived 
  from the Y2K k0-database" (2003)
- Jacimovic et al., "k0-NAA quality assessment by analysis of 
  different certified reference materials" (2017)
- IAEA-TECDOC-1215, "Use of research reactors for NAA"
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add FluxForge to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fluxforge.analysis import (
    # k0-NAA
    K0Parameters,
    K0Measurement,
    K0Result,
    K0Calculator,
    K0_DATABASE,
    get_k0_data,
    sdc_factor,
    calculate_Q0_alpha,
    calculate_k0_parameters,
    # Flux unfolding
    extract_reactions_from_processed,
    unfold_discrete_bins,
    unfold_gls,
    THERMAL_CROSS_SECTIONS,
    REACTION_ENERGIES,
)
from fluxforge.io import read_spe_file


# =============================================================================
# TRIGA-Specific Constants and Functions
# =============================================================================

# Cadmium cutoff energy (eV) - standard value
E_CD = 0.55  # eV

# TRIGA reactor characteristics (typical values)
TRIGA_CHARACTERISTICS = {
    'core_power_MW': 1.0,  # UWNR is 1 MW TRIGA
    'fuel_type': 'TRIGA LEU',  # Low enriched uranium
    'moderator': 'ZrH',  # Zirconium hydride in fuel
    'coolant': 'light water',
    'typical_f': (15, 35),  # f range in irradiation positions
    'typical_alpha': (-0.05, 0.10),  # α range
}


@dataclass
class CdRatioMeasurement:
    """
    Bare and Cd-covered flux monitor measurements for Cd-ratio method.
    
    The Cadmium ratio RCd = A_bare / A_Cd relates to flux parameters:
        RCd = (f + Q0(α)) / Q0(α)
    
    For α = 0:
        f = Q0 * (RCd - 1)
    """
    isotope: str
    bare_activity: float  # Bq
    bare_unc: float
    cd_activity: float  # Bq
    cd_unc: float
    Q0: float  # I0/σ0
    E_res: float  # eV
    half_life_s: float
    

def calculate_cd_ratio(bare: float, cd: float) -> Tuple[float, float]:
    """
    Calculate Cadmium ratio and its uncertainty.
    
    Parameters
    ----------
    bare : float
        Bare (uncovered) activity
    cd : float
        Cd-covered activity
        
    Returns
    -------
    tuple
        (R_Cd, R_Cd_unc)
    """
    if cd <= 0:
        return float('inf'), 0
    
    R_Cd = bare / cd
    # Propagated uncertainty assuming 5% on each
    R_Cd_unc = R_Cd * np.sqrt(0.05**2 + 0.05**2)
    
    return R_Cd, R_Cd_unc


def calculate_f_from_cd_ratio(R_Cd: float, Q0: float, alpha: float = 0.0) -> float:
    """
    Calculate f parameter from Cd ratio.
    
    For α = 0:
        f = Q0 * (RCd - 1)
        
    For α ≠ 0, iterative solution needed:
        f = Q0(α) * (RCd - 1)
        
    Parameters
    ----------
    R_Cd : float
        Cadmium ratio (A_bare / A_Cd)
    Q0 : float
        Q0 factor for the monitor
    alpha : float
        Epithermal shape parameter
        
    Returns
    -------
    float
        f = φ_thermal / φ_epithermal
    """
    # For small α, first-order approximation
    Q0_alpha = Q0  # Would need E_res for full calculation
    f = Q0_alpha * (R_Cd - 1)
    return f


def calculate_alpha_two_monitors(
    R_Cd_1: float, Q0_1: float, E_res_1: float,
    R_Cd_2: float, Q0_2: float, E_res_2: float,
) -> float:
    """
    Calculate α from two flux monitors with different E_res.
    
    Using the ratio of Cd ratios:
        ln(R1/R2) ≈ α * ln(E_res_2/E_res_1)
        
    Parameters
    ----------
    R_Cd_1, R_Cd_2 : float
        Cd ratios for monitors 1 and 2
    Q0_1, Q0_2 : float
        Q0 factors
    E_res_1, E_res_2 : float
        Effective resonance energies (eV)
        
    Returns
    -------
    float
        Estimated α
    """
    if E_res_1 <= 0 or E_res_2 <= 0 or E_res_1 == E_res_2:
        return 0.0
    
    # Simplified approach - would need iterative solution for full accuracy
    ratio = (R_Cd_1 - 1) / (R_Cd_2 - 1) * Q0_2 / Q0_1
    if ratio <= 0:
        return 0.0
    
    alpha = np.log(ratio) / np.log(E_res_2 / E_res_1)
    
    return alpha


# =============================================================================
# Load and Process UWNR Flux Wire Data
# =============================================================================

def load_uwnr_flux_wire_data(results_path: Path) -> pd.DataFrame:
    """
    Load the processed flux wire reaction data from UWNR experiments.
    
    Parameters
    ----------
    results_path : Path
        Path to flux_unfolding_results directory
        
    Returns
    -------
    DataFrame
        Reaction data with activities and fluxes
    """
    csv_path = results_path / "reaction_data.csv"
    
    # Read CSV (properly quoted to handle reactions like (n,g))
    df = pd.read_csv(csv_path)
    
    return df


def extract_cd_ratio_pairs(df: pd.DataFrame) -> Dict[str, CdRatioMeasurement]:
    """
    Extract bare/Cd-covered pairs for Cd-ratio method.
    
    Parameters
    ----------
    df : DataFrame
        Reaction data
        
    Returns
    -------
    dict
        CdRatioMeasurement objects keyed by element
    """
    # Nuclear data for Cd-ratio monitors
    monitor_data = {
        'Co': {
            'Q0': 1.99,
            'E_res': 132.0,  # eV
            'half_life_s': 5.2714 * 365.25 * 24 * 3600,
            'reaction': 'Co-59(n,g)Co-60',
            'isotope': 'Co60',
        },
        'Sc': {
            'Q0': 0.43,
            'E_res': 4000.0,  # eV (higher resonance)
            'half_life_s': 83.79 * 24 * 3600,
            'reaction': 'Sc-45(n,g)Sc-46',
            'isotope': 'Sc46',
        },
        'In': {
            'Q0': 17.2,  # For In-115 thermal capture
            'E_res': 1.457,  # eV (main resonance)
            'half_life_s': 49.51 * 24 * 3600,
            'reaction': 'In-113(n,g)In-114m',
            'isotope': 'In114m',
        },
        'Cu': {
            'Q0': 1.11,
            'E_res': 579.0,  # eV
            'half_life_s': 12.7 * 3600,
            'reaction': 'Cu-63(n,g)Cu-64',
            'isotope': 'Cu64',
        },
    }
    
    pairs = {}
    
    for element, data in monitor_data.items():
        # Find Cd-covered samples first (element-Cd pattern)
        cd_mask = (
            df['Sample'].str.upper().str.contains(f'{element.upper()}-CD', regex=False) &
            (df['Isotope'] == data['isotope'])
        )
        
        # Find bare samples (element at start, NOT Cd in name)
        bare_mask = (
            df['Sample'].str.upper().str.startswith(element.upper()) & 
            ~df['Sample'].str.upper().str.contains('-CD', regex=False) &
            (df['Isotope'] == data['isotope'])
        )
        
        bare_rows = df[bare_mask]
        cd_rows = df[cd_mask]
        
        if len(bare_rows) > 0 and len(cd_rows) > 0:
            # Take first matching pair
            bare_act = bare_rows['Activity_Bq'].iloc[0]
            cd_act = cd_rows['Activity_Bq'].iloc[0]
            
            pairs[element] = CdRatioMeasurement(
                isotope=element,
                bare_activity=bare_act,
                bare_unc=bare_act * 0.05,
                cd_activity=cd_act,
                cd_unc=cd_act * 0.05,
                Q0=data['Q0'],
                E_res=data['E_res'],
                half_life_s=data['half_life_s'],
            )
            print(f"  Found {element} pair: bare={bare_act:.2e}, Cd={cd_act:.2e}")
            
    return pairs


def characterize_triga_flux(pairs: Dict[str, CdRatioMeasurement]) -> K0Parameters:
    """
    Characterize TRIGA flux using Cd-ratio method.
    
    Uses multiple monitors to determine f and α.
    
    Parameters
    ----------
    pairs : dict
        CdRatioMeasurement objects
        
    Returns
    -------
    K0Parameters
        Characterized flux parameters
    """
    print("\n" + "=" * 70)
    print("TRIGA FLUX CHARACTERIZATION (Cd-Ratio Method)")
    print("=" * 70)
    
    # Calculate Cd ratios for each monitor
    cd_ratios = {}
    for element, meas in pairs.items():
        R_Cd, R_unc = calculate_cd_ratio(meas.bare_activity, meas.cd_activity)
        cd_ratios[element] = (R_Cd, R_unc)
        print(f"\n{element} monitor:")
        print(f"  Bare activity:   {meas.bare_activity:.2e} Bq")
        print(f"  Cd-covered:      {meas.cd_activity:.2e} Bq")
        print(f"  Cd ratio (RCd):  {R_Cd:.2f} ± {R_unc:.2f}")
        print(f"  Q0 = {meas.Q0:.2f}, E_res = {meas.E_res:.1f} eV")
    
    # Calculate f using each monitor (assuming α ≈ 0 initially)
    f_values = []
    print("\nEstimated f from each monitor (α = 0):")
    for element, meas in pairs.items():
        R_Cd = cd_ratios[element][0]
        if R_Cd > 1:
            f = calculate_f_from_cd_ratio(R_Cd, meas.Q0)
            f_values.append(f)
            print(f"  {element}: f = {f:.1f}")
    
    # Average f value
    f_avg = np.mean(f_values) if f_values else 25.0
    f_std = np.std(f_values) if len(f_values) > 1 else f_avg * 0.1
    
    # Calculate α using two monitors with different E_res
    alpha = 0.0
    if 'Co' in pairs and 'Sc' in pairs:
        alpha = calculate_alpha_two_monitors(
            cd_ratios['Co'][0], pairs['Co'].Q0, pairs['Co'].E_res,
            cd_ratios['Sc'][0], pairs['Sc'].Q0, pairs['Sc'].E_res,
        )
        # Clamp to reasonable range
        alpha = np.clip(alpha, -0.1, 0.2)
    
    print(f"\nFinal flux characterization:")
    print(f"  f = {f_avg:.1f} ± {f_std:.1f}")
    print(f"  α = {alpha:.3f}")
    
    # Estimate thermal flux from reaction rates
    # Using Sc as primary monitor (most reliable)
    phi_thermal = 0
    if 'Sc' in pairs:
        # A = N * σ * φ * (1 - e^(-λt)) * e^(-λt_d)
        # Simplified: φ ≈ A / (N * σ) for saturation
        # From our data, we already have flux calculated
        pass
    
    return K0Parameters(
        f=f_avg,
        alpha=alpha,
        f_uncertainty=f_std,
        alpha_uncertainty=0.02,
        phi_thermal=1e12,  # Placeholder, would get from flux calculation
    )


# =============================================================================
# UWNR Sample Analysis Example
# =============================================================================

def analyze_uwnr_sample(
    flux_params: K0Parameters,
    sample_name: str = "RAFM Steel Sample",
) -> Dict[str, K0Result]:
    """
    Example k0-NAA analysis of a sample irradiated at UWNR.
    
    This simulates what the analysis would look like for a real
    sample counted on the HPGe detector.
    
    Parameters
    ----------
    flux_params : K0Parameters
        Characterized flux parameters
    sample_name : str
        Sample identifier
        
    Returns
    -------
    dict
        Element concentrations
    """
    print("\n" + "=" * 70)
    print(f"k0-NAA ANALYSIS: {sample_name}")
    print("=" * 70)
    
    # Typical UWNR irradiation parameters
    t_irr = 8 * 3600  # 8 hour irradiation
    t_decay = 24 * 3600  # 1 day decay
    t_count = 2 * 3600  # 2 hour count
    sample_mass = 0.1  # 100 mg sample
    
    print(f"\nIrradiation conditions:")
    print(f"  Irradiation time: {t_irr/3600:.1f} hours")
    print(f"  Decay time:       {t_decay/3600:.1f} hours")
    print(f"  Counting time:    {t_count/3600:.1f} hours")
    print(f"  Sample mass:      {sample_mass*1000:.1f} mg")
    
    # Au flux monitor measurement (simulated)
    au_measurement = K0Measurement(
        product_isotope='Au-198',
        net_peak_area=180000,
        peak_area_unc=1800,
        efficiency=0.0085,
        efficiency_unc=3.0,
        t_irr=t_irr,
        t_decay=t_decay,
        t_count=t_count,
        sample_mass=0.0001,  # 0.1 mg Au foil
        gamma_energy_keV=411.8,
    )
    
    # Initialize calculator with Au reference
    calculator = K0Calculator(flux_params, au_measurement)
    
    # Simulated sample peak measurements
    # These would come from HPGe spectrum analysis in practice
    # Using isotopes commonly detected in steel samples
    sample_measurements = [
        K0Measurement(
            product_isotope='Mn-56',
            net_peak_area=125000,
            peak_area_unc=2500,
            efficiency=0.0065,
            efficiency_unc=3.0,
            t_irr=t_irr,
            t_decay=t_decay,
            t_count=t_count,
            sample_mass=sample_mass,
            gamma_energy_keV=846.8,
        ),
        K0Measurement(
            product_isotope='Cr-51',
            net_peak_area=8500,
            peak_area_unc=300,
            efficiency=0.0105,
            efficiency_unc=3.0,
            t_irr=t_irr,
            t_decay=t_decay,
            t_count=t_count,
            sample_mass=sample_mass,
            gamma_energy_keV=320.1,
        ),
        K0Measurement(
            product_isotope='Co-60',
            net_peak_area=4200,
            peak_area_unc=150,
            efficiency=0.0045,
            efficiency_unc=3.0,
            t_irr=t_irr,
            t_decay=t_decay,
            t_count=t_count,
            sample_mass=sample_mass,
            gamma_energy_keV=1332.5,
        ),
        K0Measurement(
            product_isotope='Fe-59',
            net_peak_area=2100,
            peak_area_unc=100,
            efficiency=0.0052,
            efficiency_unc=3.0,
            t_irr=t_irr,
            t_decay=t_decay,
            t_count=t_count,
            sample_mass=sample_mass,
            gamma_energy_keV=1099.2,
        ),
        K0Measurement(
            product_isotope='W-187',
            net_peak_area=1800,
            peak_area_unc=80,
            efficiency=0.0072,
            efficiency_unc=3.0,
            t_irr=t_irr,
            t_decay=t_decay,
            t_count=t_count,
            sample_mass=sample_mass,
            gamma_energy_keV=685.8,
        ),
    ]
    
    # Analyze all measurements
    print(f"\nPeak Analysis:")
    print(f"  {'Isotope':<12} {'Peak Area':<15} {'Efficiency':<12}")
    print(f"  {'-'*12} {'-'*15} {'-'*12}")
    
    results = {}
    for meas in sample_measurements:
        print(f"  {meas.product_isotope:<12} {meas.net_peak_area:<15,.0f} {meas.efficiency:<12.4f}")
        
        try:
            result = calculator.calculate_concentration(meas)
            results[result.element] = result
        except Exception as e:
            print(f"    Error: {e}")
    
    # Results table
    print(f"\n{'='*70}")
    print("CONCENTRATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Element':<8} {'Conc (μg/g)':<15} {'Uncertainty':<12} {'LOD':<12} {'k0':<10}")
    print(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    
    for element in sorted(results.keys()):
        result = results[element]
        print(f"  {element:<8} {result.concentration_ug_g:<15.2f} "
              f"±{result.concentration_unc:<11.2f} {result.detection_limit_ug_g:<12.2f} "
              f"{result.k0_used:<10.4f}")
    
    # Comparison with typical RAFM steel composition
    print(f"\n{'='*70}")
    print("COMPARISON WITH TYPICAL RAFM STEEL COMPOSITION")
    print(f"{'='*70}")
    
    rafm_composition = {
        'Cr': 9.0e4,    # 9% Cr → 90,000 μg/g
        'Mn': 5000,     # 0.5% Mn → 5,000 μg/g
        'W': 20000,     # 2% W → 20,000 μg/g
        'Co': 100,      # ~100 ppm Co
        'Fe': 880000,   # ~88% Fe → balance
    }
    
    print(f"\n  {'Element':<8} {'Measured':<15} {'Expected':<15} {'Ratio':<10}")
    print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*10}")
    
    for element in sorted(results.keys()):
        if element in rafm_composition:
            measured = results[element].concentration_ug_g
            expected = rafm_composition[element]
            ratio = measured / expected if expected > 0 else 0
            print(f"  {element:<8} {measured:<15.1f} {expected:<15.1f} {ratio:<10.3f}")
    
    return results


# =============================================================================
# Flux Spectrum Analysis
# =============================================================================

def plot_flux_spectrum_from_wires(results_path: Path, output_path: Path):
    """
    Generate comprehensive flux spectrum plot from flux wire data.
    
    Parameters
    ----------
    results_path : Path
        Path to flux_unfolding_results
    output_path : Path
        Path for output plot
    """
    # Load discrete and GLS spectra
    discrete_df = pd.read_csv(results_path / "discrete_spectrum.csv", comment='#')
    gls_df = pd.read_csv(results_path / "gls_spectrum.csv", comment='#')
    
    # Calculate center energy for GLS spectrum
    gls_df['E_center_eV'] = np.sqrt(gls_df['E_low_eV'] * gls_df['E_high_eV'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Discrete flux bins
    ax1 = axes[0, 0]
    ax1.bar(range(len(discrete_df)), discrete_df['Flux_n_cm2_s'], 
            color='steelblue', alpha=0.7, edgecolor='navy')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy Bin')
    ax1.set_ylabel('Flux (n/cm²/s)')
    ax1.set_title('Discrete Energy Bin Flux (UWNR)')
    ax1.grid(True, alpha=0.3)
    
    # 2. GLS unfolded spectrum
    ax2 = axes[0, 1]
    ax2.errorbar(gls_df['E_center_eV'], gls_df['Flux_n_cm2_s'],
                 yerr=gls_df['Flux_unc'],
                 fmt='o-', color='darkgreen', markersize=4,
                 capsize=3, label='GLS Unfolded')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Flux (n/cm²/s)')
    ax2.set_title('GLS Unfolded Spectrum (UWNR)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Lethargy plot (E * φ)
    ax3 = axes[1, 0]
    E = gls_df['E_center_eV'].values
    phi = gls_df['Flux_n_cm2_s'].values
    E_phi = E * phi
    ax3.semilogx(E, E_phi, 'o-', color='darkred', markersize=4)
    ax3.set_xlabel('Energy (eV)')
    ax3.set_ylabel('E × φ(E) (n/cm²/s × eV)')
    ax3.set_title('Lethargy Plot (E × φ)')
    ax3.grid(True, alpha=0.3)
    
    # Add TRIGA characteristic regions
    ax3.axvspan(1e-3, 0.5, alpha=0.2, color='blue', label='Thermal')
    ax3.axvspan(0.5, 1e5, alpha=0.2, color='green', label='Epithermal')
    ax3.axvspan(1e5, 2e7, alpha=0.2, color='red', label='Fast')
    ax3.legend(loc='upper right')
    
    # 4. Reaction data summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Load reaction data
    reaction_df = pd.read_csv(results_path / "reaction_data.csv")
    
    # Create summary table
    summary_text = "UWNR Flux Wire Summary\n" + "="*50 + "\n\n"
    summary_text += f"{'Reaction':<25} {'Flux (n/cm²/s)':<15}\n"
    summary_text += "-"*45 + "\n"
    
    for _, row in reaction_df.head(10).iterrows():
        summary_text += f"{row['Reaction']:<25} {row['Flux_n_cm2_s']:.2e}\n"
    
    summary_text += "\n" + "-"*45 + "\n"
    summary_text += f"Total reactions: {len(reaction_df)}\n"
    summary_text += f"Thermal flux:    {reaction_df[reaction_df['Flux_type']=='thermal']['Flux_n_cm2_s'].mean():.2e}\n"
    summary_text += f"Fast flux:       {reaction_df[reaction_df['Flux_type']=='fast']['Flux_n_cm2_s'].mean():.2e}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved flux spectrum plot to: {output_path}")


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """Run complete TRIGA k0-NAA workflow with UWNR data."""
    print("\n" + "=" * 70)
    print("FLUXFORGE: TRIGA REACTOR k0-NAA WORKFLOW")
    print("University of Wisconsin Nuclear Reactor (UWNR)")
    print("=" * 70)
    
    # Paths
    fluxforge_dir = Path(__file__).parent.parent
    results_path = fluxforge_dir / "flux_unfolding_results"
    output_dir = fluxforge_dir / "examples" / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load flux wire data
    print("\n" + "-" * 70)
    print("STEP 1: Loading UWNR Flux Wire Data")
    print("-" * 70)
    
    if not (results_path / "reaction_data.csv").exists():
        print("  ERROR: reaction_data.csv not found!")
        print("  Run complete_flux_unfolding.py first.")
        return
    
    df = load_uwnr_flux_wire_data(results_path)
    print(f"  Loaded {len(df)} reaction measurements")
    
    # Step 2: Extract Cd-ratio pairs
    print("\n" + "-" * 70)
    print("STEP 2: Extracting Cd-Ratio Pairs for Flux Characterization")
    print("-" * 70)
    
    pairs = extract_cd_ratio_pairs(df)
    print(f"  Found {len(pairs)} Cd-ratio pairs: {list(pairs.keys())}")
    
    # Step 3: Characterize TRIGA flux
    flux_params = characterize_triga_flux(pairs)
    
    # Step 4: Analyze sample
    results = analyze_uwnr_sample(flux_params, "EUROFER-97 RAFM Steel")
    
    # Step 5: Generate plots
    print("\n" + "-" * 70)
    print("STEP 5: Generating Plots")
    print("-" * 70)
    
    plot_flux_spectrum_from_wires(results_path, output_dir / "uwnr_flux_spectrum.png")
    
    # Step 6: Export results
    print("\n" + "-" * 70)
    print("STEP 6: Exporting Results")
    print("-" * 70)
    
    # Export to CSV
    results_df = pd.DataFrame([
        {
            'Element': r.element,
            'Isotope': r.product_isotope,
            'Concentration_ug_g': r.concentration_ug_g,
            'Uncertainty_ug_g': r.concentration_unc,
            'Detection_Limit_ug_g': r.detection_limit_ug_g,
            'k0_factor': r.k0_used,
            'Q0_alpha': r.Q0_alpha_used,
        }
        for r in results.values()
    ])
    
    csv_path = output_dir / "uwnr_k0naa_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved results to: {csv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nTRIGA flux characterization:")
    print(f"  f = {flux_params.f:.1f} ± {flux_params.f_uncertainty:.1f}")
    print(f"  α = {flux_params.alpha:.3f}")
    print(f"\nElements quantified: {len(results)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
