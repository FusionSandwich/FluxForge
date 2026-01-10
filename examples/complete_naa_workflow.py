#!/usr/bin/env python3
"""
Complete NAA Workflow Example

This example demonstrates the complete FluxForge NAA workflow:
1. Flux wire analysis and spectrum unfolding
2. k0-NAA concentration calculations
3. Optional ANN-based analysis

This integrates:
- Flux wire processing for neutron flux characterization
- k0-standardization for absolute concentration determination
- Multi-bin spectrum unfolding for spectral analysis
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
    # Flux unfolding
    extract_reactions_from_processed,
    unfold_discrete_bins,
    unfold_gls,
    THERMAL_CROSS_SECTIONS,
)
from fluxforge.io import read_spe_file


def demonstrate_k0_naa_workflow():
    """
    Demonstrate k0-NAA concentration calculation workflow.
    
    This shows how to use the k0-standardization method to calculate
    absolute element concentrations from gamma spectroscopy data.
    """
    print("=" * 70)
    print("k0-NAA DEMONSTRATION")
    print("=" * 70)
    
    # ==========================================================================
    # Step 1: Set up flux characterization parameters
    # ==========================================================================
    print("\n1. FLUX CHARACTERIZATION")
    print("-" * 40)
    
    # Typical values for a research reactor thermal column
    # f = φ_thermal / φ_epithermal (typically 20-50 for well-thermalized flux)
    # α = epithermal shape parameter (deviation from 1/E, typically 0-0.1)
    
    flux_params = K0Parameters(
        f=25.0,               # Thermal/epithermal ratio
        alpha=0.02,           # Epithermal shape parameter
        f_uncertainty=2.0,    # 2% uncertainty
        alpha_uncertainty=0.01,
        phi_thermal=1e13,     # n/cm²/s (for absolute method)
    )
    
    print(f"  f (thermal/epithermal): {flux_params.f:.1f} ± {flux_params.f_uncertainty:.1f}")
    print(f"  α (epithermal shape):   {flux_params.alpha:.3f} ± {flux_params.alpha_uncertainty:.3f}")
    print(f"  φ_thermal:              {flux_params.phi_thermal:.2e} n/cm²/s")
    
    # ==========================================================================
    # Step 2: Set up Au flux monitor measurement
    # ==========================================================================
    print("\n2. GOLD FLUX MONITOR")
    print("-" * 40)
    
    # Example Au-198 measurement (411.8 keV gamma)
    au_measurement = K0Measurement(
        product_isotope='Au-198',
        net_peak_area=250000,      # counts
        peak_area_unc=2500,        # 1% uncertainty
        efficiency=0.0085,         # HPGe efficiency at 411.8 keV
        efficiency_unc=3.0,        # 3% efficiency uncertainty
        t_irr=2 * 3600,            # 2 hour irradiation
        t_decay=24 * 3600,         # 1 day decay
        t_count=1 * 3600,          # 1 hour count
        sample_mass=0.0001,        # 0.1 mg Au foil
        gamma_energy_keV=411.8,
    )
    
    # Calculate SDC factors for Au
    au_data = get_k0_data('Au-198')
    S, D, C, SDC = sdc_factor(
        au_data.half_life_s,
        au_measurement.t_irr,
        au_measurement.t_decay,
        au_measurement.t_count,
    )
    
    print(f"  Au-198 measurement:")
    print(f"    Peak area: {au_measurement.net_peak_area:,.0f} ± {au_measurement.peak_area_unc:.0f} counts")
    print(f"    Efficiency: {au_measurement.efficiency:.4f}")
    print(f"    Irradiation: {au_measurement.t_irr/3600:.1f} hours")
    print(f"    Decay: {au_measurement.t_decay/3600:.1f} hours")
    print(f"    Counting: {au_measurement.t_count/3600:.1f} hours")
    print(f"  SDC factors:")
    print(f"    S (saturation): {S:.4f}")
    print(f"    D (decay):      {D:.4f}")
    print(f"    C (counting):   {C:.4f}")
    print(f"    SDC product:    {SDC:.6f}")
    
    # ==========================================================================
    # Step 3: Initialize k0 calculator
    # ==========================================================================
    print("\n3. K0 CALCULATOR INITIALIZATION")
    print("-" * 40)
    
    calculator = K0Calculator(flux_params, au_measurement)
    print("  Calculator initialized with Au reference")
    
    # Show Q0(α) correction for some nuclides
    print("\n  Q0(α) corrections:")
    for isotope in ['Co-60', 'Na-24', 'Sc-46', 'Fe-59']:
        data = get_k0_data(isotope)
        if data:
            Q0_alpha = calculate_Q0_alpha(data.Q0, flux_params.alpha, data.E_res_eV)
            print(f"    {isotope}: Q0 = {data.Q0:.2f} → Q0(α) = {Q0_alpha:.2f}")
    
    # ==========================================================================
    # Step 4: Analyze sample peaks
    # ==========================================================================
    print("\n4. SAMPLE ANALYSIS")
    print("-" * 40)
    
    # Simulated sample measurements (as if from gamma spectrum analysis)
    sample_measurements = [
        K0Measurement(
            product_isotope='Co-60',
            net_peak_area=15000,
            peak_area_unc=300,
            efficiency=0.0045,      # Lower efficiency at 1332 keV
            efficiency_unc=3.0,
            t_irr=2 * 3600,
            t_decay=24 * 3600,
            t_count=1 * 3600,
            sample_mass=0.5,        # 0.5 g sample
            gamma_energy_keV=1332.5,
        ),
        K0Measurement(
            product_isotope='Sc-46',
            net_peak_area=8500,
            peak_area_unc=200,
            efficiency=0.0058,
            efficiency_unc=3.0,
            t_irr=2 * 3600,
            t_decay=24 * 3600,
            t_count=1 * 3600,
            sample_mass=0.5,
            gamma_energy_keV=889.3,
        ),
        K0Measurement(
            product_isotope='Na-24',
            net_peak_area=45000,
            peak_area_unc=800,
            efficiency=0.0042,
            efficiency_unc=3.0,
            t_irr=2 * 3600,
            t_decay=24 * 3600,
            t_count=1 * 3600,
            sample_mass=0.5,
            gamma_energy_keV=1368.6,
        ),
    ]
    
    print("  Sample: 0.5 g geological sample")
    print("  Irradiation conditions: same as Au monitor")
    print()
    
    results = {}
    for meas in sample_measurements:
        try:
            result = calculator.calculate_concentration(meas)
            results[result.element] = result
            
            print(f"  {result.element} (from {result.product_isotope}):")
            print(f"    Concentration: {result.concentration_ug_g:.2f} ± {result.concentration_unc:.2f} μg/g")
            print(f"    Detection limit: {result.detection_limit_ug_g:.2f} μg/g")
            print(f"    k0 used: {result.k0_used:.4f}")
            print()
            
        except Exception as e:
            print(f"  Error analyzing {meas.product_isotope}: {e}")
    
    # ==========================================================================
    # Step 5: Summary table
    # ==========================================================================
    print("\n5. RESULTS SUMMARY")
    print("-" * 40)
    print(f"  {'Element':<8} {'Concentration':<20} {'LOD':<15} {'k0':<10}")
    print(f"  {'-'*8} {'-'*20} {'-'*15} {'-'*10}")
    
    for element, result in results.items():
        conc_str = f"{result.concentration_ug_g:.2f} ± {result.concentration_unc:.2f}"
        print(f"  {element:<8} {conc_str:<20} {result.detection_limit_ug_g:<15.2f} {result.k0_used:<10.4f}")
    
    return results


def demonstrate_flux_wire_k0_integration():
    """
    Demonstrate integration of flux wire analysis with k0-NAA.
    
    Shows how flux wire measurements can characterize the neutron
    spectrum, which then feeds into k0-NAA calculations.
    """
    print("\n" + "=" * 70)
    print("FLUX WIRE + k0-NAA INTEGRATION")
    print("=" * 70)
    
    # ==========================================================================
    # Step 1: Load flux wire data (simulated here)
    # ==========================================================================
    print("\n1. FLUX WIRE DATA")
    print("-" * 40)
    
    # Simulated flux wire reaction rates from processed spectra
    # These would come from extract_reactions_from_processed() in practice
    flux_wire_reactions = {
        'Au-197(n,g)Au-198': {
            'flux': 2.5e13,           # n/cm²/s
            'flux_unc': 0.25e13,
            'energy_eV': 0.0253,      # Thermal
            'cross_section': 98.65,   # barns
        },
        'Co-59(n,g)Co-60': {
            'flux': 2.3e13,
            'flux_unc': 0.30e13,
            'energy_eV': 0.0253,
            'cross_section': 37.18,
        },
        'Mn-55(n,g)Mn-56': {
            'flux': 2.6e13,
            'flux_unc': 0.20e13,
            'energy_eV': 0.0253,
            'cross_section': 13.3,
        },
        'In-115(n,g)In-116m': {
            'flux': 1.8e12,           # Resonance region
            'flux_unc': 0.25e12,
            'energy_eV': 1.457,       # Main resonance
            'cross_section': 162.3,
        },
        'Fe-54(n,p)Mn-54': {
            'flux': 5.2e11,           # Fast neutrons
            'flux_unc': 0.8e11,
            'energy_eV': 3e6,         # ~3 MeV threshold
            'cross_section': 0.085,
        },
    }
    
    print("  Reaction rates from flux wires:")
    for reaction, data in flux_wire_reactions.items():
        print(f"    {reaction}:")
        print(f"      Flux: {data['flux']:.2e} ± {data['flux_unc']:.2e} n/cm²/s")
        print(f"      Energy: {data['energy_eV']:.2e} eV")
    
    # ==========================================================================
    # Step 2: Estimate f and α from flux wire data
    # ==========================================================================
    print("\n2. FLUX CHARACTERIZATION FROM WIRES")
    print("-" * 40)
    
    # Thermal flux (average of thermal-sensitive reactions)
    phi_thermal = np.mean([
        flux_wire_reactions['Au-197(n,g)Au-198']['flux'],
        flux_wire_reactions['Co-59(n,g)Co-60']['flux'],
        flux_wire_reactions['Mn-55(n,g)Mn-56']['flux'],
    ])
    
    # Epithermal flux (from resonance-sensitive In)
    phi_epithermal = flux_wire_reactions['In-115(n,g)In-116m']['flux']
    
    # Fast flux
    phi_fast = flux_wire_reactions['Fe-54(n,p)Mn-54']['flux']
    
    # Calculate f
    f_estimated = phi_thermal / phi_epithermal
    
    print(f"  φ_thermal:    {phi_thermal:.2e} n/cm²/s")
    print(f"  φ_epithermal: {phi_epithermal:.2e} n/cm²/s")
    print(f"  φ_fast:       {phi_fast:.2e} n/cm²/s")
    print(f"  f = φ_th/φ_epi: {f_estimated:.1f}")
    
    # Estimate α from ratio of different resonance-weighted fluxes
    # This is simplified - in practice would use Cd ratio method
    alpha_estimated = 0.05  # Placeholder
    
    # ==========================================================================
    # Step 3: Create K0Parameters from flux wire data
    # ==========================================================================
    print("\n3. K0 PARAMETERS FROM FLUX WIRES")
    print("-" * 40)
    
    flux_params = K0Parameters(
        f=f_estimated,
        alpha=alpha_estimated,
        f_uncertainty=f_estimated * 0.05,  # 5% uncertainty
        alpha_uncertainty=0.02,
        phi_thermal=phi_thermal,
    )
    
    print(f"  f:        {flux_params.f:.1f}")
    print(f"  α:        {flux_params.alpha:.3f}")
    print(f"  φ_thermal: {flux_params.phi_thermal:.2e} n/cm²/s")
    
    # ==========================================================================
    # Step 4: Use characterized flux for k0-NAA
    # ==========================================================================
    print("\n4. k0-NAA WITH CHARACTERIZED FLUX")
    print("-" * 40)
    
    # Now we can do k0-NAA with known flux
    # In practice, this would use real sample measurements
    
    print("  Ready for k0-NAA with flux-wire characterized spectrum")
    print("  - Use K0Calculator(flux_params, au_measurement)")
    print("  - Calculate concentrations with flux_params.phi_thermal")
    
    return flux_params


def show_k0_database():
    """Display the k0 nuclear data database."""
    print("\n" + "=" * 70)
    print("k0 NUCLEAR DATA DATABASE")
    print("=" * 70)
    
    print(f"\n  {'Isotope':<12} {'Element':<8} {'k0':<10} {'Q0':<10} {'E_γ (keV)':<12} {'t½':<15}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")
    
    for isotope in sorted(K0_DATABASE.keys()):
        data = K0_DATABASE[isotope]
        
        # Format half-life
        t_half = data.half_life_s
        if t_half > 365.25 * 24 * 3600:
            t_str = f"{t_half / (365.25 * 24 * 3600):.2f} y"
        elif t_half > 24 * 3600:
            t_str = f"{t_half / (24 * 3600):.2f} d"
        elif t_half > 3600:
            t_str = f"{t_half / 3600:.2f} h"
        else:
            t_str = f"{t_half:.1f} s"
        
        print(f"  {isotope:<12} {data.element:<8} {data.k0_Au:<10.4f} {data.Q0:<10.2f} "
              f"{data.gamma_energy_keV:<12.1f} {t_str:<15}")


def main():
    """Run complete NAA workflow demonstration."""
    print("\n" + "=" * 70)
    print("FLUXFORGE COMPLETE NAA WORKFLOW DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstrates the integration of:")
    print("  1. k0-NAA concentration calculations")
    print("  2. Flux wire characterization")
    print("  3. Neutron spectrum unfolding")
    print()
    
    # Show k0 database
    show_k0_database()
    
    # Demonstrate k0-NAA workflow
    k0_results = demonstrate_k0_naa_workflow()
    
    # Demonstrate flux wire integration
    flux_params = demonstrate_flux_wire_k0_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print("\nKey capabilities demonstrated:")
    print("  ✓ k0-standardization for absolute NAA")
    print("  ✓ SDC correction factors (saturation, decay, counting)")
    print("  ✓ Q0(α) epithermal corrections")
    print("  ✓ Flux wire integration for spectrum characterization")
    print("  ✓ Nuclear data database with 12+ nuclides")
    print("\nNext steps:")
    print("  - Connect to real gamma spectra from HPGe detector")
    print("  - Use actual flux wire measurements")
    print("  - Implement NAA-ANN for ML-based quantification")


if __name__ == '__main__':
    main()
