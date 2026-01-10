#!/usr/bin/env python3
"""
Cadmium Cover Correction Example
================================

This example demonstrates the STAYSL PNNL-compatible cover correction
workflow for flux-wire/foil dosimetry analysis.

Two modes are available:
1. STAYSL Parity Mode: Scalar CCF using E₂(x) exponential integral
2. Best-Physics Mode: Energy-dependent transmission T(E)

For TRIGA whale-tube irradiations, the isotropic angular model is typically used.

Reference: STAYSL PNNL Manual, Section 6.4
"""

import numpy as np
import matplotlib.pyplot as plt

from fluxforge.corrections.covers import (
    # STAYSL parity mode
    CoverSpec,
    FluxAngularModel,
    STAYSL_COVER_DATA,
    compute_optical_thickness,
    compute_ccf_staysl,
    compute_staysl_cover_correction,
    exponential_integral_E2,
    # Energy-dependent mode
    compute_energy_dependent_cover_corrections,
    create_cd_sigma_total_1v,
    create_staysl_parity_report,
    # Legacy mode (still available)
    CoverMaterial,
    CoverConfiguration,
    calculate_cover_corrections,
)


def example_staysl_parity_mode():
    """
    Demonstrate STAYSL PNNL CCF computation.
    
    This is the recommended mode for parity testing with STAYSL.
    """
    print("=" * 70)
    print("STAYSL PNNL Parity Mode: Scalar CCF Computation")
    print("=" * 70)
    
    # Standard 40-mil (1 mm) Cd cover for whale tube irradiation
    cover = CoverSpec(
        material_code="CADM",
        thickness_mil=40.0,  # 40 mils = 1.016 mm
        angular_model=FluxAngularModel.ISOTROPIC,  # Typical for reactor cores
    )
    
    print(f"\nCover Configuration:")
    print(f"  Material: {cover.material_code}")
    print(f"  Thickness: {cover.thickness_mil} mil = {cover.thickness_cm:.4f} cm")
    print(f"  Density: {cover.density} g/cm³")
    print(f"  Atomic Mass: {cover.atomic_mass} g/mol")
    print(f"  σ_th (thermal): {cover.sigma_th} barns")
    print(f"  Angular Model: {cover.angular_model.value}")
    
    # Compute optical thickness
    x = compute_optical_thickness(cover)
    print(f"\nOptical Thickness x = {x:.4f}")
    
    # Compute CCF
    ccf = compute_ccf_staysl(cover)
    print(f"CCF (E₂(x) for isotropic): {ccf:.6e}")
    
    # Compare with beam model
    cover_beam = CoverSpec(
        material_code="CADM",
        thickness_mil=40.0,
        angular_model=FluxAngularModel.BEAM,
    )
    ccf_beam = compute_ccf_staysl(cover_beam)
    print(f"CCF (exp(-x) for beam):     {ccf_beam:.6e}")
    
    # Full result with provenance
    result = compute_staysl_cover_correction(cover)
    print(f"\n--- STAYSL Cover Result Artifact ---")
    for key, value in result.to_dict().items():
        print(f"  {key}: {value}")
    
    return result


def example_compare_thicknesses():
    """
    Show how CCF varies with Cd thickness.
    """
    print("\n" + "=" * 70)
    print("CCF vs Cd Thickness (STAYSL Mode)")
    print("=" * 70)
    
    thicknesses_mil = [10, 20, 30, 40, 50, 60, 80, 100]
    
    print(f"\n{'Thickness (mil)':<18} {'Thickness (cm)':<15} {'x':<12} {'CCF (iso)':<15}")
    print("-" * 60)
    
    for t_mil in thicknesses_mil:
        cover = CoverSpec(thickness_mil=t_mil, angular_model=FluxAngularModel.ISOTROPIC)
        x = compute_optical_thickness(cover)
        ccf = compute_ccf_staysl(cover)
        print(f"{t_mil:<18} {cover.thickness_cm:<15.4f} {x:<12.4f} {ccf:<15.6e}")


def example_energy_dependent_mode():
    """
    Demonstrate energy-dependent cover correction (best-physics mode).
    """
    print("\n" + "=" * 70)
    print("Energy-Dependent Cover Correction (Best-Physics Mode)")
    print("=" * 70)
    
    # Define group structure (simplified 5-group)
    group_boundaries = np.array([1e-5, 0.5, 1e3, 1e5, 1e6, 2e7])  # eV
    n_groups = len(group_boundaries) - 1
    
    print(f"\nGroup Structure: {n_groups} groups")
    for g in range(n_groups):
        E_lo, E_hi = group_boundaries[g], group_boundaries[g + 1]
        print(f"  Group {g+1}: {E_lo:.2e} - {E_hi:.2e} eV")
    
    # Create Cd σ_t(E) using 1/v approximation
    sigma_t = create_cd_sigma_total_1v(sigma_0_barns=2520.0)
    
    # 40-mil Cd cover
    cover = CoverSpec(thickness_mil=40.0, angular_model=FluxAngularModel.ISOTROPIC)
    
    # Compute energy-dependent corrections
    result = compute_energy_dependent_cover_corrections(
        cover,
        group_boundaries,
        sigma_t,
        prior_flux=lambda E: 1.0 / E,  # 1/E epithermal spectrum
        sigma_t_source="1/v approximation (Cd, σ₀=2520 b)",
    )
    
    print(f"\nGroup Transmission Factors:")
    print(f"{'Group':<8} {'E_lo (eV)':<12} {'E_hi (eV)':<12} {'T_g':<15} {'σ(T_g)':<12}")
    print("-" * 60)
    
    for g in range(result.n_groups):
        E_lo = group_boundaries[g]
        E_hi = group_boundaries[g + 1]
        T_g = result.group_transmissions[g]
        T_unc = result.group_uncertainties[g]
        print(f"{g+1:<8} {E_lo:<12.2e} {E_hi:<12.2e} {T_g:<15.6e} {T_unc:<12.2e}")
    
    return result


def example_staysl_parity_report():
    """
    Generate STAYSL-format parity report for comparison.
    """
    print("\n" + "=" * 70)
    print("STAYSL Parity Report Generation")
    print("=" * 70)
    
    # Define group structure (SAND-II style, simplified subset)
    group_boundaries = np.array([1e-5, 0.1, 0.5, 1.0, 10, 100, 1e3, 1e5, 1e7, 2e7])
    
    cover = CoverSpec(thickness_mil=40.0)
    
    # Generate report using scalar CCF mode (STAYSL parity)
    report_scalar = create_staysl_parity_report(
        reaction_id="Au-197(n,g)",
        cover=cover,
        group_boundaries_ev=group_boundaries,
        use_energy_dependent=False,
    )
    
    print(f"\nReaction: {report_scalar.reaction_id}")
    print(f"Mode: Scalar CCF (STAYSL parity)")
    print(f"\n{'Group':<8} {'E_lo (eV)':<12} {'E_hi (eV)':<12} {'CCF':<15}")
    print("-" * 50)
    
    for g in range(len(report_scalar.ccf_values)):
        E_lo = group_boundaries[g]
        E_hi = group_boundaries[g + 1]
        ccf = report_scalar.ccf_values[g]
        print(f"{g+1:<8} {E_lo:<12.2e} {E_hi:<12.2e} {ccf:<15.6f}")
    
    # Note: Can also save to CSV for comparison with STAYSL sta_spe.dat
    # report_scalar.to_csv("staysl_parity_Au197.csv")
    
    return report_scalar


def example_plot_transmission():
    """
    Plot energy-dependent transmission through Cd cover.
    """
    print("\n" + "=" * 70)
    print("Plotting Cd Transmission vs Energy")
    print("=" * 70)
    
    # Energy grid
    energies = np.logspace(-5, 7, 500)  # 10 μeV to 10 MeV
    
    # Create 1/v σ_t(E)
    sigma_t = create_cd_sigma_total_1v(sigma_0_barns=2520.0)
    
    # Compute transmission for different thicknesses
    thicknesses_mil = [10, 20, 40]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for t_mil in thicknesses_mil:
        cover = CoverSpec(thickness_mil=t_mil)
        number_density = cover.density * 6.022e23 / cover.atomic_mass
        
        transmissions = []
        for E in energies:
            sigma = sigma_t(E) * 1e-24  # barn to cm²
            Sigma = number_density * sigma
            tau = Sigma * cover.thickness_cm
            T = exponential_integral_E2(tau)
            transmissions.append(T)
        
        ax1.semilogx(energies, transmissions, label=f'{t_mil} mil')
    
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Transmission T(E)')
    ax1.set_title('Cd Cover Transmission (Isotropic Flux)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.55, color='red', linestyle='--', alpha=0.5, label='Cd cutoff')
    ax1.set_xlim(1e-5, 1e7)
    ax1.set_ylim(0, 1.05)
    
    # Plot E₂(x) function
    x_values = np.linspace(0.01, 5, 100)
    e2_values = [exponential_integral_E2(x) for x in x_values]
    exp_values = [np.exp(-x) for x in x_values]
    
    ax2.plot(x_values, e2_values, 'b-', label=r'$E_2(x)$ (isotropic)')
    ax2.plot(x_values, exp_values, 'r--', label=r'$\exp(-x)$ (beam)')
    ax2.set_xlabel('Optical thickness x')
    ax2.set_ylabel('CCF')
    ax2.set_title(r'Cover Correction Factor: $E_2(x)$ vs $\exp(-x)$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('cd_cover_transmission.png', dpi=150)
    print("\nPlot saved to: cd_cover_transmission.png")
    plt.close()


def main():
    """Run all examples."""
    # STAYSL parity mode
    result1 = example_staysl_parity_mode()
    
    # Compare thicknesses
    example_compare_thicknesses()
    
    # Energy-dependent mode
    result2 = example_energy_dependent_mode()
    
    # Generate parity report
    report = example_staysl_parity_report()
    
    # Plot transmission
    try:
        example_plot_transmission()
    except Exception as e:
        print(f"\nPlotting skipped (matplotlib not available or display error): {e}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Use STAYSL parity mode (scalar CCF) for comparison with STAYSL PNNL")
    print("  • Use energy-dependent mode for best-physics analysis")
    print("  • For TRIGA whale tube: use isotropic angular model")
    print("  • CCF is applied to response functions, NOT to activities")
    print("=" * 70)


if __name__ == "__main__":
    main()
