#!/usr/bin/env python3
"""
Unified Nuclear Data Interface Example
======================================

This example demonstrates the unified nuclear data interface for
accessing cross sections from various sources (ENDF/B-VIII.0, IRDFF-II).

Key Features:
- O1.1: Unified NuclearData interface for CE and MG cross sections
- O1.4: Multi-temperature support with temperature tags
- O1.5: ENDF ↔ IRDFF bridge utilities for explicit mapping

This prevents library mismatches between transport and activation codes
by providing a single source of truth for nuclear data.

References:
- ENDF/B-VIII.0: Nuclear Data Sheets 148 (2018) 1-142
- IRDFF-II: IAEA Technical Report Series No. 452 (2020)
"""

import numpy as np

from fluxforge.data.nuclear_data import (
    DataType,
    DataLibrary,
    ReactionIdentifier,
    TemperatureData,
    NuclearData,
    NuclearDataProvider,
    ReactionMapping,
    ENDF_IRDFF_MAPPINGS,
    ENDFIRDFFBridge,
    create_temperature_set,
    interpolate_temperature,
    create_nuclear_data,
    create_multigroup_data,
)


def example_reaction_identifier():
    """
    Demonstrate ReactionIdentifier for library-agnostic identification.
    """
    print("=" * 70)
    print("Reaction Identifier")
    print("=" * 70)
    
    # Create identifiers for common dosimetry reactions
    reactions = [
        ReactionIdentifier(target="Au-197", mt=102, product="Au-198"),
        ReactionIdentifier(target="In-115", mt=4, product="In-115m"),
        ReactionIdentifier(target="Ni-58", mt=103, product="Co-58"),
        ReactionIdentifier(target="Fe-54", mt=103, product="Mn-54"),
        ReactionIdentifier(target="Al-27", mt=107, product="Na-24"),
    ]
    
    print(f"\n{'Target':<10} {'MT':<6} {'Product':<10} {'Reaction String':<25}")
    print("-" * 55)
    
    for rxn in reactions:
        print(f"{rxn.target:<10} {rxn.mt:<6} {rxn.product:<10} {rxn.reaction_string:<25}")
    
    return reactions


def example_temperature_data():
    """
    Demonstrate temperature-dependent cross section data.
    """
    print("\n" + "=" * 70)
    print("Temperature-Dependent Cross Section Data")
    print("=" * 70)
    
    # Create cross section data at 300 K
    energies = np.logspace(-3, 7, 1000)  # 1 meV to 10 MeV
    
    # Example: Au-197(n,γ) 1/v thermal cross section
    E_th = 0.0253  # eV
    sigma_th = 98.65  # barns at thermal
    values = sigma_th * np.sqrt(E_th / energies)  # 1/v behavior
    
    temp_data_300K = TemperatureData(
        temperature_K=300.0,
        energies=energies,
        values=values,
    )
    
    print(f"\nTemperature: {temp_data_300K.temperature_K} K")
    print(f"Energy range: {energies[0]:.2e} - {energies[-1]:.2e} eV")
    print(f"Number of points: {len(energies)}")
    print(f"σ at thermal (0.0253 eV): {temp_data_300K.interpolate(0.0253):.2f} barns")
    print(f"σ at 1 eV: {temp_data_300K.interpolate(1.0):.2f} barns")
    print(f"σ at 1 MeV: {temp_data_300K.interpolate(1e6):.4f} barns")
    
    return temp_data_300K


def example_nuclear_data_container():
    """
    Demonstrate unified NuclearData container.
    """
    print("\n" + "=" * 70)
    print("Unified NuclearData Container")
    print("=" * 70)
    
    # Create complete nuclear data object for Au-197(n,γ)
    energies = np.logspace(-3, 7, 500)
    E_th = 0.0253
    sigma_th = 98.65
    
    # Create data at multiple temperatures
    temps = [293.6, 500.0, 800.0]
    temp_data_dict = {}
    
    for T in temps:
        # Simplified: same shape, slightly broadened resonances
        values = sigma_th * np.sqrt(E_th / energies)
        temp_data_dict[T] = TemperatureData(
            temperature_K=T,
            energies=energies,
            values=values,
        )
    
    nuclear_data = NuclearData(
        reaction_id=ReactionIdentifier(target="Au-197", mt=102, product="Au-198"),
        data_type=DataType.CONTINUOUS_ENERGY,
        library=DataLibrary.ENDF_B_VIII0,
        temperatures=temp_data_dict,
        metadata={
            "evaluation": "ENDF/B-VIII.0",
            "author": "NNDC",
            "date": "2018",
        },
    )
    
    print(f"\nNuclearData Container:")
    print(f"  Reaction: {nuclear_data.reaction_id.reaction_string}")
    print(f"  Data type: {nuclear_data.data_type.value}")
    print(f"  Library: {nuclear_data.library.value}")
    print(f"  Available temperatures: {nuclear_data.available_temperatures} K")
    
    # Get data at specific temperature
    data_at_500K = nuclear_data.get_at_temperature(500.0)
    print(f"\n  Data at 500 K:")
    print(f"    σ(0.0253 eV) = {data_at_500K.interpolate(0.0253):.2f} barns")
    
    # Interpolate to intermediate temperature
    data_at_450K = nuclear_data.get_at_temperature(450.0, interpolate=True)
    print(f"\n  Interpolated to 450 K:")
    print(f"    σ(0.0253 eV) = {data_at_450K.interpolate(0.0253):.2f} barns")
    
    return nuclear_data


def example_multigroup_data():
    """
    Demonstrate multi-group cross section data.
    """
    print("\n" + "=" * 70)
    print("Multi-Group Cross Section Data")
    print("=" * 70)
    
    # Define a simple 5-group structure
    group_boundaries = np.array([1e-5, 0.5, 1e3, 1e5, 1e6, 2e7])
    n_groups = len(group_boundaries) - 1
    
    # Example group-averaged cross sections (barns)
    group_values = np.array([15.0, 1.5, 0.08, 0.003, 0.001])
    
    # Create multi-group data
    mg_data = create_multigroup_data(
        target="Au-197",
        mt=102,
        group_boundaries=group_boundaries,
        group_values=group_values,
        temperature_K=300.0,
        library=DataLibrary.IRDFF_II,
    )
    
    print(f"\nMulti-Group Data:")
    print(f"  Reaction: {mg_data.reaction_id.reaction_string}")
    print(f"  Data type: {mg_data.data_type.value}")
    print(f"  Number of groups: {mg_data.n_groups}")
    print(f"\n  Group structure and cross sections:")
    print(f"  {'Group':<8} {'E_lo (eV)':<12} {'E_hi (eV)':<12} {'σ (barns)':<12}")
    print("  " + "-" * 45)
    
    for g in range(n_groups):
        E_lo, E_hi = group_boundaries[g], group_boundaries[g+1]
        sigma = group_values[g]
        print(f"  {g+1:<8} {E_lo:<12.2e} {E_hi:<12.2e} {sigma:<12.4f}")
    
    return mg_data


def example_endf_irdff_bridge():
    """
    Demonstrate ENDF to IRDFF-II mapping bridge.
    """
    print("\n" + "=" * 70)
    print("ENDF ↔ IRDFF-II Reaction Mapping Bridge")
    print("=" * 70)
    
    bridge = ENDFIRDFFBridge()
    
    print(f"\nStandard Dosimetry Reaction Mappings:")
    print(f"{'ENDF Target':<12} {'MT':<6} {'IRDFF-II Reaction':<25} {'Notes':<30}")
    print("-" * 75)
    
    for mapping in bridge.mappings:
        print(f"{mapping.endf_target:<12} {mapping.endf_mt:<6} {mapping.irdff_reaction:<25} {mapping.notes:<30}")
    
    # Demonstrate mapping functions
    print(f"\n--- Mapping Examples ---")
    
    # ENDF to IRDFF
    irdff_name = bridge.endf_to_irdff("Au-197", 102)
    print(f"\nENDF Au-197 MT=102 → IRDFF: {irdff_name}")
    
    # IRDFF to ENDF
    endf_info = bridge.irdff_to_endf("Ni-58(n,p)Co-58")
    print(f"IRDFF Ni-58(n,p)Co-58 → ENDF: {endf_info}")
    
    return bridge


def example_cross_section_consistency():
    """
    Demonstrate cross section consistency validation.
    """
    print("\n" + "=" * 70)
    print("Cross Section Consistency Validation")
    print("=" * 70)
    
    bridge = ENDFIRDFFBridge()
    
    # Create example cross section data (simulated)
    energies = np.logspace(-2, 6, 100)
    
    # ENDF data
    endf_xs = 10.0 / np.sqrt(energies)
    
    # IRDFF data (slightly different processing)
    irdff_xs = 10.1 / np.sqrt(energies) * (1 + 0.02 * np.random.randn(len(energies)))
    
    # Check consistency
    consistent, max_dev, message = bridge.validate_consistency(
        endf_xs, irdff_xs, energies, tolerance=0.1
    )
    
    print(f"\nConsistency Check:")
    print(f"  Result: {'✓ Consistent' if consistent else '✗ Inconsistent'}")
    print(f"  Maximum deviation: {max_dev*100:.1f}%")
    print(f"  Message: {message}")
    
    # Now test with intentionally inconsistent data
    irdff_xs_bad = endf_xs * 1.5  # 50% different
    consistent2, max_dev2, message2 = bridge.validate_consistency(
        endf_xs, irdff_xs_bad, energies, tolerance=0.1
    )
    
    print(f"\nInconsistent data test:")
    print(f"  Result: {'✓ Consistent' if consistent2 else '✗ Inconsistent'}")
    print(f"  Maximum deviation: {max_dev2*100:.1f}%")
    print(f"  Message: {message2}")


def example_temperature_interpolation():
    """
    Demonstrate temperature interpolation using sqrt(T) model.
    """
    print("\n" + "=" * 70)
    print("Temperature Interpolation (sqrt(T) Model)")
    print("=" * 70)
    
    # Create data at a few temperatures
    energies = np.logspace(-2, 5, 200)
    sigma_base = 100.0 / np.sqrt(energies)  # 1/v behavior
    
    # Create temperature data with slight T-dependence (simulating Doppler)
    temp_data_dict = {}
    for T in [300.0, 600.0, 1200.0]:
        # Simulate minor temperature effect
        sigma_T = sigma_base * (1 + 0.001 * (T - 300))
        temp_data_dict[T] = TemperatureData(
            temperature_K=T,
            energies=energies,
            values=sigma_T,
        )
    
    print(f"\nAvailable temperatures: {sorted(temp_data_dict.keys())} K")
    
    # Interpolate to intermediate temperature
    target_T = 450.0
    interp_data = interpolate_temperature(temp_data_dict, target_T)
    
    print(f"Interpolated to: {target_T} K")
    print(f"  σ(1 eV) at 300 K:  {temp_data_dict[300.0].interpolate(1.0):.4f} barns")
    print(f"  σ(1 eV) at 450 K:  {interp_data.interpolate(1.0):.4f} barns (interpolated)")
    print(f"  σ(1 eV) at 600 K:  {temp_data_dict[600.0].interpolate(1.0):.4f} barns")
    
    print(f"\n  Note: sqrt(T) interpolation is appropriate for Doppler broadening")


def example_convenience_factory():
    """
    Demonstrate convenience factory functions.
    """
    print("\n" + "=" * 70)
    print("Convenience Factory Functions")
    print("=" * 70)
    
    # Quick CE data creation
    energies = np.logspace(-3, 6, 500)
    values = 50.0 * np.sqrt(0.0253 / energies)  # 1/v behavior
    
    ce_data = create_nuclear_data(
        target="Co-59",
        mt=102,
        energies=energies,
        values=values,
        temperature_K=300.0,
        library=DataLibrary.ENDF_B_VIII0,
        product="Co-60",
    )
    
    print(f"\nCreated CE data:")
    print(f"  {ce_data.reaction_id.reaction_string}")
    print(f"  Library: {ce_data.library.value}")
    print(f"  Data points: {len(energies)}")
    
    # Quick MG data creation
    bounds = np.array([1e-5, 0.5, 1e4, 2e7])
    values_mg = np.array([20.0, 1.0, 0.01])
    
    mg_data = create_multigroup_data(
        target="Fe-54",
        mt=103,
        group_boundaries=bounds,
        group_values=values_mg,
    )
    
    print(f"\nCreated MG data:")
    print(f"  {mg_data.reaction_id.reaction_string}")
    print(f"  Groups: {mg_data.n_groups}")


def main():
    """Run all examples."""
    # Reaction identification
    example_reaction_identifier()
    
    # Temperature data
    example_temperature_data()
    
    # NuclearData container
    example_nuclear_data_container()
    
    # Multi-group data
    example_multigroup_data()
    
    # ENDF-IRDFF bridge
    example_endf_irdff_bridge()
    
    # Consistency validation
    example_cross_section_consistency()
    
    # Temperature interpolation
    example_temperature_interpolation()
    
    # Convenience functions
    example_convenience_factory()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey points:")
    print("  • Use ReactionIdentifier for library-agnostic reaction specification")
    print("  • NuclearData supports both CE and MG representations")
    print("  • Multi-temperature support with sqrt(T) interpolation")
    print("  • ENDFIRDFFBridge ensures consistent mapping between libraries")
    print("  • Always validate cross section consistency when mixing sources")
    print("=" * 70)


if __name__ == "__main__":
    main()
