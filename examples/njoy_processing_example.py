#!/usr/bin/env python3
"""
NJOY Processing Pipeline Example
=================================

This example demonstrates the NJOY nuclear data processing pipeline
for generating multi-group cross sections from ENDF evaluations.

NJOY is the standard code for processing evaluated nuclear data libraries.
This module provides:
- Input template generation for NJOY
- Execution wrapper with error handling
- Output parsing for group cross sections
- Reproducibility metadata tracking

Note: NJOY executable must be installed and available in PATH
to actually run the processing. This example demonstrates the
input generation and workflow specification.

References:
- NJOY2016 Manual, LA-UR-17-20093
- ENDF-6 Formats Manual, BNL-203218-2018-INRE
"""

from pathlib import Path
import numpy as np

from fluxforge.data.njoy import (
    NJOYModule,
    GroupStructure,
    GROUP_STRUCTURE_DATA,
    NJOYInput,
    NJOYResult,
    NJOYPipelineSpec,
    generate_reconr_input,
    generate_broadr_input,
    generate_groupr_input,
    generate_njoy_input,
    check_njoy_available,
    create_dosimetry_pipeline,
)


def example_group_structures():
    """
    Show available group structures.
    """
    print("=" * 70)
    print("Available Group Structures for NJOY Processing")
    print("=" * 70)
    
    print(f"\n{'Structure':<20} {'Groups':<10} {'Description':<40}")
    print("-" * 70)
    
    for gs, data in GROUP_STRUCTURE_DATA.items():
        print(f"{gs.name:<20} {data['n_groups']:<10} {data['description']:<40}")


def example_njoy_input_specification():
    """
    Create NJOY input specification for a dosimetry reaction.
    """
    print("\n" + "=" * 70)
    print("NJOY Input Specification")
    print("=" * 70)
    
    # Create input specification for Au-197(n,γ) dosimetry reaction
    njoy_input = NJOYInput(
        endf_file=Path("/path/to/n-079_Au_197.endf"),
        mat_number=7925,  # MAT number for Au-197
        temperatures=[300.0, 600.0, 900.0],  # Multiple temperatures
        group_structure=GroupStructure.SAND_II,
        modules=[
            NJOYModule.RECONR,  # Reconstruct pointwise
            NJOYModule.BROADR,  # Doppler broaden
            NJOYModule.GROUPR,  # Multi-group collapse
        ],
        tolerance=0.001,
        mt_list=[102],  # (n,γ) reaction only
        description="Au-197 dosimetry cross section for TRIGA analysis",
    )
    
    print(f"\nNJOY Input Configuration:")
    print(f"  ENDF file: {njoy_input.endf_file}")
    print(f"  MAT number: {njoy_input.mat_number}")
    print(f"  Temperatures: {njoy_input.temperatures} K")
    print(f"  Group structure: {njoy_input.group_structure.name}")
    print(f"  Modules: {[m.value for m in njoy_input.modules]}")
    print(f"  MT list: {njoy_input.mt_list}")
    print(f"  Tolerance: {njoy_input.tolerance}")
    
    return njoy_input


def example_generate_reconr_input():
    """
    Generate RECONR module input deck.
    """
    print("\n" + "=" * 70)
    print("RECONR Input Deck Generation")
    print("=" * 70)
    
    reconr_input = generate_reconr_input(
        mat=7925,
        tape_in=21,
        tape_out=22,
    )
    
    print("\nGenerated RECONR input:")
    print("-" * 50)
    print(reconr_input)
    
    return reconr_input


def example_generate_broadr_input():
    """
    Generate BROADR module input deck for Doppler broadening.
    """
    print("\n" + "=" * 70)
    print("BROADR Input Deck Generation")
    print("=" * 70)
    
    broadr_input = generate_broadr_input(
        mat=7925,
        temperatures=[300.0, 600.0],
        tape_in_endf=20,
        tape_in_pendf=21,
        tape_out=22,
    )
    
    print("\nGenerated BROADR input:")
    print("-" * 50)
    print(broadr_input)
    
    return broadr_input


def example_generate_groupr_input():
    """
    Generate GROUPR module input deck for multi-group collapse.
    """
    print("\n" + "=" * 70)
    print("GROUPR Input Deck Generation")
    print("=" * 70)
    
    groupr_input = generate_groupr_input(
        mat=7925,
        group_structure=GroupStructure.SAND_II.value,
        temperatures=[300.0],
        tape_in_endf=20,
        tape_in_pendf=22,
        tape_out=23,
    )
    
    print("\nGenerated GROUPR input:")
    print("-" * 50)
    print(groupr_input)
    
    return groupr_input


def example_full_njoy_input():
    """
    Generate complete NJOY input deck.
    """
    print("\n" + "=" * 70)
    print("Complete NJOY Input Deck")
    print("=" * 70)
    
    njoy_input = NJOYInput(
        endf_file=Path("/path/to/n-079_Au_197.endf"),
        mat_number=7925,
        temperatures=[300.0],
        group_structure=GroupStructure.SAND_II,
        modules=[NJOYModule.RECONR, NJOYModule.BROADR, NJOYModule.GROUPR],
        mt_list=[102],
    )
    
    full_input = generate_njoy_input(njoy_input)
    
    print("\nGenerated complete NJOY input:")
    print("-" * 50)
    print(full_input)
    
    return full_input


def example_dosimetry_pipeline():
    """
    Create complete dosimetry cross section processing pipeline.
    """
    print("\n" + "=" * 70)
    print("Dosimetry Cross Section Processing Pipeline")
    print("=" * 70)
    
    # Create pipeline specification using the factory function
    pipeline = create_dosimetry_pipeline()
    
    # Customize for a specific set of reactions
    pipeline.materials = [
        {"mat": 7925, "za": 79197, "name": "Au-197"},
        {"mat": 2625, "za": 26054, "name": "Fe-54"},
        {"mat": 2825, "za": 28058, "name": "Ni-58"},
    ]
    pipeline.temperatures = [293.6, 600.0]
    
    print(f"\nPipeline Specification:")
    print(f"  Name: {pipeline.name}")
    print(f"  Library: {pipeline.endf_library}")
    print(f"  Temperatures: {pipeline.temperatures} K")
    print(f"  Group structure: {pipeline.group_structure.name}")
    print(f"  Modules: {[m.value for m in pipeline.modules]}")
    print(f"  Materials: {len(pipeline.materials)}")
    
    print(f"\n  Processing steps:")
    for i, module in enumerate(pipeline.modules, 1):
        print(f"    {i}. {module.value.upper()}")
    
    print(f"\n  Materials to process:")
    for mat_info in pipeline.materials:
        print(f"    - {mat_info['name']} (MAT={mat_info['mat']})")
    
    return pipeline


def example_check_njoy():
    """
    Check if NJOY is available on the system.
    """
    print("\n" + "=" * 70)
    print("NJOY Availability Check")
    print("=" * 70)
    
    is_available, njoy_path = check_njoy_available()
    
    if is_available:
        print(f"\n✓ NJOY is available at: {njoy_path}")
    else:
        print("\n✗ NJOY is not found in PATH")
        print("  To use NJOY processing, install NJOY and add to PATH")
        print("  NJOY2016 available from: https://github.com/njoy/NJOY2016")


def main():
    """Run all examples."""
    # Show available group structures
    example_group_structures()
    
    # Create input specification
    njoy_input = example_njoy_input_specification()
    
    # Generate individual module inputs
    example_generate_reconr_input()
    example_generate_broadr_input()
    example_generate_groupr_input()
    
    # Generate complete input
    example_full_njoy_input()
    
    # Create dosimetry pipeline
    pipeline = example_dosimetry_pipeline()
    
    # Check NJOY availability
    example_check_njoy()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey points:")
    print("  • NJOY processes ENDF evaluations into multi-group cross sections")
    print("  • SAND-II (640g) structure is standard for dosimetry")
    print("  • Temperature broadening is important for accuracy")
    print("  • All processing steps are recorded for reproducibility")
    print("=" * 70)


if __name__ == "__main__":
    main()
