#!/usr/bin/env python
"""
STAYSL Interoperability Demo
=============================

Demonstrates data exchange capabilities with STAYSL PNNL and other tools:
- Import saturation rates from spreadsheets
- Export STAYSL-compatible bundles
- Read legacy lower-triangular matrix formats
- Cross-tool validation workflows
"""

import numpy as np
from pathlib import Path

# FluxForge imports
from fluxforge.io.interop import (
    SaturationRateData,
    read_saturation_rates_csv,
    write_saturation_rates_csv,
    read_lower_triangular_matrix,
    write_lower_triangular_matrix,
    STAYSLBundle,
    export_staysl_bundle,
    import_staysl_bundle,
)


def demo_saturation_rates_csv():
    """Demonstrate saturation rate CSV I/O."""
    print("\n" + "="*70)
    print("DEMO: Saturation Rate CSV Import/Export")
    print("="*70)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create example saturation rate data
    data = [
        SaturationRateData(
            reaction_id="Au197_ng",
            rate=1.234e-14,
            uncertainty=1.23e-15,
            half_life_s=233280,  # 2.7 days
            target_atoms=1.5e18,
            notes="Gold foil, 0.1 mm",
        ),
        SaturationRateData(
            reaction_id="Fe58_ng",
            rate=5.67e-16,
            uncertainty=5.67e-17,
            half_life_s=3.84e6,  # 44.5 days
            target_atoms=2.1e17,
            notes="Fe-58 wire",
        ),
        SaturationRateData(
            reaction_id="In115_ng",
            rate=8.9e-14,
            uncertainty=8.9e-15,
            half_life_s=3243,  # 54 minutes
            target_atoms=4.5e17,
            notes="Indium foil",
        ),
        SaturationRateData(
            reaction_id="Ni58_np",
            rate=2.34e-16,
            uncertainty=4.68e-17,
            half_life_s=6.07e6,  # 70.9 days
            target_atoms=3.2e18,
            notes="Nickel wire",
        ),
    ]
    
    # Write to CSV
    csv_path = output_dir / "saturation_rates.csv"
    write_saturation_rates_csv(data, csv_path)
    print(f"\nWrote saturation rates to: {csv_path}")
    print("\nCSV contents:")
    print(csv_path.read_text())
    
    # Read back and verify
    loaded = read_saturation_rates_csv(csv_path)
    print(f"\nLoaded {len(loaded)} reactions:")
    for d in loaded:
        print(f"  {d.reaction_id}: rate={d.rate:.3e} ± {d.uncertainty:.2e}")
    
    return data


def demo_lower_triangular_matrix():
    """Demonstrate lower-triangular matrix I/O."""
    print("\n" + "="*70)
    print("DEMO: Lower-Triangular Matrix I/O (Legacy Format)")
    print("="*70)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create example symmetric covariance matrix
    # This could represent dosimetry input correlations
    n = 4
    labels = ["Au197", "Fe58", "In115", "Ni58"]
    
    # Build with some off-diagonal correlations
    matrix = np.array([
        [1.0e-28, 2.5e-29, 1.0e-29, 0.0],
        [2.5e-29, 4.0e-32, 5.0e-33, 0.0],
        [1.0e-29, 5.0e-33, 9.0e-28, 0.0],
        [0.0,     0.0,     0.0,     2.5e-31],
    ])
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, np.diag(matrix) * 2)  # Undo diagonal halving
    
    # Write in legacy format
    mat_path = output_dir / "covariance_legacy.txt"
    write_lower_triangular_matrix(matrix, mat_path, precision=6)
    print(f"\nWrote lower-triangular matrix to: {mat_path}")
    print("\nFile contents:")
    print(mat_path.read_text())
    
    # Read back
    loaded = read_lower_triangular_matrix(mat_path, n=4)
    print("\nLoaded matrix (symmetric reconstruction):")
    print(loaded)
    
    # Verify symmetry
    assert np.allclose(loaded, loaded.T), "Matrix should be symmetric"
    print("\n✓ Matrix symmetry verified")
    
    return matrix


def demo_staysl_bundle():
    """Demonstrate STAYSL bundle export/import."""
    print("\n" + "="*70)
    print("DEMO: STAYSL Bundle Export/Import")
    print("="*70)
    
    output_dir = Path(__file__).parent / "output" / "staysl_bundle"
    
    # Create example bundle
    n_groups = 5
    n_reactions = 3
    
    # Energy grid (simplified)
    energy_bounds_eV = np.array([1e-5, 0.4, 1.0, 1e3, 1e6, 20e6])
    
    # Prior flux (1/E-like)
    prior_flux = np.array([1e10, 5e9, 1e9, 5e8, 1e8])
    prior_covariance = np.diag((0.2 * prior_flux)**2)  # 20% diagonal
    
    # Measurements
    reaction_ids = ["Au197_ng", "Fe58_ng", "Ni58_np"]
    measured_rates = np.array([1.5e-14, 5.0e-16, 2.0e-16])
    measurement_covariance = np.diag((0.05 * measured_rates)**2)  # 5% diagonal
    
    # Response matrix (simplified cross sections)
    response_matrix = np.array([
        [100.0, 50.0, 10.0, 1.0, 0.1],      # Au-197(n,g)
        [2.0, 1.0, 0.5, 0.1, 0.01],          # Fe-58(n,g)
        [0.0, 0.0, 0.1, 10.0, 100.0],        # Ni-58(n,p) threshold
    ])
    
    # Create bundle
    bundle = STAYSLBundle(
        prior_flux=prior_flux,
        prior_covariance=prior_covariance,
        energy_bounds_eV=energy_bounds_eV,
        reaction_ids=reaction_ids,
        measured_rates=measured_rates,
        measurement_covariance=measurement_covariance,
        response_matrix=response_matrix,
        title="Example STAYSL Bundle",
        notes="Generated by FluxForge interop demo",
    )
    
    # Validate
    warnings = bundle.validate()
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("✓ Bundle validation passed")
    
    # Export
    files = export_staysl_bundle(bundle, output_dir, prefix="example")
    print(f"\nExported bundle to: {output_dir}")
    print("Files created:")
    for name, path in files.items():
        print(f"  - {name}: {path.name}")
    
    # Show file contents
    print("\n--- Metadata JSON ---")
    print((output_dir / "example_metadata.json").read_text())
    
    print("\n--- Measurements CSV ---")
    print((output_dir / "example_measurements.csv").read_text())
    
    # Import back
    print("\n--- Reimporting Bundle ---")
    loaded = import_staysl_bundle(output_dir, prefix="example")
    
    print(f"Loaded bundle: {loaded.title}")
    print(f"  Reactions: {loaded.reaction_ids}")
    print(f"  Groups: {len(loaded.prior_flux)}")
    
    # Verify roundtrip
    np.testing.assert_array_almost_equal(loaded.prior_flux, bundle.prior_flux)
    np.testing.assert_array_almost_equal(loaded.measured_rates, bundle.measured_rates, decimal=5)
    print("\n✓ Roundtrip verification passed")
    
    return bundle


def demo_cross_tool_workflow():
    """Demonstrate cross-tool validation workflow."""
    print("\n" + "="*70)
    print("DEMO: Cross-Tool Validation Workflow")
    print("="*70)
    
    print("""
This demo shows a typical workflow for cross-validating FluxForge
results against STAYSL PNNL:

1. Export FluxForge results as STAYSL bundle
   → Creates files compatible with STAYSL input format
   
2. Run STAYSL PNNL with same inputs
   → Use exported prior, measurements, response
   
3. Compare results
   → Load STAYSL outputs
   → Compare adjusted flux, chi-squared, etc.

The key files for STAYSL compatibility:
- *_prior_flux.txt      → Prior spectrum (group format)
- *_prior_cov.txt       → Prior covariance (lower triangular)
- *_measurements.csv    → Measured rates + uncertainties
- *_meas_cov.txt        → Measurement covariance matrix
- *_response.csv        → Response matrix (reactions × groups)
- *_metadata.json       → Bundle metadata for reference

These can be reformatted for specific STAYSL input requirements.
""")
    
    return None


def main():
    """Run all demos."""
    print("FluxForge STAYSL Interoperability Demo")
    print("======================================\n")
    
    # Run demos
    demo_saturation_rates_csv()
    demo_lower_triangular_matrix()
    demo_staysl_bundle()
    demo_cross_tool_workflow()
    
    print("\n" + "="*70)
    print("All demos completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
