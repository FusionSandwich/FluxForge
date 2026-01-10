"""
FluxForge Capability Demonstration using Real Data.

This script processes experimental data and demonstrates key capabilities:
1. Gamma Spectroscopy Analysis (SPE & ASC formats)
2. Physics Corrections (Coincidence Summing)
3. MCNP/ALARA Integration (Input Parsing/Generation)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fluxforge.io import (
    read_spe_file, 
    read_genie_spectrum,
    parse_mcnp_input,
    ALARAInputGenerator,
    ALARASettings
)
from fluxforge.analysis import fit_single_peak, auto_find_peaks
from fluxforge.corrections.coincidence import CoincidenceCorrector
from fluxforge.solvers.gls import gls_adjust

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    print("Warning: pandas/matplotlib not found. Plotting disabled.")
    HAS_PLOT = False

# Directories
BASE_DIR = Path(os.path.dirname(__file__)).parent
DATA_DIR_SPE = Path("/filespace/s/smandych/CAE/projects/ALARA/testing/gamma_spec_analysis/test_data")
DATA_DIR_ASC = Path("/filespace/s/smandych/CAE/projects/ALARA/FluxForge/src/fluxforge/examples/flux_wire")
DATA_DIR_MCNP = Path("/filespace/s/smandych/CAE/projects/ALARA/test_v3_vit_J")
DATA_DIR_WORKFLOW = Path("/filespace/s/smandych/CAE/projects/ALARA/MCNP_ALARA_Workflow")
OUTPUT_DIR = Path("/filespace/s/smandych/CAE/projects/ALARA/FluxForge/examples_output")

class MockEfficiencyCurve:
    """Simple efficiency curve for testing corrections."""
    def evaluate(self, energy):
        return 0.2 * (energy / 100.0) ** (-0.8)

def save_json(data, filename):
    with open(OUTPUT_DIR / filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {filename}")

def run_spe_analysis():
    print("\n--- Part 1: SPE Analysis (Co-60) ---")
    filepath = DATA_DIR_SPE / "Co_60_raised_1.Spe"
    if not filepath.exists():
        print("Skipping SPE analysis: File not found")
        return

    # Load
    spec = read_spe_file(str(filepath))
    print(f"Loaded: {spec.spectrum_id} (Live Time: {spec.live_time}s)")

    # Peak Finding
    # Using previous tuned params
    peaks_found = auto_find_peaks(spec.channels, spec.counts, threshold=50.0)
    high_peaks = [p for p in [x[0] for x in peaks_found] if p > 500]
    
    # Identify Doublet
    p1173_idx = None
    p1332_idx = None
    
    for i in range(len(high_peaks)):
        for j in range(i+1, len(high_peaks)):
            ratio = high_peaks[j] / high_peaks[i]
            if 1.12 < ratio < 1.15:
                p1173_idx = high_peaks[i]
                p1332_idx = high_peaks[j]
                break
        if p1173_idx: break
    
    results = {
        "file": str(filepath),
        "calibration": spec.calibration,
        "peaks": []
    }

    if p1173_idx and p1332_idx:
        corrector = CoincidenceCorrector(MockEfficiencyCurve())
        
        for name, idx, energy_target in [("1173 keV", p1173_idx, 1173.2), ("1332 keV", p1332_idx, 1332.5)]:
            fit = fit_single_peak(spec.channels, spec.counts, idx, fit_width=50)
            peak = fit.peak
            
            # Correction
            corr = corrector.calculate_correction('Co60', energy_target)
            corrected_area = peak.area * corr.factor
            
            p_data = {
                "name": name,
                "centroid": peak.centroid,
                "fwhm": peak.fwhm,
                "area_raw": peak.area,
                "area_corrected": corrected_area,
                "correction_factor": corr.factor
            }
            results["peaks"].append(p_data)
            print(f"Processed {name}: Area {peak.area:.0f} -> {corrected_area:.0f}")
            
    save_json(results, "spe_analysis_results.json")

def run_asc_analysis():
    print("\n--- Part 2: Genie ASC Analysis (Co-Cd Wire) ---")
    filepath = DATA_DIR_ASC / "Co-Cd-RAFM-1_25cm.ASC"
    if not filepath.exists():
        print("Skipping ASC analysis: File not found")
        return

    # Load
    # read_genie_spectrum returns a GammaSpectrum too
    spec = read_genie_spectrum(str(filepath))
    print(f"Loaded: {spec.spectrum_id} (Live Time: {spec.live_time}s)")
    
    # Check Energy Calibration
    cal = spec.calibration.get('energy', [])
    print(f"Calibration Coefficients: {cal}")
    
    # Find Peaks
    # Co-60 lines: 1173, 1332 keV.
    # We can use energy if calibration exists
    peaks_found = []
    
    if spec.energies is not None:
        # Search in energy space? auto_find_peaks works on counts/indices
        # Let's find in channel space then map
        peak_indices = auto_find_peaks(spec.channels, spec.counts, threshold=10.0)
        peak_indices = [p[0] for p in peak_indices]
        
        for idx in peak_indices:
            energy = spec.energies[idx]
            if 1100 < energy < 1400:
                peaks_found.append({"channel": idx, "energy": energy})
    
    print(f"Found {len(peaks_found)} candidate lines in Co-60 region")
    
    results = {
        "file": str(filepath),
        "calibration": cal,
        "co60_candidates": peaks_found
    }
    save_json(results, "asc_analysis_results.json")

def run_mcnp_alara_demo():
    print("\n--- Part 3: MCNP/ALARA Integration ---")
    
    # Parsing MCNP
    mcnp_file = DATA_DIR_MCNP / "whale_J_core_clean_loc.i"
    if mcnp_file.exists():
        print(f"Parsing MCNP Input: {mcnp_file.name}")
        data = parse_mcnp_input(str(mcnp_file))
        materials = data.get('materials', {})
        print(f"Extracted {len(materials)} materials")
        
        m_summary = {k: v['lines'][0][:50]+"..." for k, v in materials.items()}
        save_json({"materials": m_summary}, "mcnp_materials.json")
    else:
        print("MCNP file not found")

    # Generating ALARA
    print("Generating ALARA Input...")
    settings = ALARASettings(
        material_name="Eurofer",
        density=7.8,
        flux_file="whale.flx",
        material_lib="fendl3bin.lib",
        element_lib="fendl3bin.lib",
        data_library="fendl3bin.lib",
        cooling_times=["0s", "1h", "1d", "1y"]
    )
    
    gen = ALARAInputGenerator(settings)
    out_path = OUTPUT_DIR / "generated_alara.inp"
    gen.write(str(out_path))
    print(f"Generated ALARA Input: {out_path}")

def run_unfolding_demo():
    print("\n--- Part 4: Flux Unfolding ---")
    
    if not HAS_PLOT:
        print("Skipping unfolding demo: pandas/matplotlib dependency missing")
        return

    # 1. Load Prior Spectrum (MCNP result)
    csv_path = DATA_DIR_WORKFLOW / "spectrum_vit_j_TEST.csv"
    if not csv_path.exists():
        print(f"Spectrum file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Column check
    if 'flux_bin [n·s⁻¹]' not in df.columns:
        print("Error: Column 'flux_bin [n·s⁻¹]' not found in CSV")
        return

    prior_flux = df['flux_bin [n·s⁻¹]'].values
    energy_bins = df['E_low[eV]'].values
    
    n_groups = len(prior_flux)
    print(f"Loaded Prior Spectrum: {n_groups} groups")

    # 2. Get Measured Activity (Mock for Demo)
    # In production, this comes from Activity = PeakArea / (Eff * Time * Branching)
    # We will simulate a measurement that is 10% higher than the prior prediction
    # to force the unfold to adjust.
    
    # Calculate Calculated Activity C = R * Phi
    # Build Mock Cross Section (1/v absorption)
    # Scaling factor to avoid numerical singularity (values < 1e-12)
    # in the pure-python linalg solver which has a hardcoded threshold.
    # We work in "micro-barns" or similar effective scaling.
    SCALE_FACTOR = 1e24 

    sigma_vector = []
    for e in energy_bins:
        # 37 barns at 0.0253 eV, 1/v behavior
        # Original: barns -> cm2 (1e-24). 
        # Scaled: cm2 * 1e24 -> barns.
        val = 37.0 * np.sqrt(0.0253 / (e + 1e-9)) * 1e-24 * SCALE_FACTOR
        sigma_vector.append(val)
        
    calc_activity = sum(s * phi for s, phi in zip(sigma_vector, prior_flux))
    print(f"Prior 'Calculated' Reaction Rate (Scaled): {calc_activity:.2e}")
    
    # Mock Measurement: 1.2x the calculated value
    measured_activity = calc_activity * 1.2
    print(f"Mock 'Measured' Reaction Rate (Scaled):   {measured_activity:.2e}")

    # 3. Covariances
    # Prior Uncertainty: 20%
    prior_cov = np.diag((prior_flux * 0.20) ** 2).tolist()
    # Measurement Uncertainty: 5%
    meas_cov = [[(measured_activity * 0.05) ** 2]]
    
    # Response Matrix [1 x G]
    response_matrix = [sigma_vector]
    
    # 4. Run Unfolding
    print("Running GLS Adjustment...")
    sol = gls_adjust(
        response=response_matrix,
        measurements=[measured_activity],
        measurement_cov=meas_cov,
        prior_flux=prior_flux.tolist(),
        prior_cov=prior_cov,
        enforce_nonnegativity=True
    )
    
    print(f"Unfolding Complete. Chi2: {sol.chi2:.4f}")
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.step(energy_bins, prior_flux, label='MCNP Prior', where='pre', color='blue')
    plt.step(energy_bins, sol.flux, label='Unfolded (GLS)', where='pre', color='red', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Flux (n/s)')
    plt.title(f'Flux Unfolding Demo (Chi2={sol.chi2:.2f})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plot_path = OUTPUT_DIR / "unfolded_spectrum.png"
    plt.savefig(plot_path)
    print(f"Saved Plot: {plot_path}")
    
    # Save JSON Results
    res_data = {
        "chi2": sol.chi2,
        "energy_bins": energy_bins.tolist(),
        "prior_flux": prior_flux.tolist(),
        "unfolded_flux": sol.flux
    }
    save_json(res_data, "unfolded_results.json")

def run_iterative_unfolding_demo():
    """Demonstrate GRAVEL and MLEM using reference benchmark data."""
    print("\n--- Part 5: GRAVEL/MLEM Iterative Unfolding ---")
    
    if not HAS_PLOT:
        print("Skipping: matplotlib not available")
        return
    
    # Import benchmark from same package
    benchmark_dir = Path(__file__).parent / "unfolding_benchmark"
    if not benchmark_dir.exists():
        print(f"Benchmark data not found: {benchmark_dir}")
        return
    
    sys.path.insert(0, str(benchmark_dir))
    try:
        from run_benchmark import run_benchmark as run_unfolding_benchmark
        print("Running GRAVEL/MLEM benchmark against reference implementation...")
        run_unfolding_benchmark()
    except ImportError as e:
        print(f"Could not import benchmark: {e}")
    except Exception as e:
        print(f"Benchmark error: {e}")

if __name__ == "__main__":
    try:
        if not OUTPUT_DIR.exists():
            OUTPUT_DIR.mkdir(parents=True)
            
        run_spe_analysis()
        run_asc_analysis()
        run_mcnp_alara_demo()
        run_unfolding_demo()
        run_iterative_unfolding_demo()
        print("\nFull Demonstration Complete!")
    except Exception as e:
        print(f"Demo Failed: {e}")
        import traceback
        traceback.print_exc()
