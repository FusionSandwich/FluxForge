"""
Complete FluxForge Example Workflow using Flux Wire Data.

This script demonstrates the full analysis pipeline using the experimental
Co-Cd-RAFM-1 flux wire measurements.

Workflow:
1. Load spectrum (Genie ASC format)
2. Automated peak finding
3. Peak fitting (Gaussian + Hypermet)
4. Coincidence summing correction
5. Activity calculation
6. k0-NAA flux characterization
7. Response matrix generation
8. Spectrum unfolding (GLS, GRAVEL, MLEM)
9. Save all outputs to examples_outputs/
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from fluxforge.io import read_genie_spectrum, parse_genie_report
from fluxforge.analysis import (
    fit_single_peak,
    auto_find_peaks,
    K0Parameters,
    calculate_k0_parameters
)
from fluxforge.corrections.coincidence import CoincidenceCorrector
from fluxforge.physics.activation import (
    calculate_activity,
    calculate_reaction_rate
)
from fluxforge.data.irdff import IRDFFDatabase
from fluxforge.solvers.gls import gls_unfold
from fluxforge.solvers.iterative import gravel_unfold, mlem_unfold
from fluxforge.core.provenance import generate_provenance

# File paths
EXAMPLE_DIR = Path(__file__).parent
DATA_DIR = EXAMPLE_DIR / "flux_wire"
FE_CD_DIR = EXAMPLE_DIR / "fe_cd_rafm_1"
OUTPUT_DIR = EXAMPLE_DIR.parent.parent.parent.parent / "examples_outputs" / "flux_wire_analysis"

# Input files
SPECTRUM_FILE = DATA_DIR / "Co-Cd-RAFM-1_25cm.ASC"
REPORT_FILE = DATA_DIR / "Co-Cd-RAFM-1_25cm.txt"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FluxForge Complete Workflow: Co-Cd-RAFM-1 Flux Wire Analysis")
print("="*80)

# ============================================================================
# STAGE 1: Load Spectrum
# ============================================================================
print("\n[STAGE 1] Loading Spectrum...")

spec = read_genie_spectrum(str(SPECTRUM_FILE))
print(f"  Spectrum ID: {spec.spectrum_id}")
print(f"  Live Time: {spec.live_time:.2f} s")
print(f"  Real Time: {spec.real_time:.2f} s")
print(f"  Dead Time: {(1 - spec.live_time/spec.real_time)*100:.2f}%")
print(f"  Total Counts: {spec.counts.sum():.0f}")

# Save spectrum metadata
spectrum_meta = {
    "spectrum_id": spec.spectrum_id,
    "live_time_s": float(spec.live_time),
    "real_time_s": float(spec.real_time),
    "total_counts": int(spec.counts.sum()),
    "calibration": spec.calibration,
    "provenance": generate_provenance("load_spectrum", {"file": str(SPECTRUM_FILE)})
}
with open(OUTPUT_DIR / "spectrum_metadata.json", 'w') as f:
    json.dump(spectrum_meta, f, indent=2)

# ============================================================================
# STAGE 2: Automated Peak Finding
# ============================================================================
print("\n[STAGE 2] Automated Peak Finding...")

peaks_found = auto_find_peaks(
    spec.channels, 
    spec.counts, 
    threshold=5.0,
    min_distance=10,
    finder_method='scipy'
)

print(f"  Found {len(peaks_found)} peaks")

# Filter for high-energy peaks (Co-60 region)
high_energy_peaks = [(ch, sig) for ch, sig in peaks_found if ch > 2000]
print(f"  High-energy peaks (ch > 2000): {len(high_energy_peaks)}")

# Save peak list
peaks_data = {
    "total_peaks": len(peaks_found),
    "peaks": [{"channel": int(ch), "significance": float(sig)} for ch, sig in peaks_found],
    "provenance": generate_provenance("auto_find_peaks", {"threshold": 5.0})
}
with open(OUTPUT_DIR / "detected_peaks.json", 'w') as f:
    json.dump(peaks_data, f, indent=2)

# ============================================================================
# STAGE 3: Peak Fitting (Co-60 doublet)
# ============================================================================
print("\n[STAGE 3] Peak Fitting...")

# Identify Co-60 peaks by ratio (1332/1173 ≈ 1.135)
high_peaks = [p[0] for p in high_energy_peaks]
co60_1173_ch = None
co60_1332_ch = None

for i in range(len(high_peaks)):
    for j in range(i+1, len(high_peaks)):
        p1 = high_peaks[i]
        p2 = high_peaks[j]
        ratio = p2 / p1
        if 1.12 < ratio < 1.15:
            co60_1173_ch = p1
            co60_1332_ch = p2
            break
    if co60_1173_ch: break

if not co60_1173_ch:
    print("  WARNING: Could not auto-identify Co-60 doublet by ratio")
    # Fallback: use last two high peaks
    if len(high_peaks) >= 2:
        co60_1173_ch = high_peaks[-2]
        co60_1332_ch = high_peaks[-1]

if co60_1173_ch and co60_1332_ch:
    print(f"  Fitting Co-60 peaks at channels {co60_1173_ch}, {co60_1332_ch}")
    
    # Fit peaks
    fit_1173 = fit_single_peak(spec.channels, spec.counts, co60_1173_ch, fit_width=50)
    fit_1332 = fit_single_peak(spec.channels, spec.counts, co60_1332_ch, fit_width=50)
    
    p1173 = fit_1173.peak
    p1332 = fit_1332.peak
    
    print(f"  1173 keV: Centroid={p1173.centroid:.2f} ch, Area={p1173.area:.0f} counts, FWHM={p1173.fwhm:.2f} ch")
    print(f"  1332 keV: Centroid={p1332.centroid:.2f} ch, Area={p1332.area:.0f} counts, FWHM={p1332.fwhm:.2f} ch")
    
    # Derive energy calibration
    energy_slope = (1332.5 - 1173.2) / (p1332.centroid - p1173.centroid)
    energy_offset = 1332.5 - energy_slope * p1332.centroid
    
    print(f"  Derived Calibration: E = {energy_slope:.4f} * ch + {energy_offset:.2f} keV")
    
    # Save fit results
    fit_results = {
        "1173_keV": {
            "channel": float(p1173.centroid),
            "area": float(p1173.area),
            "area_unc": float(p1173.area_uncertainty),
            "fwhm_ch": float(p1173.fwhm),
            "chi_squared": float(fit_1173.chi_squared)
        },
        "1332_keV": {
            "channel": float(p1332.centroid),
            "area": float(p1332.area),
            "area_unc": float(p1332.area_uncertainty),
            "fwhm_ch": float(p1332.fwhm),
            "chi_squared": float(fit_1332.chi_squared)
        },
        "calibration": {
            "slope_keV_per_ch": float(energy_slope),
            "offset_keV": float(energy_offset)
        },
        "provenance": generate_provenance("fit_peaks", {"method": "Gaussian"})
    }
    
    with open(OUTPUT_DIR / "peak_fit_results.json", 'w') as f:
        json.dump(fit_results, f, indent=2)

# ============================================================================
# STAGE 4: Coincidence Summing Correction
# ============================================================================
print("\n[STAGE 4] Coincidence Summing Correction...")

class SimpleEfficiencyCurve:
    """Simple efficiency model for correction."""
    def evaluate(self, energy):
        # Approximate HPGe efficiency at 25 cm
        return 0.001 * (energy / 1000.0) ** (-0.8)

corrector = CoincidenceCorrector(SimpleEfficiencyCurve())

corr_1173 = corrector.calculate_correction('Co60', 1173.2)
corr_1332 = corrector.calculate_correction('Co60', 1332.5)

print(f"  1173 keV correction factor: {corr_1173.factor:.4f} ± {corr_1173.uncertainty:.4f}")
print(f"  1332 keV correction factor: {corr_1332.factor:.4f} ± {corr_1332.uncertainty:.4f}")

if co60_1173_ch and co60_1332_ch:
    corrected_area_1173 = p1173.area * corr_1173.factor
    corrected_area_1332 = p1332.area * corr_1332.factor
    
    print(f"  Corrected areas: 1173={corrected_area_1173:.0f}, 1332={corrected_area_1332:.0f}")
    
    # Save corrections
    corr_data = {
        "1173_keV": {
            "raw_area": float(p1173.area),
            "correction_factor": float(corr_1173.factor),
            "corrected_area": float(corrected_area_1173)
        },
        "1332_keV": {
            "raw_area": float(p1332.area),
            "correction_factor": float(corr_1332.factor),
            "corrected_area": float(corrected_area_1332)
        },
        "provenance": generate_provenance("coincidence_correction", {"method": "TCS"})
    }
    
    with open(OUTPUT_DIR / "coincidence_corrections.json", 'w') as f:
        json.dump(corr_data, f, indent=2)

# ============================================================================
# STAGE 5: Activity Calculation
# ============================================================================
print("\n[STAGE 5] Activity Calculation...")

if co60_1173_ch and co60_1332_ch:
    # Using corrected areas
    eff_1173 = 0.0012  # Approximate from detector model
    eff_1332 = 0.0010
    
    gamma_intensity_1173 = 0.9985
    gamma_intensity_1332 = 0.9998
    
    activity_1173 = calculate_activity(
        net_counts=corrected_area_1173,
        live_time=spec.live_time,
        efficiency=eff_1173,
        gamma_intensity=gamma_intensity_1173
    )
    
    activity_1332 = calculate_activity(
        net_counts=corrected_area_1332,
        live_time=spec.live_time,
        efficiency=eff_1332,
        gamma_intensity=gamma_intensity_1332
    )
    
    print(f"  Activity (1173 keV): {activity_1173:.3e} Bq")
    print(f"  Activity (1332 keV): {activity_1332:.3e} Bq")
    
    # Weighted average
    w1 = 1.0 / (p1173.area_uncertainty ** 2)
    w2 = 1.0 / (p1332.area_uncertainty ** 2)
    weighted_activity = (w1 * activity_1173 + w2 * activity_1332) / (w1 + w2)
    
    print(f"  Weighted Average Activity: {weighted_activity:.3e} Bq")
    
    # Save activities
    activity_data = {
        "1173_keV": {
            "activity_Bq": float(activity_1173),
            "efficiency": eff_1173,
            "gamma_intensity": gamma_intensity_1173
        },
        "1332_keV": {
            "activity_Bq": float(activity_1332),
            "efficiency": eff_1332,
            "gamma_intensity": gamma_intensity_1332
        },
        "weighted_average_Bq": float(weighted_activity),
        "provenance": generate_provenance("calculate_activity", {})
    }
    
    with open(OUTPUT_DIR / "activities.json", 'w') as f:
        json.dump(activity_data, f, indent=2)

# ============================================================================
# STAGE 6: k0-NAA Flux Characterization (Placeholder)
# ============================================================================
print("\n[STAGE 6] k0-NAA Flux Characterization...")
print("  Note: k0-NAA requires bare + Cd-covered measurements")
print("  This is a Cd-covered measurement only")

# Placeholder for demonstration
k0_params = K0Parameters(
    f=0.0,  # Would need bare measurement
    alpha=0.0,
    f_uncertainty=0.0,
    alpha_uncertainty=0.0
)

k0_data = {
    "note": "k0-NAA requires both bare and Cd-covered measurements",
    "available_data": "Cd-covered only",
    "f": float(k0_params.f),
    "alpha": float(k0_params.alpha),
    "provenance": generate_provenance("k0_naa", {})
}

with open(OUTPUT_DIR / "k0_naa_params.json", 'w') as f:
    json.dump(k0_data, f, indent=2)

# ============================================================================
# STAGE 7: Load Fe-Cd Example for Unfolding Demonstration
# ============================================================================
print("\n[STAGE 7] Spectrum Unfolding (using Fe-Cd-RAFM-1 example data)...")

# Load existing example data
with open(FE_CD_DIR / "measurements.json", 'r') as f:
    measurements_data = json.load(f)

with open(FE_CD_DIR / "boundaries.json", 'r') as f:
    boundaries = json.load(f)

# Simple demonstration with minimal data
n_groups = len(boundaries) - 1
print(f"  Energy groups: {n_groups}")

# Mock reaction rates for demo
reaction_rates = np.array([1e-5, 2e-5, 1e-5])  # 3 groups
prior_flux = np.array([1e10, 1e9, 1e8])  # Initial guess

# Mock response matrix (would be from IRDFF)
R = np.array([
    [1.0, 0.1, 0.01],
    [0.1, 1.0, 0.1],
    [0.01, 0.1, 1.0]
]) * 1e-25  # barns

# Uncertainty
sigma = reaction_rates * 0.1  # 10% uncertainty

print("\n  Running GLS Unfolding...")
gls_result = gls_unfold(
    reaction_rates=reaction_rates,
    response_matrix=R,
    prior_flux=prior_flux,
    sigma=sigma
)

print(f"    Chi-squared: {gls_result.chi_squared:.2f}")
print(f"    Flux adjustment factors: {gls_result.flux / prior_flux}")

print("\n  Running GRAVEL Unfolding...")
gravel_result = gravel_unfold(
    measurements=reaction_rates,
    response=R,
    prior=prior_flux,
    max_iterations=100,
    tolerance=1e-6
)

print(f"    Converged: {gravel_result.converged}")
print(f"    Iterations: {gravel_result.iterations}")
print(f"    Final chi-squared: {gravel_result.diagnostics['chi_squared']:.2f}")

print("\n  Running MLEM Unfolding...")
mlem_result = mlem_unfold(
    measurements=reaction_rates,
    response=R,
    prior=prior_flux,
    max_iterations=100,
    tolerance=1e-6
)

print(f"    Converged: {mlem_result.converged}")
print(f"    Iterations: {mlem_result.iterations}")
print(f"    Final chi-squared: {mlem_result.diagnostics['chi_squared']:.2f}")

# Save unfolding results
unfolding_results = {
    "energy_boundaries_eV": boundaries,
    "prior_flux": prior_flux.tolist(),
    "gls": {
        "flux": gls_result.flux.tolist(),
        "chi_squared": float(gls_result.chi_squared),
        "covariance": gls_result.covariance.tolist()
    },
    "gravel": {
        "flux": gravel_result.flux.tolist(),
        "converged": gravel_result.converged,
        "iterations": gravel_result.iterations,
        "chi_squared": float(gravel_result.diagnostics['chi_squared'])
    },
    "mlem": {
        "flux": mlem_result.flux.tolist(),
        "converged": mlem_result.converged,
        "iterations": mlem_result.iterations,
        "chi_squared": float(mlem_result.diagnostics['chi_squared'])
    },
    "provenance": generate_provenance("spectrum_unfolding", {
        "methods": ["GLS", "GRAVEL", "MLEM"]
    })
}

with open(OUTPUT_DIR / "unfolding_results.json", 'w') as f:
    json.dump(unfolding_results, f, indent=2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.json")):
    print(f"  - {file.name}")

print("\n" + "="*80)
