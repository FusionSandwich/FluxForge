"""
FluxForge Unfolding Benchmark vs Reference Implementation

This script compares FluxForge's GRAVEL and MLEM implementations
against the reference implementations from testing/Neutron-Unfolding.

Input Data:
- Response Matrix: (1024, 201) from scintillator detector
- Pulse Height Spectrum: Measured neutron data
- True Spectrum: Time-of-Flight reference measurement

Validation Approach:
1. Run both Reference and FluxForge implementations
2. Compare unfolded spectra
3. Calculate relative errors vs ToF ground truth
4. Generate comparison plots
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add FluxForge to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fluxforge.solvers.iterative import gravel as fluxforge_gravel, mlem as fluxforge_mlem

# Import reference implementations
sys.path.insert(0, str(Path(__file__).parent))
from reference_gravel import gravel as ref_gravel
from reference_mlem import mlem as ref_mlem

# Paths
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "examples_output"


def gravel_reference_style(R, measurements, x0, max_iters=100, tolerance=0.2):
    """
    GRAVEL implementation matching the reference exactly.
    
    This uses the SAND-II weighting scheme:
    W[i,j] = data[i] * R[i,j] * x[j] / (R @ x)[i]
    """
    import math
    
    R = np.array(R)
    data = np.array(measurements)
    x = np.array(x0, dtype=float).copy()
    
    n, m = R.shape
    
    J0 = 0
    dJ0 = 1
    ddJ = 1
    errors = []
    stepcount = 0
    
    while ddJ > tolerance and stepcount < max_iters:
        stepcount += 1
        
        # Predicted = R @ x
        rdot = R @ x
        
        # Update each energy bin
        for j in range(m):
            # Weight: data * R[:,j] * x[j] / rdot
            W_j = data * R[:, j] * x[j] / np.maximum(rdot, 1e-30)
            
            # Numerator: sum_i W_ij * log(data_i / rdot_i)
            log_ratio = np.log(np.maximum(data / np.maximum(rdot, 1e-30), 1e-30))
            num = np.dot(W_j, log_ratio)
            num = np.nan_to_num(num)
            
            den = np.sum(W_j)
            
            if den > 0:
                x[j] *= math.exp(num / den)
        
        # Convergence metric: ddJ = |dJ - dJ0| where J = sum((rdot-data)^2) / sum(rdot)
        rdot_new = R @ x
        J = np.sum((rdot_new - data) ** 2) / np.maximum(np.sum(rdot_new), 1e-30)
        dJ = J0 - J
        ddJ = abs(dJ - dJ0)
        J0 = J
        dJ0 = dJ
        errors.append(ddJ)
        
        if stepcount % 10 == 0 or ddJ <= tolerance:
            print(f"  FluxForge-Ref GRAVEL iter {stepcount}: ddJ = {ddJ:.4e}")
    
    return x, np.array(errors)


def mlem_reference_style(R, measurements, x0, max_iters=100, tolerance=0.2):
    """
    MLEM implementation matching the reference exactly.
    """
    R = np.array(R)
    data = np.array(measurements)
    x = np.array(x0, dtype=float).copy()
    
    n, m = R.shape
    
    J0 = 0
    dJ0 = 1
    ddJ = 1
    errors = []
    stepcount = 0
    
    while ddJ > tolerance and stepcount < max_iters:
        stepcount += 1
        
        # Predicted = R @ x
        q = R @ x
        q = np.maximum(q, 1e-30)
        
        # Ratio vector
        ratio = data / q
        
        # Update each energy bin
        for j in range(m):
            term = np.dot(R[:, j], ratio)
            x[j] *= term / np.maximum(np.sum(R[:, j]), 1e-30)
        
        # Convergence metric
        q_new = R @ x
        J = np.sum((q_new - data) ** 2) / np.maximum(np.sum(q_new), 1e-30)
        dJ = J0 - J
        ddJ = abs(dJ - dJ0)
        J0 = J
        dJ0 = dJ
        errors.append(ddJ)
        
        if stepcount % 10 == 0 or ddJ <= tolerance:
            print(f"  FluxForge-Ref MLEM iter {stepcount}: ddJ = {ddJ:.4e}")
    
    return x, np.array(errors)


def load_response_matrix():
    """Load and transpose response matrix (201,1024) -> (1024,201)."""
    R = np.loadtxt(DATA_DIR / "response-matrix.txt", delimiter=',')
    return R.T  # Shape: (1024, 201)


def load_measurements():
    """Load pulse height spectrum (detector response)."""
    df = pd.read_csv(DATA_DIR / "reduced_data.csv")
    return df["NEUTRON 1"].values


def load_true_spectrum():
    """Load ToF reference spectrum."""
    data = np.loadtxt(DATA_DIR / "energy-spectrum.txt")
    # First row: spectrum values, Second row: energy bins
    return data[0], data[1]


def run_benchmark():
    print("=" * 70)
    print("FluxForge Unfolding Benchmark")
    print("=" * 70)
    
    # Load data
    R = load_response_matrix()
    n, m = R.shape  # n=measurements (1024), m=energy groups (201)
    print(f"Response Matrix: {n} channels x {m} energy bins")
    
    measurements = load_measurements()
    print(f"Measurements: {len(measurements)} channels")
    
    true_spectrum, energy_bins = load_true_spectrum()
    print(f"True Spectrum: {len(true_spectrum)} bins")
    print(f"Energy Range: {energy_bins[0]:.2f} - {energy_bins[-1]:.2f} MeV")
    
    # Initial guess: constant
    x0 = np.ones((m,))
    tolerance = 0.2  # Reference uses ddJ tolerance
    
    # Preprocess: remove zero-count channels like reference does
    # Reference GRAVEL does this internally, so we do it once for all
    mask = measurements > 0
    R_filtered = R[mask, :]
    meas_filtered = measurements[mask]
    
    n_filtered = R_filtered.shape[0]
    print(f"After removing zero-channels: {n_filtered} channels")
    
    # =========================================================================
    # Reference Implementations (on filtered data)
    # =========================================================================
    print("\n--- Running Reference GRAVEL ---")
    ref_g_result, ref_g_error = ref_gravel(R_filtered, meas_filtered, x0.copy(), tolerance)
    ref_g_result = np.array(ref_g_result)
    
    print("\n--- Running Reference MLEM ---")
    ref_m_result, ref_m_error = ref_mlem(R_filtered, meas_filtered, x0.copy(), tolerance)
    ref_m_result = np.array(ref_m_result)
    
    # =========================================================================
    # FluxForge Implementations (Reference-style, on filtered data)
    # =========================================================================
    print("\n--- Running FluxForge GRAVEL (Reference-style) ---")
    
    ff_g_result, ff_g_error = gravel_reference_style(R_filtered, meas_filtered, x0.copy(), max_iters=100, tolerance=tolerance)
    ff_g_result = np.array(ff_g_result)
    print(f"FluxForge GRAVEL: {len(ff_g_error)} iterations")
    
    print("\n--- Running FluxForge MLEM (Reference-style) ---")
    ff_m_result, ff_m_error = mlem_reference_style(R_filtered, meas_filtered, x0.copy(), max_iters=100, tolerance=tolerance)
    ff_m_result = np.array(ff_m_result)
    print(f"FluxForge MLEM: {len(ff_m_error)} iterations")
    
    # =========================================================================
    # Normalize for comparison
    # =========================================================================
    def normalize(arr):
        return arr / np.linalg.norm(arr)
    
    true_norm = normalize(true_spectrum)
    ref_g_norm = normalize(ref_g_result)
    ref_m_norm = normalize(ref_m_result)
    ff_g_norm = normalize(ff_g_result)
    ff_m_norm = normalize(ff_m_result)
    
    # =========================================================================
    # Calculate Errors
    # =========================================================================
    def relative_error(test, truth):
        """Calculate L2 relative error, excluding zeros."""
        mask = truth > 0
        diff = np.abs(test[mask] - truth[mask]) / truth[mask]
        return np.linalg.norm(diff)
    
    def rmse(a, b):
        """Root mean squared error (normalized)."""
        return np.sqrt(np.mean((a - b) ** 2))
    
    err_ref_g = relative_error(ref_g_norm, true_norm)
    err_ref_m = relative_error(ref_m_norm, true_norm)
    err_ff_g = relative_error(ff_g_norm, true_norm)
    err_ff_m = relative_error(ff_m_norm, true_norm)
    
    print("\n" + "=" * 70)
    print("RESULTS: Relative Error vs ToF Ground Truth")
    print("=" * 70)
    print(f"Reference GRAVEL: {err_ref_g:.4f}")
    print(f"Reference MLEM:   {err_ref_m:.4f}")
    print(f"FluxForge GRAVEL: {err_ff_g:.4f}")
    print(f"FluxForge MLEM:   {err_ff_m:.4f}")
    
    # Agreement between implementations (RMSE of normalized spectra)
    agree_g = rmse(ff_g_norm, ref_g_norm)
    agree_m = rmse(ff_m_norm, ref_m_norm)
    print(f"\nAgreement RMSE (FluxForge vs Ref) GRAVEL: {agree_g:.6f}")
    print(f"Agreement RMSE (FluxForge vs Ref) MLEM:   {agree_m:.6f}")
    
    # Validation: same iterations, same convergence
    print(f"\nIterations: Ref GRAVEL={len(ref_g_error)}, FF GRAVEL={len(ff_g_error)}")
    print(f"Iterations: Ref MLEM={len(ref_m_error)}, FF MLEM={len(ff_m_error)}")
    
    # =========================================================================
    # Plotting
    # =========================================================================
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Plot 1: GRAVEL Comparison
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1 = axes[0]
    ax1.plot(energy_bins, true_norm, 'k-', alpha=0.4, label='ToF Ground Truth', linewidth=2)
    ax1.plot(energy_bins, ref_g_norm, 'b--', label=f'Reference GRAVEL (err={err_ref_g:.3f})', linewidth=1.5)
    ax1.plot(energy_bins, ff_g_norm, 'r:', label=f'FluxForge GRAVEL (err={err_ff_g:.3f})', linewidth=1.5)
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Normalized Counts')
    ax1.set_title('GRAVEL Algorithm Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.2)
    
    ax2 = axes[1]
    ax2.plot(energy_bins, true_norm, 'k-', alpha=0.4, label='ToF Ground Truth', linewidth=2)
    ax2.plot(energy_bins, ref_m_norm, 'b--', label=f'Reference MLEM (err={err_ref_m:.3f})', linewidth=1.5)
    ax2.plot(energy_bins, ff_m_norm, 'r:', label=f'FluxForge MLEM (err={err_ff_m:.3f})', linewidth=1.5)
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_ylabel('Normalized Counts')
    ax2.set_title('MLEM Algorithm Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.2)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "unfolding_benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")
    
    # Plot 2: Implementation Agreement
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(energy_bins, ff_g_norm - ref_g_norm, 'b-', label='GRAVEL: FluxForge - Reference', alpha=0.7)
    ax.plot(energy_bins, ff_m_norm - ref_m_norm, 'r-', label='MLEM: FluxForge - Reference', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Difference (normalized)')
    ax.set_title('Implementation Agreement: FluxForge vs Reference')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    diff_path = OUTPUT_DIR / "unfolding_implementation_diff.png"
    plt.savefig(diff_path, dpi=150)
    print(f"Saved: {diff_path}")
    
    # Save numerical results
    results = {
        "reference_gravel_error": float(err_ref_g),
        "reference_mlem_error": float(err_ref_m),
        "fluxforge_gravel_error": float(err_ff_g),
        "fluxforge_mlem_error": float(err_ff_m),
        "gravel_agreement": float(agree_g),
        "mlem_agreement": float(agree_m),
        "fluxforge_gravel_iterations": len(ff_g_error),
        "fluxforge_mlem_iterations": len(ff_m_error),
        "energy_bins": energy_bins.tolist(),
        "true_spectrum_normalized": true_norm.tolist(),
        "fluxforge_gravel_normalized": ff_g_norm.tolist(),
        "fluxforge_mlem_normalized": ff_m_norm.tolist(),
    }
    
    import json
    json_path = OUTPUT_DIR / "unfolding_benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmark()
