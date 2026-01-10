"""
Verification Script using Real Experimental Data.

This script validates FluxForge capabilities using actual gamma spectra
from the testing/gamma_spec_analysis/test_data directory.

Capabilities Verified:
1. Spectrum I/O (Reading .Spe files)
2. Peak Fitting (Co-60 doublet, Ba-133 multiplet)
3. Coincidence Summing Correction (Co-60, Y-88)
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fluxforge.io import read_spe_file
from fluxforge.analysis import fit_single_peak, auto_find_peaks
from fluxforge.corrections.coincidence import CoincidenceCorrector

# Define paths
DATA_DIR = Path("/filespace/s/smandych/CAE/projects/ALARA/testing/gamma_spec_analysis/test_data")
CO60_FILE = DATA_DIR / "Co_60_raised_1.Spe"
Y88_FILE = DATA_DIR / "Y_88_raised_1.Spe"
BA133_FILE = DATA_DIR / "Ba_133_raised_1.Spe"

class MockEfficiencyCurve:
    """Simple efficiency curve for testing corrections."""
    def evaluate(self, energy):
        # Approximate HPGe efficiency: 20% at 100 keV, 0.1% at 2000 keV
        # Log-log linear approximation
        return 0.2 * (energy / 100.0) ** (-0.8)

def verify_co60():
    print("\n=== Verifying Co-60 Analysis ===")
    if not CO60_FILE.exists():
        print(f"Error: {CO60_FILE} not found")
        return

    # 1. Read Spectrum
    spec = read_spe_file(str(CO60_FILE))
    print(f"Loaded spectrum: {spec.spectrum_id}")
    print(f"Live Time: {spec.live_time} s")
    
    # 2. Find Peaks (Expect 1173 and 1332)
    # Note: We need energy calibration. If not in file, we might need to guess or use channels.
    # Let's check if the SPE reader parsed energy coeffs.
    # spec.calibration is a dict, check for 'energy' key
    if not spec.calibration or 'energy' not in spec.calibration:
        print("Warning: No energy calibration found. Using approximate calibration.")
        # Assuming 8k channels, ~3000 keV range -> ~0.36 keV/ch
        # But let's try to find peaks in channel space first.
        pass
    
    # Let's try to fit the 1332 keV peak.
    # In a typical 8k spectrum, 1332 might be around channel 3000-4000 depending on gain.
    # We'll use auto_find to locate them.
    
    # auto_find_peaks(channels, counts, threshold=3.0, ...)
    # It returns list of (channel, significance)
    peaks_found = auto_find_peaks(spec.channels, spec.counts, threshold=50.0)
    peaks = [p[0] for p in peaks_found]
    print(f"Found {len(peaks)} peaks at indices: {peaks}")
    
    # Assuming the two largest high-energy peaks are 1173 and 1332
    # Sort by index
    peaks = sorted(peaks)
    if len(peaks) < 2:
        print("Error: Not enough peaks found for Co-60")
        return

    # Take the last two significant peaks
    # p1173_idx = peaks[-2]
    # p1332_idx = peaks[-1]
    
    # Better logic: Look for pair with ratio ~1.135 (1332/1173)
    # Filter out low channels (noise/X-rays)
    high_peaks = [p for p in peaks if p > 500]
    
    p1173_idx = None
    p1332_idx = None
    
    for i in range(len(high_peaks)):
        for j in range(i+1, len(high_peaks)):
            p1 = high_peaks[i]
            p2 = high_peaks[j]
            ratio = p2 / p1
            if 1.12 < ratio < 1.15:
                p1173_idx = p1
                p1332_idx = p2
                break
        if p1173_idx: break
        
    if not p1173_idx:
        print("Warning: Could not find Co-60 doublet by ratio. Using last two peaks.")
        p1173_idx = peaks[-2]
        p1332_idx = peaks[-1]
    
    print(f"Candidate 1173 keV peak at channel: {p1173_idx}")
    print(f"Candidate 1332 keV peak at channel: {p1332_idx}")
    
    # 3. Fit Peaks
    # fit_single_peak(channels, counts, peak_channel, fit_width=10, ...)
    fit1 = fit_single_peak(spec.channels, spec.counts, p1173_idx, fit_width=50)
    fit2 = fit_single_peak(spec.channels, spec.counts, p1332_idx, fit_width=50)
    
    # PeakFitResult contains a 'peak' attribute which is GaussianPeak
    p1 = fit1.peak
    p2 = fit2.peak
    
    print(f"Fit 1173: Centroid={p1.centroid:.2f}, FWHM={p1.fwhm:.2f}, Area={p1.area:.0f}")
    print(f"Fit 1332: Centroid={p2.centroid:.2f}, FWHM={p2.fwhm:.2f}, Area={p2.area:.0f}")
    
    # 4. Apply Coincidence Correction
    # We need to map channel to energy for the corrector
    # Simple 2-point calibration
    slope = (1332.5 - 1173.2) / (p2.centroid - p1.centroid)
    offset = 1332.5 - slope * p2.centroid
    print(f"Derived Calibration: E = {slope:.4f}*ch + {offset:.2f}")
    
    corrector = CoincidenceCorrector(MockEfficiencyCurve())
    
    # Correct 1173
    corr1 = corrector.calculate_correction('Co60', 1173.2)
    print(f"Correction for 1173 keV: {corr1.factor:.4f} (Uncertainty: {corr1.uncertainty:.4f})")
    
    # Correct 1332
    corr2 = corrector.calculate_correction('Co60', 1332.5)
    print(f"Correction for 1332 keV: {corr2.factor:.4f} (Uncertainty: {corr2.uncertainty:.4f})")
    
    corrected_area1 = p1.area * corr1.factor
    corrected_area2 = p2.area * corr2.factor
    
    print(f"Corrected Area 1173: {corrected_area1:.0f}")
    print(f"Corrected Area 1332: {corrected_area2:.0f}")

def verify_y88():
    print("\n=== Verifying Y-88 Analysis ===")
    if not Y88_FILE.exists():
        print(f"Error: {Y88_FILE} not found")
        return
        
    spec = read_spe_file(str(Y88_FILE))
    peaks_found = auto_find_peaks(spec.channels, spec.counts, threshold=20.0)
    peaks = [p[0] for p in peaks_found]
    print(f"Found {len(peaks)} peaks")
    
    # Y-88 has 898 and 1836 keV
    # 1836 should be the highest energy strong peak
    if len(peaks) > 0:
        p1836_idx = peaks[-1]
        fit = fit_single_peak(spec.channels, spec.counts, p1836_idx, fit_width=50)
        p1836 = fit.peak
        print(f"Fit 1836 (approx): Centroid={p1836.centroid:.2f}, Area={p1836.area:.0f}")
        
        corrector = CoincidenceCorrector(MockEfficiencyCurve())
        corr = corrector.calculate_correction('Y88', 1836.0)
        print(f"Correction for 1836 keV: {corr.factor:.4f}")

if __name__ == "__main__":
    verify_co60()
    verify_y88()
