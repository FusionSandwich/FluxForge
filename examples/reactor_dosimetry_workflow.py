#!/usr/bin/env python3
"""
Reactor Dosimetry Workflow Example

This example demonstrates the complete 7-step reactor dosimetry workflow
as described in INL/EXT-21-64191 "Impact of Flux Wire Selection on Neutron
Spectrum Adjustment" (Holschuh et al., 2021).

The workflow follows Figure 1 from the INL report:

    1. Selection of Flux Wires
       └─→ 2. Neutron Irradiation of Flux Wires
            └─→ 3. Measurement of Flux Wires
                 └─→ 4. Computational Model of Energy-dependent Neutron Flux
                      └─→ 5. Spectrum Unfolding Algorithm
                           └─→ 6. Experimentally Adjusted Energy-Dependent Neutron Flux
                                └─→ 7. Determination of Experiment Fluences of Interest

This example uses simulated HFIR-like data to demonstrate the complete
workflow including:
- Flux wire selection using IRDFF-II reactions
- Reaction rate calculation from measured activities
- MCNP spectrum as initial guess
- GRAVEL/SAND-II spectrum unfolding
- 1-MeV equivalent fluence calculation (ASTM E722)
- DPA calculation (ASTM E693)

References:
    [1] T. Holschuh et al., INL/EXT-21-64191 (2021)
    [2] ASTM E721, E722, E944 (Reactor Dosimetry Standards)
    [3] A. Trkov et al., Nuclear Data Sheets 163, 1-108 (2020) - IRDFF-II

Author: FluxForge Development Team
Date: 2026-01-08
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add FluxForge to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# FluxForge Imports
# =============================================================================

from fluxforge.analysis.flux_wire_selection import (
    WireCategory,
    FLUX_WIRE_DATABASE,
    INL_ROBUST_COMBOS,
    get_wire_reactions,
    analyze_wire_combination,
    suggest_wire_combinations,
    recommend_wire_additions,
    print_wire_summary,
    calculate_1mev_equivalent_fluence,
    calculate_dpa,
)

from fluxforge.physics.activation import (
    IrradiationSegment,
    irradiation_buildup_factor,
    reaction_rate_from_activity,
)

from fluxforge.data.irdff import (
    IRDFFDatabase,
    IRDFF_REACTIONS,
    build_response_matrix,
    get_flux_wire_energy_groups,
)

from fluxforge.workflows.spectrum_unfolding import (
    SpectrumUnfolder,
    UnfoldingResult,
    FluxWireMeasurement,
    quick_unfold,
)


# =============================================================================
# Simulated HFIR Data (based on INL report)
# =============================================================================

# HFIR PT-1 (Pneumatic Tube) simulated spectrum characteristics
# Based on light-water moderated reactor with thermal and fast components

HFIR_IRRADIATION_PARAMS = {
    "position": "PT-1",
    "reactor_power_MW": 85.0,
    "irradiation_time_s": 3600.0,  # 1 hour
    "cooling_time_s": 7200.0,  # 2 hours before measurement
    "thermal_flux_n_cm2_s": 2.5e14,
    "epithermal_flux_n_cm2_s": 1.2e13,
    "fast_flux_n_cm2_s": 8.0e13,
}


def generate_hfir_spectrum(n_groups: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simulated HFIR-like neutron spectrum.
    
    This is a simplified model combining:
    - Maxwell-Boltzmann thermal peak at ~0.025 eV
    - 1/E epithermal slowing-down region
    - Watt fission spectrum for fast neutrons
    
    Returns
    -------
    energy_edges : np.ndarray
        Energy group boundaries in MeV
    flux : np.ndarray
        Flux per energy group (n/cm²/s)
    """
    # Energy groups from 1e-11 to 20 MeV
    energy_edges_MeV = np.logspace(-11, np.log10(20), n_groups + 1)
    energy_centers = np.sqrt(energy_edges_MeV[:-1] * energy_edges_MeV[1:])
    energy_widths = energy_edges_MeV[1:] - energy_edges_MeV[:-1]
    
    # Thermal Maxwell-Boltzmann (T = 300 K → kT = 0.0253 eV)
    kT = 0.0253e-6  # MeV
    thermal = energy_centers * np.exp(-energy_centers / kT)
    thermal = thermal / np.max(thermal) * HFIR_IRRADIATION_PARAMS["thermal_flux_n_cm2_s"]
    
    # Epithermal 1/E (0.5 eV to 100 keV)
    epithermal_mask = (energy_centers > 0.5e-6) & (energy_centers < 0.1)
    epithermal = np.zeros_like(energy_centers)
    epithermal[epithermal_mask] = 1.0 / energy_centers[epithermal_mask]
    epithermal = epithermal / np.max(epithermal + 1e-30) * HFIR_IRRADIATION_PARAMS["epithermal_flux_n_cm2_s"]
    
    # Fast Watt fission spectrum: χ(E) = C * exp(-E/a) * sinh(sqrt(b*E))
    # a ≈ 0.988 MeV, b ≈ 2.249 MeV^-1 for U-235 thermal fission
    a, b = 0.988, 2.249
    fast_mask = energy_centers > 0.1
    fast = np.zeros_like(energy_centers)
    E_fast = energy_centers[fast_mask]
    fast[fast_mask] = np.exp(-E_fast / a) * np.sinh(np.sqrt(b * E_fast))
    fast = fast / np.max(fast + 1e-30) * HFIR_IRRADIATION_PARAMS["fast_flux_n_cm2_s"]
    
    # Combine
    flux = thermal + epithermal + fast
    
    # Normalize to per-MeV (lethargy) for proper integration
    flux_per_MeV = flux / energy_widths
    
    return energy_edges_MeV, flux_per_MeV


def simulate_reaction_rate(
    reaction: str,
    spectrum_edges_MeV: np.ndarray,
    spectrum_flux: np.ndarray,
    irrad_time_s: float = 3600.0,
    irdff_db: Optional[IRDFFDatabase] = None,
) -> Tuple[float, float]:
    """
    Calculate simulated reaction rate by folding spectrum with cross section.
    
    R = ∫ σ(E) × φ(E) dE
    
    Returns
    -------
    rate : float
        Reaction rate (reactions per target atom per second)
    uncertainty : float
        Estimated uncertainty (assuming 5% cross section + 3% flux)
    """
    if irdff_db is None:
        irdff_db = IRDFFDatabase(verbose=False)
    
    try:
        xs = irdff_db.get_cross_section(reaction)
        if xs is None:
            return 0.0, 0.0
    except Exception as e:
        print(f"Warning: Could not get cross section for {reaction}: {e}")
        return 0.0, 0.0
    
    # Calculate group centers and widths
    energy_centers_MeV = np.sqrt(spectrum_edges_MeV[:-1] * spectrum_edges_MeV[1:])
    energy_widths_MeV = spectrum_edges_MeV[1:] - spectrum_edges_MeV[:-1]
    
    # Evaluate cross section at group centers (convert MeV to eV)
    sigma_barn = xs.evaluate(energy_centers_MeV * 1e6)
    
    # Convert barn to cm²
    sigma_cm2 = sigma_barn * 1e-24
    
    # Integrate: R = Σ σ(E) × φ(E) × ΔE
    rate = np.sum(sigma_cm2 * spectrum_flux * energy_widths_MeV)
    
    # Uncertainty estimate: 5% xs + 3% flux in quadrature
    uncertainty = rate * np.sqrt(0.05**2 + 0.03**2)
    
    return rate, uncertainty


def rate_to_activity(
    reaction_rate: float,
    n_atoms: float,
    half_life_s: float,
    irrad_time_s: float,
    cool_time_s: float,
) -> float:
    """
    Convert reaction rate to measured activity.
    
    A(t) = R × N × (1 - exp(-λt_irr)) × exp(-λt_cool)
    """
    decay_const = np.log(2) / half_life_s
    
    # Saturation factor
    saturation = 1 - np.exp(-decay_const * irrad_time_s)
    
    # Decay factor
    decay = np.exp(-decay_const * cool_time_s)
    
    activity = reaction_rate * n_atoms * saturation * decay
    
    return activity


# =============================================================================
# Step 1: Selection of Flux Wires
# =============================================================================

def step1_select_flux_wires() -> List[str]:
    """
    Step 1: Select optimal flux wire combination.
    
    Following INL recommendations:
    - Use {Ti, Fe, Co} as baseline (INL "standard" set)
    - Add 4th wire for improved energy coverage
    - Consider IRDFF-II reactions and threshold energies
    """
    print("\n" + "="*70)
    print("STEP 1: Selection of Flux Wires")
    print("="*70)
    
    # Analyze INL baseline combination
    inl_baseline = ["Ti", "Fe", "Co"]
    print(f"\nINL Baseline Wire Set: {inl_baseline}")
    
    baseline_score = analyze_wire_combination(inl_baseline, verbose=False)
    print(f"  Energy coverage score: {baseline_score.energy_coverage:.2f}")
    print(f"  Threshold spacing score: {baseline_score.threshold_spacing:.2f}")
    print(f"  Has thermal reaction: {baseline_score.has_thermal}")
    print(f"  Has fast reactions: {baseline_score.has_fast}")
    print(f"  Overall score: {baseline_score.overall_score:.2f}")
    
    # Get wire addition recommendations
    print("\n  Recommendations for additional wires:")
    additions = recommend_wire_additions(inl_baseline, spectrum_type="reactor")
    for wire, reason, improvement in additions[:3]:
        print(f"    + {wire}: {reason} (+{improvement:.2f})")
    
    # Use INL recommended 4-wire combo
    selected_wires = ["Ti", "Fe", "Co", "Au"]
    print(f"\nSelected Wire Set: {selected_wires}")
    
    final_score = analyze_wire_combination(selected_wires, verbose=False)
    print(f"  Overall score: {final_score.overall_score:.2f}")
    print(f"  Reactions available: {len(final_score.reactions)}")
    for rxn in final_score.reactions[:6]:
        print(f"    - {rxn}")
    
    # Print recommendations
    for rec in final_score.recommendations:
        print(f"  {rec}")
    
    return selected_wires


# =============================================================================
# Step 2 & 3: Irradiation and Measurement
# =============================================================================

@dataclass
class FluxWireData:
    """Container for measured flux wire data."""
    reaction: str
    target: str
    product: str
    activity_Bq: float
    uncertainty_Bq: float
    sample_mass_g: float
    half_life_s: float
    irrad_time_s: float
    cool_time_s: float
    

def step2_3_irradiate_and_measure(
    wire_elements: List[str],
    spectrum_edges_MeV: np.ndarray,
    spectrum_flux: np.ndarray,
) -> List[FluxWireData]:
    """
    Steps 2 & 3: Simulate irradiation and measurement.
    
    In practice:
    - Step 2: Irradiate flux wires in reactor
    - Step 3: Measure gamma spectra with HPGe detector
    
    Here we simulate the process using the generated spectrum.
    """
    print("\n" + "="*70)
    print("STEPS 2-3: Neutron Irradiation and Measurement")
    print("="*70)
    
    irrad_time = HFIR_IRRADIATION_PARAMS["irradiation_time_s"]
    cool_time = HFIR_IRRADIATION_PARAMS["cooling_time_s"]
    
    print(f"\nIrradiation Parameters:")
    print(f"  Position: {HFIR_IRRADIATION_PARAMS['position']}")
    print(f"  Duration: {irrad_time/3600:.2f} hours")
    print(f"  Cooling time: {cool_time/3600:.2f} hours")
    print(f"  Thermal flux: {HFIR_IRRADIATION_PARAMS['thermal_flux_n_cm2_s']:.2e} n/cm²/s")
    print(f"  Fast flux: {HFIR_IRRADIATION_PARAMS['fast_flux_n_cm2_s']:.2e} n/cm²/s")
    
    # Initialize IRDFF database
    irdff_db = IRDFFDatabase(verbose=False)
    
    # Get reactions for selected wires
    measurements: List[FluxWireData] = []
    
    print(f"\nSimulated Measurements:")
    print(f"{'Reaction':<25} {'Rate (s⁻¹)':<12} {'Activity (Bq)':<15} {'Unc (%)':<8}")
    print("-" * 65)
    
    for element in wire_elements:
        reactions = get_wire_reactions(element)
        
        for rxn in reactions:
            # Skip if no gamma (e.g., S-32(n,p)P-32)
            if rxn.gamma_intensity < 0.01:
                continue
                
            reaction_str = rxn.reaction_str
            
            # Calculate reaction rate from spectrum
            rate, rate_unc = simulate_reaction_rate(
                reaction_str, spectrum_edges_MeV, spectrum_flux, 
                irrad_time, irdff_db
            )
            
            if rate <= 0:
                continue
            
            # Simulate sample: 0.01 g wire, natural abundance
            sample_mass_g = 0.01
            avogadro = 6.022e23
            # Approximate atomic mass from target
            atomic_mass = float(''.join(c for c in rxn.target if c.isdigit()))
            n_atoms = sample_mass_g * avogadro / atomic_mass
            
            # Calculate activity
            activity = rate_to_activity(
                rate, n_atoms, rxn.half_life_s, irrad_time, cool_time
            )
            
            # Add measurement uncertainty (2-5% from gamma spectrometry)
            meas_unc = activity * np.random.uniform(0.02, 0.05)
            
            # Add some noise to simulate real measurement
            activity_meas = activity * (1 + np.random.normal(0, 0.02))
            
            if activity_meas > 0:
                measurements.append(FluxWireData(
                    reaction=reaction_str,
                    target=rxn.target,
                    product=rxn.product,
                    activity_Bq=activity_meas,
                    uncertainty_Bq=meas_unc,
                    sample_mass_g=sample_mass_g,
                    half_life_s=rxn.half_life_s,
                    irrad_time_s=irrad_time,
                    cool_time_s=cool_time,
                ))
                
                rel_unc = 100 * meas_unc / activity_meas
                print(f"{reaction_str:<25} {rate:<12.3e} {activity_meas:<15.3e} {rel_unc:<8.1f}")
    
    print(f"\nTotal reactions measured: {len(measurements)}")
    
    return measurements


# =============================================================================
# Step 4: A Priori Computational Spectrum
# =============================================================================

def step4_load_apriori_spectrum(
    true_spectrum_edges: np.ndarray,
    true_spectrum_flux: np.ndarray,
    deviation: str = "small",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 4: Load computational a priori spectrum.
    
    In practice this would come from MCNP. Here we simulate different
    scenarios from the INL study:
    - "small": 5% deviation from true
    - "large": 50% deviation from true
    - "thermal_enhanced": Thermal component too high
    - "fast_enhanced": Fast component too high
    """
    print("\n" + "="*70)
    print("STEP 4: Computational Model of Energy-dependent Neutron Flux")
    print("="*70)
    
    n_groups = len(true_spectrum_flux)
    energy_centers = np.sqrt(true_spectrum_edges[:-1] * true_spectrum_edges[1:])
    
    if deviation == "nominal":
        # No deviation - perfect a priori
        apriori_flux = true_spectrum_flux.copy()
        print(f"\n  A priori spectrum: Nominal (matches true spectrum)")
        
    elif deviation == "small":
        # Small deviation (~5% at 1 MeV)
        noise = 1.0 + 0.05 * np.random.randn(n_groups)
        apriori_flux = true_spectrum_flux * noise
        print(f"\n  A priori spectrum: Small deviation (~5% at 1 MeV)")
        
    elif deviation == "large":
        # Large deviation (~50% at 1 MeV)
        # Slope change in log space
        slope_factor = np.power(energy_centers / 1e-6, -0.1)  # Slight tilt
        noise = 1.0 + 0.3 * np.random.randn(n_groups)
        apriori_flux = true_spectrum_flux * slope_factor * noise
        print(f"\n  A priori spectrum: Large deviation (~50% at 1 MeV)")
        
    elif deviation == "thermal_enhanced":
        # Increased thermal, decreased fast
        thermal_mask = energy_centers < 1e-6
        fast_mask = energy_centers > 0.1
        apriori_flux = true_spectrum_flux.copy()
        apriori_flux[thermal_mask] *= 2.0
        apriori_flux[fast_mask] *= 0.5
        print(f"\n  A priori spectrum: Increased thermal (+100%), decreased fast (-50%)")
        
    elif deviation == "fast_enhanced":
        # Increased fast, decreased thermal
        thermal_mask = energy_centers < 1e-6
        fast_mask = energy_centers > 0.1
        apriori_flux = true_spectrum_flux.copy()
        apriori_flux[thermal_mask] *= 0.5
        apriori_flux[fast_mask] *= 2.0
        print(f"\n  A priori spectrum: Increased fast (+100%), decreased thermal (-50%)")
    
    else:
        apriori_flux = true_spectrum_flux.copy()
    
    # Ensure positive
    apriori_flux = np.maximum(apriori_flux, 1e-30)
    
    print(f"  Energy groups: {n_groups}")
    print(f"  Energy range: {true_spectrum_edges[0]:.2e} - {true_spectrum_edges[-1]:.2e} MeV")
    
    return true_spectrum_edges, apriori_flux


# =============================================================================
# Step 5: Spectrum Unfolding
# =============================================================================

def step5_unfold_spectrum(
    measurements: List[FluxWireData],
    apriori_edges_MeV: np.ndarray,
    apriori_flux: np.ndarray,
    method: str = "GRAVEL",
) -> UnfoldingResult:
    """
    Step 5: Perform spectrum unfolding.
    
    Uses FluxForge's implementation of:
    - GRAVEL (log-space SAND-II variant)
    - MLEM (Maximum Likelihood Expectation Maximization)
    
    These implement similar approaches to STAYSL PNNL referenced in INL report.
    """
    print("\n" + "="*70)
    print(f"STEP 5: Spectrum Unfolding Algorithm ({method})")
    print("="*70)
    
    # Convert to eV for FluxForge
    energy_edges_eV = apriori_edges_MeV * 1e6
    
    # Create unfolder
    unfolder = SpectrumUnfolder(
        custom_energy_edges=energy_edges_eV,
        verbose=False,
    )
    
    print(f"\nReactions used for unfolding:")
    
    # Add measurements
    for m in measurements:
        # Calculate saturation and decay factors
        decay_const = np.log(2) / m.half_life_s
        sat_factor = 1 - np.exp(-decay_const * m.irrad_time_s)
        decay_factor = np.exp(-decay_const * m.cool_time_s)
        
        unfolder.add_reaction(
            reaction=m.reaction,
            activity_Bq=m.activity_Bq,
            uncertainty_Bq=m.uncertainty_Bq,
            saturation_factor=sat_factor,
            decay_factor=decay_factor,
        )
        print(f"  {m.reaction}: {m.activity_Bq:.3e} ± {m.uncertainty_Bq:.3e} Bq")
    
    # Set a priori spectrum
    # FluxForge expects flux per unit energy, convert if needed
    energy_widths_eV = energy_edges_eV[1:] - energy_edges_eV[:-1]
    flux_per_eV = apriori_flux / energy_widths_eV * 1e6  # Convert from per MeV to per eV
    unfolder.set_initial_guess(np.abs(flux_per_eV), source="MCNP_simulated")
    
    # Perform unfolding
    print(f"\nRunning {method} unfolding...")
    result = unfolder.unfold(
        method=method,
        max_iterations=500,
        tolerance=1e-5,
        chi2_tolerance=0.05,
        relaxation=0.7,
    )
    
    print(f"\nUnfolding Results:")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {result.converged}")
    print(f"  Chi²/dof: {result.chi_squared:.4f}")
    print(f"  Integral flux: {result.integral_flux:.3e} n/cm²/s")
    
    return result


# =============================================================================
# Step 6: Experimentally Adjusted Spectrum
# =============================================================================

def step6_compare_spectra(
    true_edges_MeV: np.ndarray,
    true_flux: np.ndarray,
    apriori_flux: np.ndarray,
    unfolded_result: UnfoldingResult,
) -> Dict:
    """
    Step 6: Compare experimentally adjusted spectrum to true and a priori.
    """
    print("\n" + "="*70)
    print("STEP 6: Experimentally Adjusted Energy-Dependent Neutron Flux")
    print("="*70)
    
    # Get unfolded flux (convert back to per MeV for comparison)
    energy_widths_eV = unfolded_result.energy_edges[1:] - unfolded_result.energy_edges[:-1]
    unfolded_flux_per_MeV = unfolded_result.flux * energy_widths_eV / 1e6
    
    # Normalize for comparison
    true_norm = true_flux / np.sum(true_flux)
    apriori_norm = apriori_flux / np.sum(apriori_flux)
    unfolded_norm = unfolded_flux_per_MeV / np.sum(unfolded_flux_per_MeV)
    
    # Calculate comparison metrics
    energy_centers_MeV = np.sqrt(true_edges_MeV[:-1] * true_edges_MeV[1:])
    
    # RMS deviation
    apriori_rms = np.sqrt(np.mean((apriori_norm - true_norm)**2))
    unfolded_rms = np.sqrt(np.mean((unfolded_norm - true_norm)**2))
    
    # Ratio at key energies
    thermal_idx = np.argmin(np.abs(energy_centers_MeV - 0.025e-6))
    fast_idx = np.argmin(np.abs(energy_centers_MeV - 1.0))
    
    print(f"\nSpectrum Comparison (normalized):")
    print(f"  A priori vs True RMS deviation: {apriori_rms:.4e}")
    print(f"  Unfolded vs True RMS deviation: {unfolded_rms:.4e}")
    print(f"  Improvement factor: {apriori_rms/unfolded_rms:.2f}x")
    
    # Regional comparison
    thermal_mask = energy_centers_MeV < 1e-6
    fast_mask = energy_centers_MeV > 0.1
    
    thermal_true = np.sum(true_flux[thermal_mask])
    thermal_unfolded = np.sum(unfolded_flux_per_MeV[thermal_mask])
    fast_true = np.sum(true_flux[fast_mask])
    fast_unfolded = np.sum(unfolded_flux_per_MeV[fast_mask])
    
    print(f"\nRegional Flux Comparison:")
    print(f"  Thermal (< 1 eV):")
    print(f"    True: {thermal_true:.3e}, Unfolded: {thermal_unfolded:.3e}")
    print(f"    Ratio: {thermal_unfolded/thermal_true:.3f}")
    print(f"  Fast (> 0.1 MeV):")
    print(f"    True: {fast_true:.3e}, Unfolded: {fast_unfolded:.3e}")
    print(f"    Ratio: {fast_unfolded/fast_true:.3f}")
    
    return {
        "true_flux": true_flux,
        "apriori_flux": apriori_flux,
        "unfolded_flux": unfolded_flux_per_MeV,
        "energy_centers_MeV": energy_centers_MeV,
        "apriori_rms": apriori_rms,
        "unfolded_rms": unfolded_rms,
        "thermal_ratio": thermal_unfolded / thermal_true,
        "fast_ratio": fast_unfolded / fast_true,
    }


# =============================================================================
# Step 7: Determine Fluences of Interest
# =============================================================================

def step7_calculate_fluences(
    unfolded_result: UnfoldingResult,
    irrad_time_s: float,
) -> Dict:
    """
    Step 7: Calculate experiment fluences of interest.
    
    Includes:
    - Total fluence
    - Thermal fluence (E < 0.55 eV Cd cutoff)
    - Epithermal fluence (0.55 eV - 100 keV)
    - Fast fluence (> 100 keV)
    - 1-MeV equivalent fluence (ASTM E722)
    - DPA for iron (ASTM E693)
    """
    print("\n" + "="*70)
    print("STEP 7: Determination of Experiment Fluences of Interest")
    print("="*70)
    
    # Convert to MeV and fluence (flux × time)
    energy_edges_MeV = unfolded_result.energy_edges / 1e6
    energy_centers_MeV = unfolded_result.energy_midpoints / 1e6
    energy_widths_MeV = unfolded_result.energy_widths / 1e6
    
    # Flux per MeV
    flux_per_MeV = unfolded_result.flux * unfolded_result.energy_widths / 1e6
    fluence_per_MeV = flux_per_MeV * irrad_time_s
    
    # Calculate regional fluences
    cd_cutoff_MeV = 0.55e-6  # 0.55 eV
    thermal_mask = energy_centers_MeV < cd_cutoff_MeV
    epithermal_mask = (energy_centers_MeV >= cd_cutoff_MeV) & (energy_centers_MeV < 0.1)
    fast_mask = energy_centers_MeV >= 0.1
    fast_1mev_mask = energy_centers_MeV >= 1.0
    
    total_fluence = np.sum(fluence_per_MeV * energy_widths_MeV)
    thermal_fluence = np.sum(fluence_per_MeV[thermal_mask] * energy_widths_MeV[thermal_mask])
    epithermal_fluence = np.sum(fluence_per_MeV[epithermal_mask] * energy_widths_MeV[epithermal_mask])
    fast_fluence = np.sum(fluence_per_MeV[fast_mask] * energy_widths_MeV[fast_mask])
    fast_1mev_fluence = np.sum(fluence_per_MeV[fast_1mev_mask] * energy_widths_MeV[fast_1mev_mask])
    
    print(f"\nFluence Results (irradiation time: {irrad_time_s/3600:.2f} hours):")
    print(f"  Total fluence:      {total_fluence:.3e} n/cm²")
    print(f"  Thermal (< 0.55 eV): {thermal_fluence:.3e} n/cm²")
    print(f"  Epithermal:         {epithermal_fluence:.3e} n/cm²")
    print(f"  Fast (> 0.1 MeV):   {fast_fluence:.3e} n/cm²")
    print(f"  Fast (> 1 MeV):     {fast_1mev_fluence:.3e} n/cm²")
    
    # Calculate 1-MeV equivalent fluence (ASTM E722)
    print(f"\n1-MeV Silicon Equivalent Fluence (ASTM E722):")
    
    fluence_1mev_kerma, hardness_kerma = calculate_1mev_equivalent_fluence(
        energy_edges_MeV, fluence_per_MeV, damage_function="kerma_si"
    )
    fluence_1mev_niel, hardness_niel = calculate_1mev_equivalent_fluence(
        energy_edges_MeV, fluence_per_MeV, damage_function="niel_si"
    )
    
    print(f"  KERMA-based:   {fluence_1mev_kerma:.3e} n/cm² (1-MeV eq)")
    print(f"  NIEL-based:    {fluence_1mev_niel:.3e} n/cm² (1-MeV eq)")
    print(f"  Spectral hardness (φ>1MeV/φtotal): {hardness_kerma:.4f}")
    
    # Calculate DPA for iron (ASTM E693)
    print(f"\nDisplacements Per Atom - DPA (ASTM E693):")
    
    dpa_fe = calculate_dpa(energy_edges_MeV, fluence_per_MeV, material="Fe")
    
    print(f"  Iron (Fe): {dpa_fe:.3e} DPA")
    print(f"  Note: Approximate - use NJOY/SPECTER for accurate DPA")
    
    return {
        "total_fluence": total_fluence,
        "thermal_fluence": thermal_fluence,
        "epithermal_fluence": epithermal_fluence,
        "fast_fluence": fast_fluence,
        "fast_1mev_fluence": fast_1mev_fluence,
        "fluence_1mev_kerma": fluence_1mev_kerma,
        "fluence_1mev_niel": fluence_1mev_niel,
        "spectral_hardness": hardness_kerma,
        "dpa_fe": dpa_fe,
    }


# =============================================================================
# Main Workflow
# =============================================================================

def run_reactor_dosimetry_workflow(
    apriori_deviation: str = "small",
    n_groups: int = 100,
    unfolding_method: str = "GRAVEL",
):
    """
    Execute complete 7-step reactor dosimetry workflow.
    
    Parameters
    ----------
    apriori_deviation : str
        A priori spectrum deviation: 'nominal', 'small', 'large', 
        'thermal_enhanced', 'fast_enhanced'
    n_groups : int
        Number of energy groups (100 or 640 as in INL study)
    unfolding_method : str
        Unfolding algorithm: 'GRAVEL' or 'MLEM'
    """
    print("\n" + "#"*70)
    print("#  REACTOR DOSIMETRY WORKFLOW - Following INL/EXT-21-64191")
    print("#"*70)
    print(f"\nConfiguration:")
    print(f"  Energy groups: {n_groups}")
    print(f"  A priori deviation: {apriori_deviation}")
    print(f"  Unfolding method: {unfolding_method}")
    
    # Generate true spectrum
    print("\nGenerating simulated HFIR-like spectrum...")
    true_edges, true_flux = generate_hfir_spectrum(n_groups)
    
    # Step 1: Select flux wires
    selected_wires = step1_select_flux_wires()
    
    # Steps 2-3: Irradiate and measure
    measurements = step2_3_irradiate_and_measure(
        selected_wires, true_edges, true_flux
    )
    
    # Step 4: Load a priori spectrum (with deviation)
    apriori_edges, apriori_flux = step4_load_apriori_spectrum(
        true_edges, true_flux, deviation=apriori_deviation
    )
    
    # Step 5: Spectrum unfolding
    unfolded_result = step5_unfold_spectrum(
        measurements, apriori_edges, apriori_flux, method=unfolding_method
    )
    
    # Step 6: Compare spectra
    comparison = step6_compare_spectra(
        true_edges, true_flux, apriori_flux, unfolded_result
    )
    
    # Step 7: Calculate fluences
    fluences = step7_calculate_fluences(
        unfolded_result,
        irrad_time_s=HFIR_IRRADIATION_PARAMS["irradiation_time_s"],
    )
    
    # Summary
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print(f"\nFlux Wire Combination: {selected_wires}")
    print(f"Reactions Used: {len(measurements)}")
    print(f"Unfolding Method: {unfolding_method}")
    print(f"Convergence: {'Yes' if unfolded_result.converged else 'No'}")
    print(f"Chi²/dof: {unfolded_result.chi_squared:.4f}")
    print(f"\nSpectrum Quality:")
    print(f"  RMS improvement over a priori: {comparison['apriori_rms']/comparison['unfolded_rms']:.2f}x")
    print(f"\nKey Results:")
    print(f"  Total fluence: {fluences['total_fluence']:.3e} n/cm²")
    print(f"  1-MeV equivalent: {fluences['fluence_1mev_kerma']:.3e} n/cm² (ASTM E722)")
    print(f"  Iron DPA: {fluences['dpa_fe']:.3e} (ASTM E693)")
    
    return {
        "wires": selected_wires,
        "measurements": measurements,
        "unfolded_result": unfolded_result,
        "comparison": comparison,
        "fluences": fluences,
    }


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reactor Dosimetry Workflow (INL/EXT-21-64191)"
    )
    parser.add_argument(
        "--deviation", type=str, default="small",
        choices=["nominal", "small", "large", "thermal_enhanced", "fast_enhanced"],
        help="A priori spectrum deviation level"
    )
    parser.add_argument(
        "--groups", type=int, default=100,
        help="Number of energy groups (100 or 640)"
    )
    parser.add_argument(
        "--method", type=str, default="GRAVEL",
        choices=["GRAVEL", "MLEM"],
        help="Unfolding algorithm"
    )
    
    args = parser.parse_args()
    
    results = run_reactor_dosimetry_workflow(
        apriori_deviation=args.deviation,
        n_groups=args.groups,
        unfolding_method=args.method,
    )
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
