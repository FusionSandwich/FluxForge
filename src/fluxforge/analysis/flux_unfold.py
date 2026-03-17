"""
Flux Wire Spectrum Unfolding Module

Combines flux wire activation measurements with IRDFF-II cross sections
to unfold neutron spectra using various methods:
- Discrete N-bin unfolding
- GLS (Generalized Least Squares) continuous spectrum adjustment
- Regularized GRAVEL/MLEM iterative methods

Workflow:
1. Parse flux wire activation data (raw or processed)
2. Convert activities to reaction rates
3. Build response matrix from IRDFF-II cross sections
4. Unfold to obtain neutron spectrum
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fluxforge.analysis.flux_wire_analysis import (
    FLUX_WIRE_NUCLIDES,
    FluxWireAnalysisResult,
    analyze_flux_wire,
    analyze_flux_wire_targeted,
    get_sample_element,
    get_expected_isotopes,
)
from fluxforge.data.flux_wire_unfolding import (
    get_flux_wire_isotope_fraction,
    get_flux_wire_reaction_characteristic_energies,
    get_flux_wire_reaction_cross_section_defaults,
    get_flux_wire_reaction_id as get_reaction_id,
    get_flux_wire_response_parameters,
    load_flux_wire_product_reactions,
    load_flux_wire_reaction_defaults,
    load_flux_wire_sample_defaults,
)
from fluxforge.io.flux_wire import (
    FluxWireData,
    NuclideResult,
    read_processed_txt,
    read_raw_asc,
    load_flux_wire_directory,
)
from fluxforge.io.spe import GammaSpectrum
from fluxforge.core.response import (
    EnergyGroupStructure,
    ReactionCrossSection,
    ResponseMatrix,
    build_response_matrix,
)


# =============================================================================
# Reaction Rate Extraction
# =============================================================================

FLUX_WIRE_SAMPLES = load_flux_wire_sample_defaults()
THERMAL_CROSS_SECTIONS = get_flux_wire_reaction_cross_section_defaults()
REACTION_ENERGIES = get_flux_wire_reaction_characteristic_energies()
PRODUCT_REACTIONS = load_flux_wire_product_reactions()
REACTION_DEFAULTS = load_flux_wire_reaction_defaults()
REACTION_RESPONSE_PARAMS = get_flux_wire_response_parameters()

AVOGADRO = 6.02214076e23  # atoms/mol


def calculate_n_atoms(
    element: str,
    mass_mg: Optional[float] = None,
    isotope_fraction: float = 1.0,
) -> float:
    """
    Calculate number of target atoms in a flux wire sample.
    
    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Co', 'Cu')
    mass_mg : float, optional
        Sample mass in mg. Uses default from FLUX_WIRE_SAMPLES if None.
    isotope_fraction : float
        Fraction of target isotope (e.g., 0.6917 for Cu-63 in natural Cu)
        
    Returns
    -------
    float
        Number of target atoms
    """
    params = FLUX_WIRE_SAMPLES.get(element, {})
    if mass_mg is None:
        mass_mg = params.get('mass_mg', 10.0)
    
    atomic_mass = params.get('atomic_mass', 60.0)
    purity = params.get('purity', 0.9999)
    
    mass_g = mass_mg / 1000.0
    n_atoms = (mass_g * AVOGADRO / atomic_mass) * purity * isotope_fraction
    
    return n_atoms


@dataclass
class FluxWireReaction:
    """
    Represents a flux wire dosimetry reaction.
    
    Attributes
    ----------
    sample_id : str
        Sample identifier (e.g., "Co-Cd-RAFM-1_25cm")
    reaction_id : str
        IRDFF reaction name (e.g., "Co-59(n,g)Co-60")
    isotope : str
        Product isotope (e.g., "Co60")
    activity_bq : float
        Measured activity in Bq
    activity_unc_bq : float
        Activity uncertainty in Bq
    reaction_rate : float
        Reaction rate (reactions/atom/s)
    reaction_rate_unc : float
        Reaction rate uncertainty
    n_atoms : float
        Number of target atoms in sample
    irradiation_time_s : float
        Irradiation duration in seconds
    decay_time_s : float
        Time from end of irradiation to measurement
    """
    sample_id: str
    reaction_id: str
    isotope: str
    activity_bq: float
    activity_unc_bq: float = 0.0
    reaction_rate: float = 0.0
    reaction_rate_unc: float = 0.0
    n_atoms: float = 1.0
    irradiation_time_s: float = 0.0
    decay_time_s: float = 0.0

def activity_to_reaction_rate(
    activity_bq: float,
    n_atoms: float,
    half_life_s: float,
    irradiation_time_s: float,
    decay_time_s: float = 0.0,
    live_time_s: float = 0.0,
) -> float:
    """
    Convert measured activity to reaction rate.
    
    The activity at measurement time is related to reaction rate by:
    
        A = R * N * λ * [1 - exp(-λ*t_irr)] * exp(-λ*t_decay) * C_live
    
    where:
        R = reaction rate (reactions/atom/s) = σ × Φ
        N = number of target atoms
        λ = decay constant = ln(2) / t_half
        t_irr = irradiation time
        t_decay = decay time from end of irradiation to measurement
        C_live = live time correction factor
    
    Parameters
    ----------
    activity_bq : float
        Measured activity in Bq
    n_atoms : float
        Number of target atoms
    half_life_s : float
        Half-life of product isotope in seconds
    irradiation_time_s : float
        Total irradiation time in seconds
    decay_time_s : float
        Decay time from irradiation end to measurement start
    live_time_s : float
        Measurement live time (for decay during counting correction)
        
    Returns
    -------
    float
        Reaction rate in reactions/atom/s
    """
    if n_atoms <= 0 or half_life_s <= 0:
        return 0.0
    
    decay_const = np.log(2) / half_life_s
    
    # Saturation factor: accounts for buildup during irradiation
    if irradiation_time_s > 0:
        saturation = 1.0 - np.exp(-decay_const * irradiation_time_s)
    else:
        # Assume saturated (long irradiation)
        saturation = 1.0
    
    # Decay factor: accounts for decay between irradiation and measurement
    decay_factor = np.exp(-decay_const * decay_time_s) if decay_time_s > 0 else 1.0
    
    # Live time correction: average activity during counting
    if live_time_s > 0:
        live_correction = (1.0 - np.exp(-decay_const * live_time_s)) / (decay_const * live_time_s)
    else:
        live_correction = 1.0
    
    # Solve for reaction rate
    # A = R * N * saturation * decay_factor * live_correction
    denominator = n_atoms * saturation * decay_factor * live_correction
    
    if denominator <= 0:
        return 0.0
    
    return activity_bq / denominator


def get_isotope_fraction(reaction_id: str, element: str) -> float:
    """Get the isotopic abundance fraction for the target isotope."""
    return get_flux_wire_isotope_fraction(reaction_id, element)


def reaction_rate_to_flux(
    reaction_rate: float,
    reaction_id: str,
) -> Tuple[float, str]:
    """
    Convert reaction rate to flux using appropriate cross section.
    
    Parameters
    ----------
    reaction_rate : float
        Reaction rate in reactions/atom/s
    reaction_id : str
        IRDFF reaction identifier
        
    Returns
    -------
    tuple of (float, str)
        Flux in n/cm²/s and the flux type (thermal, epithermal, or fast)
    """
    xs_data = THERMAL_CROSS_SECTIONS.get(reaction_id, {})
    
    # Thermal (n,g) reactions
    if 'sigma_thermal' in xs_data and xs_data.get('sigma_thermal', 0) > 0:
        sigma_thermal = xs_data['sigma_thermal'] * 1e-24  # Convert barns to cm²
        flux = reaction_rate / sigma_thermal
        return flux, 'thermal'
    
    # Threshold reactions
    elif 'sigma_avg' in xs_data:
        sigma_avg = xs_data['sigma_avg'] * 1e-24  # Convert barns to cm²
        flux = reaction_rate / sigma_avg
        return flux, 'fast'
    
    return 0.0, 'unknown'


def extract_reactions_from_processed(
    data: FluxWireData,
    sample_mass_mg: Optional[float] = None,  # Use default if None
    irradiation_time_s: float = 3600.0,  # Default 1 hour
    decay_time_s: float = 0.0,
    calculate_flux: bool = True,
) -> List[FluxWireReaction]:
    """
    Extract reaction information from processed flux wire data.
    
    Parameters
    ----------
    data : FluxWireData
        Processed flux wire data with nuclide results
    sample_mass_mg : float, optional
        Sample mass in mg. Uses default from FLUX_WIRE_SAMPLES if None.
    irradiation_time_s : float
        Irradiation time in seconds
    decay_time_s : float
        Decay time from irradiation to measurement
    calculate_flux : bool
        If True, also calculate flux from reaction rate
        
    Returns
    -------
    list of FluxWireReaction
        Extracted reactions with activities, reaction rates, and flux
    """
    reactions = []
    sample_element = get_sample_element(data.sample_id)
    
    for nuclide in data.nuclides:
        isotope = nuclide.isotope
        activity_bq = nuclide.activity_bq
        activity_unc = nuclide.activity_unc * 3.7e4 if nuclide.activity_unc else activity_bq * 0.1
        
        # Get half-life
        half_life_s = FLUX_WIRE_NUCLIDES.get(isotope, {}).get('half_life_s', 0)
        if half_life_s <= 0:
            # Try to parse from data
            half_life_s = nuclide.half_life_s if hasattr(nuclide, 'half_life_s') else 0
        
        # Get reaction ID
        reaction_id = get_reaction_id(isotope, sample_element)
        
        # Calculate number of target atoms
        isotope_fraction = get_isotope_fraction(reaction_id, sample_element or '')
        n_atoms = calculate_n_atoms(
            element=sample_element or 'Co',
            mass_mg=sample_mass_mg,
            isotope_fraction=isotope_fraction,
        )
        
        # Calculate reaction rate
        if half_life_s > 0 and n_atoms > 0:
            rate = activity_to_reaction_rate(
                activity_bq=activity_bq,
                n_atoms=n_atoms,
                half_life_s=half_life_s,
                irradiation_time_s=irradiation_time_s,
                decay_time_s=decay_time_s,
                live_time_s=data.live_time,
            )
            rate_unc = rate * (activity_unc / activity_bq) if activity_bq > 0 else 0
        else:
            rate = 0.0
            rate_unc = 0.0
        
        # Calculate flux from reaction rate
        flux = 0.0
        flux_type = 'unknown'
        if calculate_flux and rate > 0:
            flux, flux_type = reaction_rate_to_flux(rate, reaction_id)
        
        rxn = FluxWireReaction(
            sample_id=data.sample_id,
            reaction_id=reaction_id,
            isotope=isotope,
            activity_bq=activity_bq,
            activity_unc_bq=activity_unc,
            reaction_rate=rate,
            reaction_rate_unc=rate_unc,
            n_atoms=n_atoms,
            irradiation_time_s=irradiation_time_s,
            decay_time_s=decay_time_s,
        )
        # Add flux as extra attribute
        rxn.flux = flux
        rxn.flux_type = flux_type
        
        reactions.append(rxn)
    
    return reactions


def extract_reactions_from_raw(
    raw_data: FluxWireData,
    reference_data: Optional[FluxWireData] = None,
    sample_mass_mg: Optional[float] = None,
    irradiation_time_s: float = 3600.0,
    decay_time_s: float = 0.0,
    calculate_flux: bool = True,
    peak_threshold: float = 0.0,
    background_spectrum: Optional[GammaSpectrum] = None,
    background_scale_mode: str = "live",
    background_scale_factor: Optional[float] = None,
    background_subtract: bool = True,
    profile_name: Optional[str] = None,
) -> List[FluxWireReaction]:
    """
    Extract reaction information from raw flux wire spectra.
    
    This uses targeted peak extraction. Optional processed references are
    comparison-only and do not alter the raw-spectrum analysis path.
    """
    if raw_data.has_spectrum is False:
        return []
    
    # Targeted analysis for activities
    analysis = analyze_flux_wire_targeted(
        data=raw_data,
        reference_data=reference_data,
        peak_threshold=peak_threshold,
        min_energy_keV=80.0,
        max_energy_keV=3000.0,
        background_spectrum=background_spectrum,
        background_scale_mode=background_scale_mode,
        background_scale_factor=background_scale_factor,
        background_subtract=background_subtract,
        profile_name=profile_name,
    )
    
    sample_element = get_sample_element(raw_data.sample_id)
    reactions: List[FluxWireReaction] = []
    
    for isotope, activity in analysis.nuclide_activities.items():
        activity_bq = activity.get('activity_bq', 0.0)
        activity_unc = activity.get('activity_unc_bq', activity_bq * 0.1)
        
        # Get half-life
        half_life_s = FLUX_WIRE_NUCLIDES.get(isotope, {}).get('half_life_s', 0.0)
        
        # Get reaction ID
        reaction_id = get_reaction_id(isotope, sample_element)
        
        # Calculate number of target atoms
        isotope_fraction = get_isotope_fraction(reaction_id, sample_element or '')
        n_atoms = calculate_n_atoms(
            element=sample_element or 'Co',
            mass_mg=sample_mass_mg,
            isotope_fraction=isotope_fraction,
        )
        
        # Calculate reaction rate
        if half_life_s > 0 and n_atoms > 0 and activity_bq > 0:
            rate = activity_to_reaction_rate(
                activity_bq=activity_bq,
                n_atoms=n_atoms,
                half_life_s=half_life_s,
                irradiation_time_s=irradiation_time_s,
                decay_time_s=decay_time_s,
                live_time_s=raw_data.live_time,
            )
            rate_unc = rate * (activity_unc / activity_bq) if activity_bq > 0 else 0.0
        else:
            rate = 0.0
            rate_unc = 0.0
        
        # Calculate flux from reaction rate
        flux = 0.0
        flux_type = 'unknown'
        if calculate_flux and rate > 0:
            flux, flux_type = reaction_rate_to_flux(rate, reaction_id)
        
        rxn = FluxWireReaction(
            sample_id=raw_data.sample_id,
            reaction_id=reaction_id,
            isotope=isotope,
            activity_bq=activity_bq,
            activity_unc_bq=activity_unc,
            reaction_rate=rate,
            reaction_rate_unc=rate_unc,
            n_atoms=n_atoms,
            irradiation_time_s=irradiation_time_s,
            decay_time_s=decay_time_s,
        )
        rxn.flux = flux
        rxn.flux_type = flux_type
        
        reactions.append(rxn)
    
    return reactions


# =============================================================================
# Energy Group Structures
# =============================================================================

def make_equal_lethargy_groups(n_groups: int, e_min_eV: float = 0.0253, e_max_eV: float = 2e7) -> EnergyGroupStructure:
    """
    Create equal-lethargy energy group structure.
    
    Parameters
    ----------
    n_groups : int
        Number of energy groups
    e_min_eV : float
        Minimum energy in eV (default: thermal 0.0253 eV)
    e_max_eV : float
        Maximum energy in eV (default: 20 MeV)
        
    Returns
    -------
    EnergyGroupStructure
        Energy group boundaries
    """
    log_min = np.log(e_min_eV)
    log_max = np.log(e_max_eV)
    boundaries = np.exp(np.linspace(log_min, log_max, n_groups + 1))
    return EnergyGroupStructure(boundaries_eV=list(boundaries))


def make_vitamin_j_175_groups() -> EnergyGroupStructure:
    """
    Create VITAMIN-J 175-group structure (common for fusion).
    
    Returns
    -------
    EnergyGroupStructure
        VITAMIN-J 175-group boundaries
    """
    # Standard 175-group boundaries from VITAMIN-J
    # Simplified version - would need full list from library
    # For now, create approximate equal-lethargy structure
    return make_equal_lethargy_groups(175, e_min_eV=1e-5, e_max_eV=2e7)


# =============================================================================
# Discrete N-bin Unfolding
# =============================================================================

@dataclass
class DiscreteUnfoldResult:
    """
    Result from discrete N-bin spectrum unfolding.
    
    Attributes
    ----------
    energy_bounds_eV : np.ndarray
        Energy group boundaries (N+1 values)
    flux : np.ndarray
        Flux in each energy bin (N values)
    flux_unc : np.ndarray
        Flux uncertainties
    reactions : list
        List of reactions used
    chi2 : float
        Chi-squared of fit
    """
    energy_bounds_eV: np.ndarray
    flux: np.ndarray
    flux_unc: np.ndarray
    reactions: List[FluxWireReaction]
    chi2: float = 0.0
    method: str = "discrete"


def unfold_discrete_bins(
    reactions: List[FluxWireReaction],
    n_bins: int = 10,
    e_min_eV: float = 0.0253,
    e_max_eV: float = 2e7,
    cross_sections: Optional[Dict[str, np.ndarray]] = None,
) -> DiscreteUnfoldResult:
    """
    Perform discrete N-bin spectrum unfolding.
    
    This is a simple approach where we assign each reaction to one dominant
    energy bin based on its threshold or resonance energy.
    
    Parameters
    ----------
    reactions : list of FluxWireReaction
        Measured reactions with activities
    n_bins : int
        Number of energy bins
    e_min_eV, e_max_eV : float
        Energy range
    cross_sections : dict, optional
        Pre-loaded cross section data
        
    Returns
    -------
    DiscreteUnfoldResult
        Unfolded spectrum
    """
    # Create energy groups
    groups = make_equal_lethargy_groups(n_bins, e_min_eV, e_max_eV)
    boundaries = np.array(groups.boundaries_eV)
    bin_centers = np.sqrt(boundaries[:-1] * boundaries[1:])
    
    # Initialize flux arrays
    flux = np.zeros(n_bins)
    flux_unc = np.zeros(n_bins)
    flux_count = np.zeros(n_bins)  # Number of reactions per bin
    
    # Assign each reaction to appropriate bin
    for rxn in reactions:
        if rxn.reaction_rate <= 0:
            continue
        
        # Get characteristic energy for this reaction
        e_char = REACTION_ENERGIES.get(rxn.reaction_id, 1e6)  # Default to 1 MeV
        
        # Find bin containing this energy
        bin_idx = np.searchsorted(boundaries[1:], e_char)
        bin_idx = min(bin_idx, n_bins - 1)
        
        # Use flux value if available, otherwise use reaction rate
        rxn_flux = getattr(rxn, 'flux', 0.0)
        if rxn_flux > 0:
            flux[bin_idx] += rxn_flux
            flux_unc[bin_idx] += (rxn_flux * rxn.reaction_rate_unc / rxn.reaction_rate if rxn.reaction_rate > 0 else 0)**2
        else:
            # Fallback to reaction rate
            flux[bin_idx] += rxn.reaction_rate
            flux_unc[bin_idx] += rxn.reaction_rate_unc**2
        flux_count[bin_idx] += 1
    
    # Average where multiple reactions contribute
    for i in range(n_bins):
        if flux_count[i] > 1:
            flux[i] /= flux_count[i]
            flux_unc[i] = np.sqrt(flux_unc[i]) / flux_count[i]
        elif flux_count[i] == 1:
            flux_unc[i] = np.sqrt(flux_unc[i])
    
    return DiscreteUnfoldResult(
        energy_bounds_eV=boundaries,
        flux=flux,
        flux_unc=flux_unc,
        reactions=reactions,
        method="discrete_binning",
    )


# =============================================================================
# GLS Spectrum Adjustment
# =============================================================================

@dataclass
class GLSUnfoldResult:
    """
    Result from GLS spectrum unfolding.
    
    Attributes
    ----------
    energy_bounds_eV : np.ndarray
        Energy group boundaries
    flux : np.ndarray
        Adjusted flux spectrum
    flux_unc : np.ndarray
        Flux uncertainties from covariance diagonal
    covariance : np.ndarray
        Full posterior covariance matrix
    chi2 : float
        Chi-squared of adjustment
    reactions : list
        Reactions used in unfolding
    """
    energy_bounds_eV: np.ndarray
    flux: np.ndarray
    flux_unc: np.ndarray
    covariance: np.ndarray
    chi2: float
    reactions: List[FluxWireReaction]
    method: str = "GLS"


def unfold_gls(
    reactions: List[FluxWireReaction],
    n_groups: int = 20,  # Reduced default for stability
    e_min_eV: float = 0.0253,
    e_max_eV: float = 2e7,
    prior_flux: Optional[np.ndarray] = None,
    prior_uncertainty: float = 2.0,  # Prior relative uncertainty (more constraining)
    cross_sections: Optional[Dict[str, np.ndarray]] = None,
    regularization: float = 1e-10,  # Regularization for matrix stability
) -> GLSUnfoldResult:
    """
    Perform GLS spectrum adjustment unfolding.
    
    Uses Generalized Least Squares to adjust a prior spectrum to match
    the measured reaction rates.
    
    Parameters
    ----------
    reactions : list of FluxWireReaction
        Measured reactions with reaction rates
    n_groups : int
        Number of energy groups for output spectrum
    e_min_eV, e_max_eV : float
        Energy range
    prior_flux : np.ndarray, optional
        Prior guess spectrum. If None, uses flat 1/E spectrum.
    prior_uncertainty : float
        Relative uncertainty on prior (as multiplicative factor)
    cross_sections : dict, optional
        Pre-loaded group cross sections
    regularization : float
        Small value added to covariance diagonals for stability
        
    Returns
    -------
    GLSUnfoldResult
        Unfolded spectrum with covariance
    """
    # Create energy groups
    groups = make_equal_lethargy_groups(n_groups, e_min_eV, e_max_eV)
    boundaries = np.array(groups.boundaries_eV)
    
    # Filter reactions with valid flux or rates
    valid_reactions = [r for r in reactions if getattr(r, 'flux', 0) > 0 or r.reaction_rate > 0]
    n_reactions = len(valid_reactions)
    
    if n_reactions == 0:
        raise ValueError("No valid reactions with positive flux or reaction rates")
    
    # Create prior flux (1/E spectrum if not provided)
    if prior_flux is None:
        bin_widths = boundaries[1:] - boundaries[:-1]
        bin_centers = np.sqrt(boundaries[:-1] * boundaries[1:])
        prior_flux = bin_widths / bin_centers  # 1/E spectrum
        # Scale to reasonable magnitude based on flux values
        flux_values = [getattr(r, 'flux', 0) for r in valid_reactions if getattr(r, 'flux', 0) > 0]
        if flux_values:
            avg_flux = np.mean(flux_values)
        else:
            avg_flux = np.mean([r.reaction_rate for r in valid_reactions])
        prior_flux = prior_flux / prior_flux.sum() * avg_flux * n_groups
    
    prior_flux = np.array(prior_flux)
    
    # Prior covariance (diagonal with specified uncertainty + regularization)
    prior_cov = np.diag((prior_flux * prior_uncertainty)**2 + regularization)
    
    # Build response matrix using numpy
    response = np.zeros((n_reactions, n_groups))
    measurements = np.zeros(n_reactions)
    measurement_unc = np.zeros(n_reactions)
    
    for i, rxn in enumerate(valid_reactions):
        # Create row of response matrix
        row = _make_response_row(rxn.reaction_id, boundaries, n_groups)
        response[i, :] = row
        # Use flux if available, otherwise use reaction rate
        rxn_flux = getattr(rxn, 'flux', 0.0)
        if rxn_flux > 0:
            measurements[i] = rxn_flux
            measurement_unc[i] = max(rxn_flux * rxn.reaction_rate_unc / rxn.reaction_rate if rxn.reaction_rate > 0 else 0.1 * rxn_flux, 0.1 * rxn_flux)
        else:
            measurements[i] = rxn.reaction_rate
            measurement_unc[i] = max(rxn.reaction_rate_unc, 0.1 * rxn.reaction_rate)
    
    # Measurement covariance (diagonal) with regularization
    measurement_cov = np.diag(measurement_unc**2 + regularization)
    
    # GLS adjustment using numpy (more robust than pure Python version)
    # φ_hat = φ_prior + K * (m - R * φ_prior)
    # K = C_prior * R^T * (R * C_prior * R^T + C_meas)^{-1}
    
    try:
        # Innovation covariance: S = R * C_prior * R^T + C_meas
        RC = response @ prior_cov
        S = RC @ response.T + measurement_cov
        
        # Add extra regularization if needed
        S += np.eye(n_reactions) * regularization * np.max(np.diag(S))
        
        # Kalman gain: K = C_prior * R^T * S^{-1}
        # Use pseudo-inverse for robustness
        S_inv = np.linalg.pinv(S, rcond=1e-10)
        K = prior_cov @ response.T @ S_inv
        
        # Residual
        predicted = response @ prior_flux
        residual = measurements - predicted
        
        # Update
        phi_hat = prior_flux + K @ residual
        
        # Enforce non-negativity
        phi_hat = np.maximum(phi_hat, 0.0)
        
        # Posterior covariance: C_post = C_prior - K * R * C_prior
        posterior_cov = prior_cov - K @ response @ prior_cov
        
        # Chi-squared
        chi2 = float(residual @ S_inv @ residual)
        
    except np.linalg.LinAlgError as e:
        raise ValueError(f"GLS adjustment failed: {e}")
    
    flux_unc = np.sqrt(np.maximum(np.diag(posterior_cov), 0))
    
    return GLSUnfoldResult(
        energy_bounds_eV=boundaries,
        flux=phi_hat,
        flux_unc=flux_unc,
        covariance=posterior_cov,
        chi2=chi2,
        reactions=valid_reactions,
        method="GLS",
    )


def _make_response_row(
    reaction_id: str,
    energy_bounds: np.ndarray,
    n_groups: int,
) -> List[float]:
    """
    Create simplified response matrix row for a reaction.
    
    This is a placeholder that creates a Gaussian-like response
    centered on the reaction's threshold/resonance energy.
    For production use, would load actual IRDFF-II cross sections.
    """
    e_center, log_width = REACTION_RESPONSE_PARAMS.get(reaction_id, (1e6, 1.0))
    
    # Create response as Gaussian in log-energy space
    bin_centers = np.sqrt(energy_bounds[:-1] * energy_bounds[1:])
    log_centers = np.log(bin_centers)
    log_e0 = np.log(e_center)
    
    # Gaussian response
    response = np.exp(-0.5 * ((log_centers - log_e0) / log_width)**2)
    
    # Normalize so total response is reasonable
    response = response / response.sum() if response.sum() > 0 else response
    
    return list(response)


# =============================================================================
# High-Level Workflow Functions
# =============================================================================

@dataclass
class FluxWireUnfoldResult:
    """
    Complete result from flux wire spectrum unfolding.
    """
    discrete_result: Optional[DiscreteUnfoldResult] = None
    gls_result: Optional[GLSUnfoldResult] = None
    reactions: List[FluxWireReaction] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'n_reactions': len(self.reactions),
            'source_files': self.source_files,
        }
        
        if self.discrete_result:
            result['discrete'] = {
                'energy_bounds_eV': self.discrete_result.energy_bounds_eV.tolist(),
                'flux': self.discrete_result.flux.tolist(),
                'flux_unc': self.discrete_result.flux_unc.tolist(),
                'chi2': self.discrete_result.chi2,
            }
        
        if self.gls_result:
            result['gls'] = {
                'energy_bounds_eV': self.gls_result.energy_bounds_eV.tolist(),
                'flux': self.gls_result.flux.tolist(),
                'flux_unc': self.gls_result.flux_unc.tolist(),
                'chi2': self.gls_result.chi2,
            }
        
        return result


def unfold_flux_wires(
    flux_wire_files: List[Union[str, Path]],
    n_discrete_bins: int = 10,
    n_gls_groups: int = 50,
    irradiation_time_s: float = 3600.0,
    n_atoms: float = 1e20,
    verbose: bool = True,
) -> FluxWireUnfoldResult:
    """
    Unfold neutron spectrum from flux wire measurements.
    
    Parameters
    ----------
    flux_wire_files : list of str or Path
        Paths to processed flux wire files (.txt)
    n_discrete_bins : int
        Number of bins for discrete unfolding
    n_gls_groups : int
        Number of groups for GLS unfolding
    irradiation_time_s : float
        Irradiation time in seconds
    n_atoms : float
        Approximate number of target atoms per sample
    verbose : bool
        Print progress information
        
    Returns
    -------
    FluxWireUnfoldResult
        Complete unfolding results
    """
    if verbose:
        print("=" * 80)
        print("FLUX WIRE SPECTRUM UNFOLDING")
        print("=" * 80)
    
    # Load all flux wire files
    all_reactions = []
    source_files = []
    
    for filepath in flux_wire_files:
        filepath = Path(filepath)
        if not filepath.exists():
            if verbose:
                print(f"Warning: File not found: {filepath}")
            continue
        
        source_files.append(str(filepath))
        
        try:
            data = read_processed_txt(filepath)
            reactions = extract_reactions_from_processed(
                data,
                irradiation_time_s=irradiation_time_s,
            )
            all_reactions.extend(reactions)
            
            if verbose:
                print(f"\n{data.sample_id}:")
                for rxn in reactions:
                    print(f"  {rxn.reaction_id}: {rxn.activity_bq:.3e} Bq, "
                          f"R = {rxn.reaction_rate:.3e} /atom/s")
        
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process {filepath}: {e}")
    
    if not all_reactions:
        raise ValueError("No valid reactions extracted from files")
    
    if verbose:
        print(f"\nTotal reactions: {len(all_reactions)}")
    
    result = FluxWireUnfoldResult(
        reactions=all_reactions,
        source_files=source_files,
    )
    
    # Discrete binning
    if verbose:
        print(f"\n--- Discrete {n_discrete_bins}-bin unfolding ---")
    
    try:
        discrete = unfold_discrete_bins(
            reactions=all_reactions,
            n_bins=n_discrete_bins,
        )
        result.discrete_result = discrete
        
        if verbose:
            print("\nDiscrete spectrum (non-zero bins):")
            for i in range(len(discrete.flux)):
                if discrete.flux[i] > 0:
                    e_lo = discrete.energy_bounds_eV[i]
                    e_hi = discrete.energy_bounds_eV[i+1]
                    print(f"  {e_lo:.2e} - {e_hi:.2e} eV: {discrete.flux[i]:.3e}")
    
    except Exception as e:
        if verbose:
            print(f"Discrete unfolding failed: {e}")
    
    # GLS unfolding
    if verbose:
        print(f"\n--- GLS {n_gls_groups}-group adjustment ---")
    
    try:
        gls = unfold_gls(
            reactions=all_reactions,
            n_groups=n_gls_groups,
        )
        result.gls_result = gls
        
        if verbose:
            print(f"Chi-squared: {gls.chi2:.3f}")
            print(f"Total flux: {gls.flux.sum():.3e}")
    
    except Exception as e:
        if verbose:
            print(f"GLS unfolding failed: {e}")
    
    return result


def plot_unfolded_spectrum(
    result: FluxWireUnfoldResult,
    output_file: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot unfolded spectrum results.
    
    Parameters
    ----------
    result : FluxWireUnfoldResult
        Unfolding results
    output_file : str, optional
        Path to save figure
    show : bool
        Whether to display figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot discrete spectrum
    if result.discrete_result is not None:
        ax = axes[0]
        dr = result.discrete_result
        
        # Bar plot
        widths = np.diff(dr.energy_bounds_eV)
        centers = np.sqrt(dr.energy_bounds_eV[:-1] * dr.energy_bounds_eV[1:])
        
        ax.bar(centers, dr.flux, width=widths*0.8, alpha=0.7, label='Flux')
        ax.errorbar(centers, dr.flux, yerr=dr.flux_unc, fmt='none', 
                   color='black', capsize=2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Flux (arbitrary)')
        ax.set_title(f'Discrete {len(dr.flux)}-bin Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot GLS spectrum
    if result.gls_result is not None:
        ax = axes[1]
        gr = result.gls_result
        
        centers = np.sqrt(gr.energy_bounds_eV[:-1] * gr.energy_bounds_eV[1:])
        
        ax.fill_between(centers, 
                       gr.flux - gr.flux_unc, 
                       gr.flux + gr.flux_unc,
                       alpha=0.3, label='±1σ')
        ax.plot(centers, gr.flux, 'b-', linewidth=1.5, label='GLS adjusted')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Flux (arbitrary)')
        ax.set_title(f'GLS {len(gr.flux)}-group Spectrum (χ²={gr.chi2:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    if show:
        plt.show()


# =============================================================================
# Command-line interface helper
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Demo with example data
    proc_dir = Path("/filespace/s/smandych/CAE/projects/ALARA/rafm_irradiation_ldrd/irradiation_QG_processed/flux_wires")
    
    if proc_dir.exists():
        files = sorted(proc_dir.glob("*.txt"))
        print(f"Found {len(files)} flux wire files")
        
        result = unfold_flux_wires(
            flux_wire_files=files,
            n_discrete_bins=10,
            n_gls_groups=50,
            verbose=True,
        )
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Files processed: {len(result.source_files)}")
        print(f"Total reactions: {len(result.reactions)}")
        
        if result.discrete_result:
            print(f"Discrete bins: {len(result.discrete_result.flux)}")
        
        if result.gls_result:
            print(f"GLS groups: {len(result.gls_result.flux)}")
            print(f"GLS chi2: {result.gls_result.chi2:.3f}")
