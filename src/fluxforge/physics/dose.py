"""
Dose Rate Calculations for Gamma Radiation

Provides functions for calculating dose rates from gamma-emitting isotopes,
including gamma-dose rate, isotope-dose rate integration, and fluence estimation.

Based on methods from irrad_spectroscopy and ICRP/NIST standards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# Default air mass energy-absorption coefficients (cm²/g) at various energies (keV)
# Source: NIST XCOM database for air
DEFAULT_AIR_COEFFICIENTS = {
    'energy': np.array([
        10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800,
        1000, 1250, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000
    ], dtype=float),  # keV
    'energy_absorption': np.array([
        4.614, 1.287, 0.529, 0.147, 0.0648, 0.0406, 0.0305, 0.0243, 0.0233, 
        0.0250, 0.0268, 0.0288, 0.0296, 0.0297, 0.0296, 0.0289, 0.0280, 
        0.0268, 0.0255, 0.0235, 0.0205, 0.0186, 0.0172, 0.0161, 0.0146, 0.0135
    ], dtype=float)  # cm²/g
}


@dataclass
class GammaLine:
    """Representation of a single gamma line with energy and emission probability."""
    energy_keV: float
    intensity: float  # probability per decay (0 to 1)
    energy_unc_keV: float = 0.0
    intensity_unc: float = 0.0


@dataclass
class DoseRateResult:
    """Container for dose rate calculation results."""
    dose_rate_uSv_h: float
    uncertainty_uSv_h: float = 0.0
    details: Optional[Dict] = None


def decay_constant(half_life_s: float) -> float:
    """Calculate decay constant lambda = ln(2) / t_half."""
    return math.log(2.0) / half_life_s


def decay_activity(initial_activity: float, half_life_s: float, time_s: float) -> float:
    """Calculate activity at time t given initial activity and half-life."""
    lam = decay_constant(half_life_s)
    return initial_activity * math.exp(-lam * time_s)


def mean_lifetime(half_life_s: float) -> float:
    """Calculate mean lifetime tau = 1/lambda."""
    return 1.0 / decay_constant(half_life_s)


def interpolate_coefficient(
    energy_keV: float,
    energies: np.ndarray = DEFAULT_AIR_COEFFICIENTS['energy'],
    coefficients: np.ndarray = DEFAULT_AIR_COEFFICIENTS['energy_absorption']
) -> float:
    """
    Interpolate mass energy-absorption coefficient at given energy.
    
    Uses log-log interpolation for physical accuracy.
    
    Parameters
    ----------
    energy_keV : float
        Gamma energy in keV
    energies : np.ndarray
        Reference energies in keV
    coefficients : np.ndarray
        Corresponding mass energy-absorption coefficients in cm²/g
    
    Returns
    -------
    float
        Interpolated coefficient in cm²/g
    """
    if energy_keV <= energies[0]:
        return coefficients[0]
    if energy_keV >= energies[-1]:
        return coefficients[-1]
    
    # Log-log interpolation
    log_E = np.log(energy_keV)
    log_energies = np.log(energies)
    log_coeffs = np.log(coefficients)
    
    return float(np.exp(np.interp(log_E, log_energies, log_coeffs)))


def gamma_dose_rate(
    energy_keV: float,
    intensity: float,
    activity_Bq: float,
    distance_cm: float,
    material: str = 'air'
) -> float:
    """
    Calculate dose rate from a single gamma line.
    
    Based on the Health Physics Society formula:
    https://hps.org/publicinformation/ate/faqs/gammaandexposure.html
    
    Parameters
    ----------
    energy_keV : float
        Gamma energy in keV
    intensity : float
        Emission probability per decay (0 to 1)
    activity_Bq : float
        Source activity in Bq (disintegrations per second)
    distance_cm : float
        Distance from source in cm
    material : str
        Absorbing material (default 'air')
    
    Returns
    -------
    float
        Dose rate in µSv/h
    """
    # Get mass energy-absorption coefficient
    mu_en = interpolate_coefficient(energy_keV)
    
    # Conversion factors:
    # 5.263e-6: exposure rate constant factor (R·cm²)/(h·keV·Bq)
    # 1/107.185: Roentgen to Sievert for air
    # 1e3: keV to MeV correction × Sv to µSv
    custom_factor = 5.263e-6 * (1.0 / 107.185) * 1e3
    
    dose_rate = custom_factor * energy_keV * intensity * mu_en * activity_Bq / (distance_cm ** 2)
    
    return dose_rate


def isotope_dose_rate(
    gamma_lines: List[GammaLine],
    activity_Bq: float,
    distance_cm: float,
    half_life_s: Optional[float] = None,
    integration_time_s: Optional[float] = None,
    material: str = 'air'
) -> DoseRateResult:
    """
    Calculate total dose rate from an isotope's gamma spectrum.
    
    Parameters
    ----------
    gamma_lines : List[GammaLine]
        List of gamma lines with energies and intensities
    activity_Bq : float
        Source activity in Bq
    distance_cm : float
        Distance from source in cm
    half_life_s : float, optional
        Half-life in seconds (needed for integrated dose)
    integration_time_s : float, optional
        If provided, integrate dose over this time period
    material : str
        Absorbing material (default 'air')
    
    Returns
    -------
    DoseRateResult
        Total dose rate result with uncertainty
    """
    total_dose_rate = 0.0
    total_variance = 0.0
    details = {}
    
    for i, line in enumerate(gamma_lines):
        dr = gamma_dose_rate(
            energy_keV=line.energy_keV,
            intensity=line.intensity,
            activity_Bq=activity_Bq,
            distance_cm=distance_cm,
            material=material
        )
        total_dose_rate += dr
        
        # Uncertainty propagation (intensity uncertainty dominates)
        if line.intensity > 0 and line.intensity_unc > 0:
            rel_unc = line.intensity_unc / line.intensity
            total_variance += (dr * rel_unc) ** 2
        
        details[f'line_{i}'] = {
            'energy_keV': line.energy_keV,
            'intensity': line.intensity,
            'dose_rate_uSv_h': dr
        }
    
    # Integrate over time if requested
    if integration_time_s is not None and half_life_s is not None:
        lam = decay_constant(half_life_s)
        # Integral of A0 * exp(-λt) from 0 to T = A0 * (1 - exp(-λT)) / λ
        # Convert from dose rate to dose: multiply by time factor
        if lam > 0:
            decay_factor = (1.0 - math.exp(-lam * integration_time_s)) / lam
            # Convert from per-second to per-hour for dose rate formula
            time_factor_h = integration_time_s / 3600.0
            # The integrated dose in µSv
            integrated_dose = total_dose_rate * decay_factor / 3600.0
            details['integrated_dose_uSv'] = integrated_dose
            details['integration_time_s'] = integration_time_s
    
    return DoseRateResult(
        dose_rate_uSv_h=total_dose_rate,
        uncertainty_uSv_h=math.sqrt(total_variance),
        details=details
    )


def fluence_from_activity(
    activity_Bq: float,
    cross_section_barn: float,
    molar_mass_g: float,
    sample_mass_g: float,
    irradiation_time_s: float,
    half_life_s: float,
    abundance: float = 1.0,
    cooldown_time_s: float = 0.0
) -> float:
    """
    Calculate neutron fluence from measured activation product activity.
    
    Parameters
    ----------
    activity_Bq : float
        Measured activity in Bq (at time of measurement)
    cross_section_barn : float
        Activation cross section in barn (1 barn = 1e-24 cm²)
    molar_mass_g : float
        Molar mass of target isotope in g/mol
    sample_mass_g : float
        Mass of sample in g
    irradiation_time_s : float
        Irradiation duration in seconds
    half_life_s : float
        Product half-life in seconds
    abundance : float
        Isotopic abundance fraction (0 to 1)
    cooldown_time_s : float
        Time between end of irradiation and measurement
    
    Returns
    -------
    float
        Neutron fluence in n/cm²
    """
    AVOGADRO = 6.02214076e23
    BARN_TO_CM2 = 1e-24
    
    lam = decay_constant(half_life_s)
    
    # Correct activity back to end of irradiation
    A_eoi = activity_Bq * math.exp(lam * cooldown_time_s)
    
    # Number of target atoms
    N_target = (sample_mass_g / molar_mass_g) * AVOGADRO * abundance
    
    # Cross section in cm²
    sigma_cm2 = cross_section_barn * BARN_TO_CM2
    
    # Saturation factor
    saturation = 1.0 - math.exp(-lam * irradiation_time_s)
    
    if saturation <= 0 or sigma_cm2 <= 0 or N_target <= 0:
        raise ValueError("Invalid parameters for fluence calculation")
    
    # Fluence = A / (N * sigma * lambda * saturation_factor)
    # Actually: A = N * sigma * phi * lambda * saturation where phi is flux (n/cm²/s)
    # For fluence Phi (n/cm²) = flux * time, need different formula
    # A = N * sigma * Phi_dot * saturation_factor * lambda
    # where saturation_factor = (1 - e^(-λT))/λ for constant flux
    
    # For fluence: Phi = A / (N * sigma * saturation_factor * lambda / λ)
    # Simplifies to: Phi = A * λ / (N * sigma * (1 - e^(-λT)))
    
    fluence = (A_eoi * lam) / (N_target * sigma_cm2 * saturation * lam)
    # Above simplifies to: A_eoi / (N_target * sigma_cm2 * saturation)
    
    return A_eoi / (N_target * sigma_cm2 * saturation)


@dataclass
class ShieldingLayer:
    """Representation of a shielding layer."""
    material: str
    thickness_cm: float
    linear_attenuation_coeff: float  # cm⁻¹ at reference energy


def attenuated_dose_rate(
    unshielded_dose_rate: float,
    shielding_layers: List[ShieldingLayer],
    buildup_factors: Optional[List[float]] = None
) -> float:
    """
    Calculate dose rate after passing through shielding.
    
    Parameters
    ----------
    unshielded_dose_rate : float
        Dose rate without shielding in µSv/h
    shielding_layers : List[ShieldingLayer]
        List of shielding layers
    buildup_factors : List[float], optional
        Buildup factors for each layer (default 1.0)
    
    Returns
    -------
    float
        Attenuated dose rate in µSv/h
    """
    if buildup_factors is None:
        buildup_factors = [1.0] * len(shielding_layers)
    
    transmission = 1.0
    for layer, B in zip(shielding_layers, buildup_factors):
        mu_x = layer.linear_attenuation_coeff * layer.thickness_cm
        transmission *= B * math.exp(-mu_x)
    
    return unshielded_dose_rate * transmission
