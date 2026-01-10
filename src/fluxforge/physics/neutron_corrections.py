"""
Neutron Flux Corrections for Activation Analysis

This module provides corrections for neutron activation measurements:
- Self-shielding corrections (E1.4)
- Cadmium cover corrections (E1.5)

These corrections are essential for accurate reaction rate determination
in reactor dosimetry and neutron activation analysis (NAA).

References:
    ASTM E262: Self-shielding correction methods
    ASTM E481: Cd ratio measurements
    De Corte et al., JRNC 133 (1989) 43-130: k0-NAA formalism
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================

# Barn to cm² conversion
BARN_TO_CM2 = 1.0e-24

# Avogadro's number
N_AVOGADRO = 6.02214076e23

# Common nuclide atomic weights (g/mol)
ATOMIC_WEIGHTS = {
    "Au-197": 196.967,
    "Co-59": 58.933,
    "Fe-54": 53.940,
    "Fe-56": 55.845,
    "Ti-46": 45.953,
    "Ti-47": 46.952,
    "Ti-48": 47.948,
    "Ni-58": 57.935,
    "Ni-60": 59.931,
    "Cu-63": 62.930,
    "Cu-65": 64.928,
    "In-115": 114.904,
    "Zr-90": 89.905,
    "Nb-93": 92.906,
    "Na-23": 22.990,
    "Sc-45": 44.956,
    "Mn-55": 54.938,
}

# Thermal neutron cross sections at 2200 m/s (barn)
THERMAL_CROSS_SECTIONS = {
    "Au-197(n,g)Au-198": 98.65,
    "Co-59(n,g)Co-60": 37.18,
    "Fe-54(n,p)Mn-54": 0.0,  # Threshold reaction
    "Ti-46(n,p)Sc-46": 0.0,  # Threshold reaction
    "Ti-47(n,p)Sc-47": 0.0,  # Threshold reaction
    "Ti-48(n,p)Sc-48": 0.0,  # Threshold reaction
    "Ni-58(n,p)Co-58": 0.0,  # Threshold reaction
    "In-115(n,n')In-115m": 0.0,  # Threshold reaction
    "In-115(n,g)In-116m": 162.3,
    "Na-23(n,g)Na-24": 0.530,
    "Mn-55(n,g)Mn-56": 13.3,
}

# Resonance integral values (barn) - infinite dilution
RESONANCE_INTEGRALS = {
    "Au-197(n,g)Au-198": 1550.0,
    "Co-59(n,g)Co-60": 74.2,
    "In-115(n,g)In-116m": 3200.0,
    "Na-23(n,g)Na-24": 0.311,
    "Mn-55(n,g)Mn-56": 14.0,
}


# =============================================================================
# Self-Shielding Corrections
# =============================================================================

@dataclass
class SelfShieldingResult:
    """
    Result of self-shielding calculation.
    
    Attributes
    ----------
    G_th : float
        Thermal self-shielding factor (0 to 1)
    G_epi : float
        Epithermal self-shielding factor (0 to 1)
    G_total : float
        Combined self-shielding factor
    sigma_eff_th : float
        Effective thermal cross section (barn)
    sigma_eff_epi : float
        Effective resonance integral (barn)
    method : str
        Calculation method used
    """
    G_th: float = 1.0
    G_epi: float = 1.0
    G_total: float = 1.0
    sigma_eff_th: float = 0.0
    sigma_eff_epi: float = 0.0
    method: str = ""


def calculate_thermal_self_shielding_factor(
    sigma_0: float,
    thickness: float,
    n_density: float,
    geometry: str = "foil",
) -> float:
    """
    Calculate thermal neutron self-shielding factor G_th.
    
    The self-shielding factor accounts for flux depression within
    the sample due to neutron absorption.
    
    Parameters
    ----------
    sigma_0 : float
        Thermal cross section at 2200 m/s (barn)
    thickness : float
        Sample thickness (cm)
    n_density : float
        Number density (atoms/cm³)
    geometry : str
        Sample geometry: 'foil', 'wire', 'sphere'
    
    Returns
    -------
    float
        Thermal self-shielding factor G_th (0 to 1)
    
    Notes
    -----
    For thin samples (x << 1): G_th ≈ 1 - x/2
    For thick samples: G_th → 1/x (foil), 2/x (wire)
    
    where x = n * sigma * t is the optical thickness.
    """
    if sigma_0 <= 0:
        return 1.0
    
    # Calculate optical thickness
    sigma_cm2 = sigma_0 * BARN_TO_CM2
    x = n_density * sigma_cm2 * thickness
    
    if x < 1e-6:
        return 1.0
    
    if geometry == "foil":
        # For foil geometry (infinite slab):
        # G = (1 - exp(-x)) / x
        if x < 0.01:
            # Taylor expansion for small x
            G = 1.0 - x/2 + x**2/6 - x**3/24
        else:
            G = (1.0 - math.exp(-x)) / x
    
    elif geometry == "wire":
        # For cylindrical wire geometry:
        # G ≈ 2 * (1 - exp(-x)) / x for small radius
        # More accurate: use modified Bessel functions
        if x < 0.01:
            G = 1.0 - x/4 + x**2/12
        else:
            # Simplified approximation
            G = 2.0 * (1.0 - math.exp(-x/2)) / x
    
    elif geometry == "sphere":
        # For spherical geometry:
        # G = 3/(2x) * (1 - 2/x² * (1 - (1+x)*exp(-x)))
        if x < 0.01:
            G = 1.0 - x/3 + x**2/12
        else:
            term1 = 1.0 - (2.0/x**2) * (1.0 - (1.0 + x) * math.exp(-x))
            G = (3.0 / (2.0 * x)) * term1
    
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    
    return max(0.0, min(1.0, G))


def calculate_epithermal_self_shielding_factor(
    sigma_0: float,
    I_0: float,
    thickness: float,
    n_density: float,
    E_r: float = 5.0,
    geometry: str = "foil",
) -> float:
    """
    Calculate epithermal neutron self-shielding factor G_epi.
    
    For resonance reactions, the self-shielding is more complex
    due to the energy-dependent cross section structure.
    
    Parameters
    ----------
    sigma_0 : float
        Thermal cross section (barn)
    I_0 : float
        Infinite dilution resonance integral (barn)
    thickness : float
        Sample thickness (cm)
    n_density : float
        Number density (atoms/cm³)
    E_r : float
        Effective resonance energy (eV), default 5.0 eV
    geometry : str
        Sample geometry: 'foil', 'wire', 'sphere'
    
    Returns
    -------
    float
        Epithermal self-shielding factor G_epi (0 to 1)
    
    Notes
    -----
    Uses the simplified single-resonance approximation.
    For more accurate results, energy-dependent Monte Carlo
    or multi-group calculations are recommended.
    """
    if I_0 <= 0 or sigma_0 <= 0:
        return 1.0
    
    # Estimate peak resonance cross section
    # Using single-level Breit-Wigner approximation
    # sigma_peak ≈ I_0 / (2 * pi * Gamma) where Gamma is resonance width
    # Simplified: sigma_peak ≈ I_0 * 10 (typical ratio)
    sigma_peak = min(I_0 * 10, 100000)  # Cap at 100,000 barn
    
    # Calculate optical thickness at resonance peak
    sigma_cm2 = sigma_peak * BARN_TO_CM2
    x = n_density * sigma_cm2 * thickness
    
    if x < 1e-6:
        return 1.0
    
    # Use similar formulas as thermal, but with resonance cross section
    # Apply correction factor for resonance narrowing
    if geometry == "foil":
        if x < 0.01:
            G = 1.0 - x/2 + x**2/6
        else:
            G = (1.0 - math.exp(-x)) / x
    elif geometry == "wire":
        if x < 0.01:
            G = 1.0 - x/4
        else:
            G = 2.0 * (1.0 - math.exp(-x/2)) / x
    elif geometry == "sphere":
        if x < 0.01:
            G = 1.0 - x/3
        else:
            term1 = 1.0 - (2.0/x**2) * (1.0 - (1.0 + x) * math.exp(-x))
            G = (3.0 / (2.0 * x)) * term1
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
    
    # Apply resonance shape correction (accounts for 1/v wings)
    # Simplified: G_epi is typically closer to 1 than G_th for same sample
    G = 1.0 - 0.5 * (1.0 - G)  # Reduce correction by 50%
    
    return max(0.0, min(1.0, G))


def calculate_self_shielding(
    reaction: str,
    thickness: float,
    density: float,
    geometry: str = "foil",
    atom_fraction: float = 1.0,
    sigma_0: Optional[float] = None,
    I_0: Optional[float] = None,
    atomic_weight: Optional[float] = None,
) -> SelfShieldingResult:
    """
    Calculate complete self-shielding correction for a reaction.
    
    Parameters
    ----------
    reaction : str
        Reaction name, e.g., 'Au-197(n,g)Au-198'
    thickness : float
        Sample thickness (cm)
    density : float
        Material density (g/cm³)
    geometry : str
        Sample geometry: 'foil', 'wire', 'sphere'
    atom_fraction : float
        Atomic fraction of target nuclide in material
    sigma_0 : float, optional
        Override thermal cross section (barn)
    I_0 : float, optional
        Override resonance integral (barn)
    atomic_weight : float, optional
        Override atomic weight (g/mol)
    
    Returns
    -------
    SelfShieldingResult
        Self-shielding factors and effective cross sections
    
    Examples
    --------
    >>> result = calculate_self_shielding(
    ...     "Au-197(n,g)Au-198",
    ...     thickness=0.01,  # 100 μm foil
    ...     density=19.3,    # gold density
    ...     geometry="foil"
    ... )
    >>> print(f"G_th = {result.G_th:.3f}")
    >>> print(f"G_epi = {result.G_epi:.3f}")
    """
    result = SelfShieldingResult()
    result.method = f"Analytical ({geometry})"
    
    # Get cross sections from database or use provided values
    if sigma_0 is None:
        sigma_0 = THERMAL_CROSS_SECTIONS.get(reaction, 0.0)
    if I_0 is None:
        I_0 = RESONANCE_INTEGRALS.get(reaction, 0.0)
    
    # Get atomic weight
    if atomic_weight is None:
        # Extract target nuclide from reaction string
        target = reaction.split("(")[0]
        atomic_weight = ATOMIC_WEIGHTS.get(target, 50.0)
    
    # Calculate number density
    n_density = (density * N_AVOGADRO * atom_fraction) / atomic_weight
    
    # Calculate thermal self-shielding
    result.G_th = calculate_thermal_self_shielding_factor(
        sigma_0, thickness, n_density, geometry
    )
    result.sigma_eff_th = sigma_0 * result.G_th
    
    # Calculate epithermal self-shielding
    result.G_epi = calculate_epithermal_self_shielding_factor(
        sigma_0, I_0, thickness, n_density, geometry=geometry
    )
    result.sigma_eff_epi = I_0 * result.G_epi
    
    # Combined factor (weighted average for mixed spectrum)
    # Typical thermal reactor: ~70% thermal, ~30% epithermal contribution
    result.G_total = 0.7 * result.G_th + 0.3 * result.G_epi
    
    return result


# =============================================================================
# Cadmium Cover Corrections
# =============================================================================

@dataclass
class CdCoverResult:
    """
    Result of cadmium cover correction calculation.
    
    Attributes
    ----------
    F_Cd : float
        Cadmium correction factor
    R_Cd : float
        Cadmium ratio (bare/covered)
    I_eff : float
        Effective resonance integral (barn)
    thermal_fraction : float
        Fraction of reaction from thermal neutrons
    epithermal_fraction : float
        Fraction of reaction from epithermal neutrons
    """
    F_Cd: float = 1.0
    R_Cd: float = 1.0
    I_eff: float = 0.0
    thermal_fraction: float = 0.0
    epithermal_fraction: float = 0.0


# Cadmium cutoff energy (eV)
CD_CUTOFF_ENERGY = 0.55

# Cadmium transmission factors for common thicknesses
# F_Cd = fraction of epicadmium neutrons transmitted below cutoff
CD_TRANSMISSION_FACTORS = {
    0.5: 0.98,   # 0.5 mm Cd
    1.0: 0.99,   # 1.0 mm Cd (standard)
    2.0: 0.995,  # 2.0 mm Cd
}


def calculate_cd_ratio_correction(
    reaction: str,
    cd_thickness: float = 1.0,
    sigma_0: Optional[float] = None,
    I_0: Optional[float] = None,
) -> float:
    """
    Calculate the cadmium correction factor F_Cd.
    
    The Cd correction factor accounts for epithermal neutrons
    that are still absorbed by the Cd cover near the cutoff energy.
    
    Parameters
    ----------
    reaction : str
        Reaction name, e.g., 'Au-197(n,g)Au-198'
    cd_thickness : float
        Cadmium cover thickness (mm), default 1.0 mm
    sigma_0 : float, optional
        Thermal cross section (barn)
    I_0 : float, optional
        Resonance integral (barn)
    
    Returns
    -------
    float
        Cadmium correction factor F_Cd
    
    Notes
    -----
    F_Cd is typically close to 1 for most reactions.
    It becomes significant for 1/v absorbers like B-10.
    
    The corrected epithermal activity is:
        A_epi = A_Cd_covered / F_Cd
    """
    # Get cross sections
    if sigma_0 is None:
        sigma_0 = THERMAL_CROSS_SECTIONS.get(reaction, 0.0)
    if I_0 is None:
        I_0 = RESONANCE_INTEGRALS.get(reaction, 0.0)
    
    if I_0 <= 0:
        return 1.0
    
    # For 1/v absorbers, F_Cd depends on cross section ratio
    # F_Cd = 1 - (sigma_0 / I_0) * integral(sigma(E)/sigma_0 * phi(E) dE) 
    #        from 0 to E_Cd
    # Simplified approximation:
    if sigma_0 > 0:
        # 1/v contribution below Cd cutoff
        # integral of 1/sqrt(E) from 0.0253 eV to 0.55 eV
        # = 2 * sqrt(0.55) - 2 * sqrt(0.0253) ≈ 1.16
        one_v_integral = 1.16
        # Normalize to thermal equivalent
        one_v_contrib = sigma_0 * one_v_integral / I_0
        F_Cd = 1.0 - one_v_contrib * (1.0 - CD_TRANSMISSION_FACTORS.get(cd_thickness, 0.99))
    else:
        F_Cd = 1.0
    
    return max(0.5, min(1.0, F_Cd))


def calculate_cd_ratio(
    activity_bare: float,
    activity_cd_covered: float,
    uncertainty_bare: float = 0.0,
    uncertainty_covered: float = 0.0,
) -> Tuple[float, float]:
    """
    Calculate cadmium ratio from bare and covered measurements.
    
    Parameters
    ----------
    activity_bare : float
        Activity from bare (uncovered) sample
    activity_cd_covered : float
        Activity from Cd-covered sample
    uncertainty_bare : float
        Uncertainty in bare activity
    uncertainty_covered : float
        Uncertainty in covered activity
    
    Returns
    -------
    Tuple[float, float]
        Cadmium ratio and its uncertainty
    
    Notes
    -----
    R_Cd = A_bare / A_covered
    
    For purely thermal reactions: R_Cd → ∞
    For purely epithermal reactions: R_Cd → 1
    """
    if activity_cd_covered <= 0:
        return float('inf'), 0.0
    
    R_Cd = activity_bare / activity_cd_covered
    
    # Propagate uncertainty
    if R_Cd > 0 and (uncertainty_bare > 0 or uncertainty_covered > 0):
        rel_unc = math.sqrt(
            (uncertainty_bare / activity_bare) ** 2 +
            (uncertainty_covered / activity_cd_covered) ** 2
        )
        dR_Cd = R_Cd * rel_unc
    else:
        dR_Cd = 0.0
    
    return R_Cd, dR_Cd


def extract_thermal_epithermal_components(
    activity_bare: float,
    activity_cd_covered: float,
    reaction: str,
    f: float = 0.03,
    alpha: float = 0.0,
    cd_thickness: float = 1.0,
) -> CdCoverResult:
    """
    Separate thermal and epithermal reaction rate components.
    
    Uses the cadmium difference method to separate contributions
    from thermal and epithermal neutrons.
    
    Parameters
    ----------
    activity_bare : float
        Activity from bare (uncovered) sample
    activity_cd_covered : float
        Activity from Cd-covered sample
    reaction : str
        Reaction name for looking up cross sections
    f : float
        Thermal-to-epithermal flux ratio, default 0.03
    alpha : float
        Epithermal flux shape parameter, default 0
    cd_thickness : float
        Cadmium cover thickness (mm)
    
    Returns
    -------
    CdCoverResult
        Separated thermal and epithermal components
    
    Notes
    -----
    A_bare = A_thermal + A_epithermal
    A_Cd = A_epithermal / F_Cd
    A_thermal = A_bare - A_Cd * F_Cd
    """
    result = CdCoverResult()
    
    # Get cross sections
    sigma_0 = THERMAL_CROSS_SECTIONS.get(reaction, 0.0)
    I_0 = RESONANCE_INTEGRALS.get(reaction, 0.0)
    
    # Calculate Cd correction factor
    F_Cd = calculate_cd_ratio_correction(reaction, cd_thickness, sigma_0, I_0)
    result.F_Cd = F_Cd
    
    # Calculate Cd ratio
    if activity_cd_covered > 0:
        result.R_Cd = activity_bare / activity_cd_covered
    else:
        result.R_Cd = float('inf')
    
    # Separate components
    if activity_cd_covered > 0:
        # Epithermal component (corrected)
        A_epithermal = activity_cd_covered * F_Cd
        
        # Thermal component (by difference)
        A_thermal = activity_bare - A_epithermal
        
        # Calculate fractions
        total = activity_bare
        result.thermal_fraction = max(0, A_thermal) / total
        result.epithermal_fraction = min(1, A_epithermal / total)
    else:
        result.thermal_fraction = 1.0
        result.epithermal_fraction = 0.0
    
    # Calculate effective resonance integral
    # I_eff = I_0 - 0.426 * sigma_0 (Westcott convention)
    if I_0 > 0 and sigma_0 > 0:
        result.I_eff = I_0 - 0.426 * sigma_0
    else:
        result.I_eff = I_0
    
    return result


def apply_cd_cover_correction(
    reaction_rate: float,
    cd_ratio: float,
    reaction: str,
    f: float = 0.03,
) -> float:
    """
    Apply cadmium cover correction to get thermal reaction rate.
    
    Parameters
    ----------
    reaction_rate : float
        Total measured reaction rate (bare sample)
    cd_ratio : float
        Measured cadmium ratio R_Cd
    reaction : str
        Reaction name
    f : float
        Thermal-to-epithermal flux ratio
    
    Returns
    -------
    float
        Corrected thermal reaction rate
    
    Notes
    -----
    For thermal flux determination:
        phi_th = R / (sigma_0 * (1 - 1/R_Cd))
    
    For epithermal flux determination:
        phi_epi = R / (I_0 * f * (1/R_Cd - f/R_Cd))
    """
    if cd_ratio <= 1:
        # All epithermal
        return 0.0
    
    if cd_ratio == float('inf'):
        # All thermal
        return reaction_rate
    
    # Thermal fraction based on Cd ratio
    thermal_fraction = 1.0 - 1.0 / cd_ratio
    
    return reaction_rate * thermal_fraction


# =============================================================================
# Combined Corrections
# =============================================================================

@dataclass 
class NeutronCorrections:
    """
    Combined neutron flux corrections.
    
    Attributes
    ----------
    self_shielding : SelfShieldingResult
        Self-shielding correction factors
    cd_cover : CdCoverResult
        Cadmium cover correction factors
    total_correction : float
        Combined correction factor to apply to reaction rate
    """
    self_shielding: Optional[SelfShieldingResult] = None
    cd_cover: Optional[CdCoverResult] = None
    total_correction: float = 1.0


def calculate_all_corrections(
    reaction: str,
    thickness: float,
    density: float,
    geometry: str = "foil",
    activity_bare: Optional[float] = None,
    activity_cd_covered: Optional[float] = None,
    cd_thickness: float = 1.0,
    atom_fraction: float = 1.0,
) -> NeutronCorrections:
    """
    Calculate all applicable neutron corrections.
    
    Parameters
    ----------
    reaction : str
        Reaction name
    thickness : float
        Sample thickness (cm)
    density : float
        Material density (g/cm³)
    geometry : str
        Sample geometry
    activity_bare : float, optional
        Bare sample activity (for Cd correction)
    activity_cd_covered : float, optional
        Cd-covered activity (for Cd correction)
    cd_thickness : float
        Cd cover thickness (mm)
    atom_fraction : float
        Target nuclide fraction
    
    Returns
    -------
    NeutronCorrections
        All applicable corrections
    
    Examples
    --------
    >>> corr = calculate_all_corrections(
    ...     "Au-197(n,g)Au-198",
    ...     thickness=0.01,
    ...     density=19.3,
    ...     geometry="foil",
    ...     activity_bare=1000,
    ...     activity_cd_covered=100,
    ... )
    >>> print(f"Total correction: {corr.total_correction:.3f}")
    """
    result = NeutronCorrections()
    
    # Calculate self-shielding
    result.self_shielding = calculate_self_shielding(
        reaction=reaction,
        thickness=thickness,
        density=density,
        geometry=geometry,
        atom_fraction=atom_fraction,
    )
    
    # Calculate Cd cover corrections if data available
    if activity_bare is not None and activity_cd_covered is not None:
        result.cd_cover = extract_thermal_epithermal_components(
            activity_bare=activity_bare,
            activity_cd_covered=activity_cd_covered,
            reaction=reaction,
            cd_thickness=cd_thickness,
        )
    
    # Combined correction factor
    # Self-shielding affects measured activity (divide by G to correct)
    if result.self_shielding:
        ss_correction = 1.0 / result.self_shielding.G_total
    else:
        ss_correction = 1.0
    
    # Cd correction already gives separated components
    result.total_correction = ss_correction
    
    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "SelfShieldingResult",
    "CdCoverResult",
    "NeutronCorrections",
    # Self-shielding functions
    "calculate_thermal_self_shielding_factor",
    "calculate_epithermal_self_shielding_factor",
    "calculate_self_shielding",
    # Cd cover functions
    "calculate_cd_ratio_correction",
    "calculate_cd_ratio",
    "extract_thermal_epithermal_components",
    "apply_cd_cover_correction",
    # Combined
    "calculate_all_corrections",
    # Constants
    "THERMAL_CROSS_SECTIONS",
    "RESONANCE_INTEGRALS",
    "ATOMIC_WEIGHTS",
    "CD_CUTOFF_ENERGY",
]
