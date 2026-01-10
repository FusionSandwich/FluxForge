"""
Gamma self-attenuation corrections for activity inference.

This module provides corrections for gamma-ray attenuation within the
sample and container, which is mandatory for activity inference from
HPGe measurements when material/geometry warrants it.

This correction is SEPARATE from neutron self-shielding and belongs
to the HPGe activity path (Stage C in the FluxForge pipeline).

Reference: Standard gamma spectroscopy corrections
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class SampleGeometry(Enum):
    """Sample geometry types for attenuation calculation."""
    
    POINT = "point"  # Point source approximation
    DISK = "disk"  # Thin disk/foil
    CYLINDER = "cylinder"  # Cylindrical sample
    SPHERE = "sphere"  # Spherical sample
    MARINELLI = "marinelli"  # Marinelli beaker
    BOX = "box"  # Rectangular box


@dataclass
class MaterialAttenuation:
    """
    Material gamma-ray attenuation properties.
    
    Attributes:
        name: Material name
        density_g_cm3: Mass density
        mass_atten_coeff: Dict mapping energy (keV) to μ/ρ (cm²/g)
    """
    
    name: str
    density_g_cm3: float
    mass_atten_coeff: Dict[float, float]  # energy_keV -> μ/ρ in cm²/g
    
    def get_mu(self, energy_kev: float) -> float:
        """
        Get linear attenuation coefficient μ at given energy.
        
        Uses log-log interpolation between tabulated points.
        
        Args:
            energy_kev: Gamma energy in keV
            
        Returns:
            Linear attenuation coefficient μ in cm⁻¹
        """
        energies = sorted(self.mass_atten_coeff.keys())
        mu_rho_values = [self.mass_atten_coeff[e] for e in energies]
        
        if energy_kev <= energies[0]:
            mu_rho = mu_rho_values[0]
        elif energy_kev >= energies[-1]:
            mu_rho = mu_rho_values[-1]
        else:
            # Log-log interpolation
            log_E = np.log(energies)
            log_mu = np.log(mu_rho_values)
            log_E_target = np.log(energy_kev)
            log_mu_interp = np.interp(log_E_target, log_E, log_mu)
            mu_rho = np.exp(log_mu_interp)
        
        return mu_rho * self.density_g_cm3


# Common material attenuation data (μ/ρ in cm²/g)
# Values at selected energies from NIST XCOM
STANDARD_MATERIALS = {
    "iron": MaterialAttenuation(
        name="Iron",
        density_g_cm3=7.87,
        mass_atten_coeff={
            100: 0.371,
            200: 0.147,
            500: 0.0838,
            1000: 0.0599,
            1500: 0.0485,
            2000: 0.0426,
        }
    ),
    "aluminum": MaterialAttenuation(
        name="Aluminum",
        density_g_cm3=2.70,
        mass_atten_coeff={
            100: 0.171,
            200: 0.122,
            500: 0.0843,
            1000: 0.0615,
            1500: 0.0508,
            2000: 0.0454,
        }
    ),
    "gold": MaterialAttenuation(
        name="Gold",
        density_g_cm3=19.3,
        mass_atten_coeff={
            100: 5.16,
            200: 1.09,
            500: 0.178,
            1000: 0.0686,
            1500: 0.0484,
            2000: 0.0414,
        }
    ),
    "copper": MaterialAttenuation(
        name="Copper",
        density_g_cm3=8.96,
        mass_atten_coeff={
            100: 0.456,
            200: 0.150,
            500: 0.0836,
            1000: 0.0590,
            1500: 0.0478,
            2000: 0.0420,
        }
    ),
    "nickel": MaterialAttenuation(
        name="Nickel",
        density_g_cm3=8.9,
        mass_atten_coeff={
            100: 0.439,
            200: 0.147,
            500: 0.0830,
            1000: 0.0588,
            1500: 0.0477,
            2000: 0.0420,
        }
    ),
    "titanium": MaterialAttenuation(
        name="Titanium",
        density_g_cm3=4.5,
        mass_atten_coeff={
            100: 0.271,
            200: 0.133,
            500: 0.0847,
            1000: 0.0601,
            1500: 0.0492,
            2000: 0.0436,
        }
    ),
    "polyethylene": MaterialAttenuation(
        name="Polyethylene",
        density_g_cm3=0.93,
        mass_atten_coeff={
            100: 0.167,
            200: 0.123,
            500: 0.0869,
            1000: 0.0628,
            1500: 0.0517,
            2000: 0.0459,
        }
    ),
    "water": MaterialAttenuation(
        name="Water",
        density_g_cm3=1.0,
        mass_atten_coeff={
            100: 0.171,
            200: 0.137,
            500: 0.0969,
            1000: 0.0707,
            1500: 0.0575,
            2000: 0.0493,
        }
    ),
}


@dataclass
class SampleConfiguration:
    """
    Sample configuration for attenuation calculation.
    
    Attributes:
        geometry: Sample geometry type
        material: Sample material
        thickness_cm: Sample thickness (or diameter for cylinder/sphere)
        radius_cm: Sample radius (for disk/cylinder)
        height_cm: Sample height (for cylinder)
        container: Optional container material and thickness
    """
    
    geometry: SampleGeometry
    material: MaterialAttenuation
    thickness_cm: float
    radius_cm: Optional[float] = None
    height_cm: Optional[float] = None
    container_material: Optional[MaterialAttenuation] = None
    container_thickness_cm: float = 0.0


@dataclass
class AttenuationCorrectionFactor:
    """
    Attenuation correction factor for a gamma line.
    
    Attributes:
        energy_kev: Gamma energy
        C_att: Attenuation correction factor (>= 1)
        C_att_uncertainty: Uncertainty in correction
        sample_contribution: Contribution from sample
        container_contribution: Contribution from container
    """
    
    energy_kev: float
    C_att: float  # Multiply measured counts by this to get corrected
    C_att_uncertainty: float = 0.0
    sample_contribution: float = 1.0
    container_contribution: float = 1.0


def disk_self_attenuation_factor(
    mu: float,
    thickness_cm: float,
) -> float:
    """
    Calculate self-attenuation factor for a thin disk sample.
    
    For a disk source with uniform activity distribution:
    C_att = μt / (1 - exp(-μt))
    
    For thin samples (μt << 1): C_att → 1
    For thick samples (μt >> 1): C_att → μt
    
    Args:
        mu: Linear attenuation coefficient (cm⁻¹)
        thickness_cm: Sample thickness (cm)
        
    Returns:
        Attenuation correction factor
    """
    x = mu * thickness_cm
    
    if x < 1e-10:
        return 1.0  # No attenuation for very thin samples
    elif x > 50:
        return x  # Thick limit
    else:
        return x / (1.0 - math.exp(-x))


def cylinder_self_attenuation_factor(
    mu: float,
    diameter_cm: float,
    averaging_points: int = 20,
) -> float:
    """
    Calculate self-attenuation factor for a cylindrical sample.
    
    Uses numerical integration over the cylinder cross-section
    with uniform activity distribution.
    
    Args:
        mu: Linear attenuation coefficient (cm⁻¹)
        diameter_cm: Cylinder diameter (cm)
        averaging_points: Number of integration points
        
    Returns:
        Attenuation correction factor
    """
    R = diameter_cm / 2.0
    
    if mu * R < 1e-10:
        return 1.0
    
    # Integrate over radial positions
    total_weight = 0.0
    weighted_transmission = 0.0
    
    for i in range(averaging_points):
        r = R * (i + 0.5) / averaging_points  # Radial position
        weight = 2 * np.pi * r  # Cylindrical weighting
        
        # Average path length from this position
        # Simplified: use average chord length
        avg_path = np.pi * R / 4  # Average chord for random exit direction
        transmission = np.exp(-mu * avg_path)
        
        weighted_transmission += weight * transmission
        total_weight += weight
    
    if total_weight > 0:
        avg_transmission = weighted_transmission / total_weight
        if avg_transmission > 1e-10:
            return 1.0 / avg_transmission
    
    return 1.0


def sphere_self_attenuation_factor(
    mu: float,
    diameter_cm: float,
) -> float:
    """
    Calculate self-attenuation factor for a spherical sample.
    
    Uses analytical result for uniform sphere:
    C_att = 3/(μR)² * [1 - 2/(μR) + 2/(μR)² * (1 - exp(-2μR))]
    
    Args:
        mu: Linear attenuation coefficient (cm⁻¹)
        diameter_cm: Sphere diameter (cm)
        
    Returns:
        Attenuation correction factor
    """
    R = diameter_cm / 2.0
    x = mu * R
    
    if x < 1e-10:
        return 1.0
    elif x > 50:
        return 3.0 / x  # Thick limit approximation
    else:
        # Full analytical formula
        exp_2x = math.exp(-2 * x)
        term1 = 1.0
        term2 = -2.0 / x
        term3 = (2.0 / x**2) * (1.0 - exp_2x)
        transmission = (3.0 / (2 * x)) * (term1 + term2 + term3)
        if transmission > 1e-10:
            return 1.0 / transmission
        return 100.0  # Very thick


def calculate_sample_attenuation(
    config: SampleConfiguration,
    energy_kev: float,
) -> float:
    """
    Calculate sample self-attenuation factor.
    
    Args:
        config: Sample configuration
        energy_kev: Gamma energy
        
    Returns:
        Sample attenuation correction factor
    """
    mu = config.material.get_mu(energy_kev)
    
    if config.geometry == SampleGeometry.POINT:
        return 1.0
    
    elif config.geometry == SampleGeometry.DISK:
        return disk_self_attenuation_factor(mu, config.thickness_cm)
    
    elif config.geometry == SampleGeometry.CYLINDER:
        return cylinder_self_attenuation_factor(mu, config.thickness_cm)
    
    elif config.geometry == SampleGeometry.SPHERE:
        return sphere_self_attenuation_factor(mu, config.thickness_cm)
    
    else:
        # Default to disk approximation
        return disk_self_attenuation_factor(mu, config.thickness_cm)


def calculate_container_attenuation(
    config: SampleConfiguration,
    energy_kev: float,
) -> float:
    """
    Calculate container attenuation factor.
    
    Assumes gamma rays pass through container wall once on average.
    
    Args:
        config: Sample configuration
        energy_kev: Gamma energy
        
    Returns:
        Container attenuation correction factor
    """
    if config.container_material is None or config.container_thickness_cm <= 0:
        return 1.0
    
    mu = config.container_material.get_mu(energy_kev)
    transmission = math.exp(-mu * config.container_thickness_cm)
    
    if transmission > 1e-10:
        return 1.0 / transmission
    return 100.0  # Very thick container


def calculate_attenuation_correction(
    config: SampleConfiguration,
    energy_kev: float,
    include_container: bool = True,
) -> AttenuationCorrectionFactor:
    """
    Calculate total attenuation correction factor.
    
    The correction factor C_att is applied to measured counts:
    counts_corrected = counts_measured * C_att
    
    Args:
        config: Sample configuration
        energy_kev: Gamma energy
        include_container: Whether to include container attenuation
        
    Returns:
        AttenuationCorrectionFactor with all components
    """
    C_sample = calculate_sample_attenuation(config, energy_kev)
    
    if include_container:
        C_container = calculate_container_attenuation(config, energy_kev)
    else:
        C_container = 1.0
    
    C_total = C_sample * C_container
    
    # Estimate uncertainty (assume 5% in attenuation coefficients)
    rel_unc = 0.05 * (C_total - 1.0) if C_total > 1.0 else 0.0
    
    return AttenuationCorrectionFactor(
        energy_kev=energy_kev,
        C_att=C_total,
        C_att_uncertainty=rel_unc,
        sample_contribution=C_sample,
        container_contribution=C_container,
    )


def calculate_attenuation_corrections(
    config: SampleConfiguration,
    energies_kev: List[float],
) -> List[AttenuationCorrectionFactor]:
    """
    Calculate attenuation corrections for multiple gamma lines.
    
    Args:
        config: Sample configuration
        energies_kev: List of gamma energies
        
    Returns:
        List of correction factors
    """
    return [
        calculate_attenuation_correction(config, E)
        for E in energies_kev
    ]


@dataclass
class AttenuationCorrectionLibrary:
    """
    Attenuation correction library for a sample.
    
    Contains energy-dependent attenuation corrections that can
    be applied during activity inference.
    """
    
    sample_id: str
    config: SampleConfiguration
    factors: List[AttenuationCorrectionFactor]
    
    def get_correction(self, energy_kev: float) -> float:
        """Get interpolated correction factor at given energy."""
        if not self.factors:
            return 1.0
        
        # Find bracketing energies
        energies = [f.energy_kev for f in self.factors]
        corrections = [f.C_att for f in self.factors]
        
        if energy_kev <= energies[0]:
            return corrections[0]
        elif energy_kev >= energies[-1]:
            return corrections[-1]
        else:
            # Log-log interpolation
            log_E = np.log(energies)
            log_C = np.log(corrections)
            log_E_target = np.log(energy_kev)
            log_C_interp = np.interp(log_E_target, log_E, log_C)
            return np.exp(log_C_interp)
    
    def to_dict(self) -> dict:
        """Export to dictionary for serialization."""
        return {
            "schema": "fluxforge.attenuation_library.v1",
            "sample_id": self.sample_id,
            "config": {
                "geometry": self.config.geometry.value,
                "material": self.config.material.name,
                "thickness_cm": self.config.thickness_cm,
                "has_container": self.config.container_material is not None,
            },
            "n_energies": len(self.factors),
            "factors": [
                {
                    "energy_kev": f.energy_kev,
                    "C_att": f.C_att,
                    "C_att_uncertainty": f.C_att_uncertainty,
                    "sample_contribution": f.sample_contribution,
                    "container_contribution": f.container_contribution,
                }
                for f in self.factors
            ],
        }


def create_attenuation_library(
    sample_id: str,
    config: SampleConfiguration,
    energy_range_kev: Tuple[float, float] = (50, 3000),
    n_points: int = 50,
) -> AttenuationCorrectionLibrary:
    """
    Create attenuation correction library for a sample.
    
    Generates correction factors on a logarithmic energy grid.
    
    Args:
        sample_id: Sample identifier
        config: Sample configuration
        energy_range_kev: Energy range (min, max)
        n_points: Number of energy points
        
    Returns:
        AttenuationCorrectionLibrary
    """
    energies = np.logspace(
        np.log10(energy_range_kev[0]),
        np.log10(energy_range_kev[1]),
        n_points
    )
    
    factors = calculate_attenuation_corrections(config, list(energies))
    
    return AttenuationCorrectionLibrary(
        sample_id=sample_id,
        config=config,
        factors=factors,
    )


def get_standard_material(name: str) -> MaterialAttenuation:
    """Get standard material attenuation data."""
    key = name.lower()
    if key in STANDARD_MATERIALS:
        return STANDARD_MATERIALS[key]
    raise ValueError(f"Unknown material: {name}. Available: {list(STANDARD_MATERIALS.keys())}")
