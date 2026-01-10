"""
Neutron self-shielding corrections for activation monitors.

This module provides SHIELD-style neutron self-shielding calculations
for foils and wires, including:
- Geometry-aware corrections (slab, cylinder, sphere)
- Fine energy grid internal resolution for resonance effects
- Self-shielding library artifact generation
- Isotropic and beam flux type support

Reference: STAYSL PNNL, SHIELD methodology
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Dict, Tuple

import numpy as np
from scipy import integrate


class Geometry(Enum):
    """Monitor geometry types."""
    
    SLAB = "slab"  # Flat foil
    CYLINDER = "cylinder"  # Wire
    SPHERE = "sphere"  # Spherical sample


class FluxType(Enum):
    """Incident flux angular distribution."""
    
    ISOTROPIC = "isotropic"  # Uniform in all directions
    BEAM = "beam"  # Collimated beam (perpendicular incidence)


@dataclass
class MaterialProperties:
    """
    Material properties for self-shielding calculation.
    
    Attributes:
        name: Material identifier
        density_g_cm3: Mass density
        atomic_mass_amu: Atomic mass of target isotope
        number_density_per_cm3: Atom number density (computed if not provided)
        isotopic_abundance: Fraction of target isotope in element
    """
    
    name: str
    density_g_cm3: float
    atomic_mass_amu: float
    isotopic_abundance: float = 1.0
    number_density_per_cm3: Optional[float] = None
    
    def __post_init__(self):
        if self.number_density_per_cm3 is None:
            # N = ρ * N_A / A
            N_A = 6.02214076e23  # Avogadro's number
            self.number_density_per_cm3 = (
                self.density_g_cm3 * N_A * self.isotopic_abundance 
                / self.atomic_mass_amu
            )


@dataclass
class MonitorGeometry:
    """
    Geometry specification for a monitor (foil/wire).
    
    Attributes:
        geometry_type: Slab, cylinder, or sphere
        thickness_cm: Thickness (slab) or diameter (cylinder/sphere)
        width_cm: Width for rectangular foil (optional)
        length_cm: Length for foil/wire (optional)
    """
    
    geometry_type: Geometry
    thickness_cm: float
    width_cm: Optional[float] = None
    length_cm: Optional[float] = None
    
    @property
    def characteristic_length_cm(self) -> float:
        """Return the characteristic length for self-shielding calculation."""
        if self.geometry_type == Geometry.SLAB:
            return self.thickness_cm
        elif self.geometry_type == Geometry.CYLINDER:
            return self.thickness_cm  # diameter
        else:  # SPHERE
            return self.thickness_cm  # diameter


@dataclass
class SelfShieldingFactor:
    """
    Self-shielding correction factor for a single energy group.
    
    Attributes:
        energy_low_ev: Lower energy bound
        energy_high_ev: Upper energy bound
        G_ss: Self-shielding factor (0 < G_ss <= 1)
        G_ss_uncertainty: Uncertainty in G_ss
    """
    
    energy_low_ev: float
    energy_high_ev: float
    G_ss: float
    G_ss_uncertainty: float = 0.0


def slab_self_shielding_isotropic(
    sigma_t_cm2: float,
    N: float,
    thickness_cm: float,
) -> float:
    """
    Calculate self-shielding factor for a slab in isotropic flux.
    
    G_ss = (1 - exp(-Σ_t * t)) / (Σ_t * t) * correction
    
    For thin samples: G_ss → 1
    For thick samples: G_ss → 0
    
    Args:
        sigma_t_cm2: Total microscopic cross section (cm²)
        N: Number density (atoms/cm³)
        thickness_cm: Sample thickness (cm)
        
    Returns:
        Self-shielding factor G_ss (0 < G_ss <= 1)
    """
    Sigma_t = sigma_t_cm2 * N  # Macroscopic cross section
    x = Sigma_t * thickness_cm
    
    if x < 1e-10:
        return 1.0  # No self-shielding for thin samples
    elif x > 50:
        return 1.0 / x  # Thick limit
    else:
        # First-order approximation for isotropic flux
        # More accurate: use Sievert's integral for slab geometry
        G0 = (1.0 - math.exp(-x)) / x
        
        # For isotropic flux, apply correction using E_3 exponential integral
        # G_ss ≈ 2 * E_3(x)
        # Approximate E_3(x) using series expansion
        if x < 1:
            # Small x expansion
            correction = 0.5 * (1 - x/3 + x**2/12 - x**3/60)
        else:
            # Large x approximation
            correction = math.exp(-x) / (2 * x)
        
        return 2.0 * correction


def slab_self_shielding_beam(
    sigma_t_cm2: float,
    N: float,
    thickness_cm: float,
) -> float:
    """
    Calculate self-shielding factor for a slab in beam (perpendicular) flux.
    
    G_ss = (1 - exp(-Σ_t * t)) / (Σ_t * t)
    
    Args:
        sigma_t_cm2: Total microscopic cross section (cm²)
        N: Number density (atoms/cm³)
        thickness_cm: Sample thickness (cm)
        
    Returns:
        Self-shielding factor G_ss (0 < G_ss <= 1)
    """
    Sigma_t = sigma_t_cm2 * N
    x = Sigma_t * thickness_cm
    
    if x < 1e-10:
        return 1.0
    elif x > 50:
        return 1.0 / x
    else:
        return (1.0 - math.exp(-x)) / x


def cylinder_self_shielding_isotropic(
    sigma_t_cm2: float,
    N: float,
    diameter_cm: float,
) -> float:
    """
    Calculate self-shielding factor for a cylinder (wire) in isotropic flux.
    
    Uses approximation for long cylinder with perpendicular flux integration.
    
    Args:
        sigma_t_cm2: Total microscopic cross section (cm²)
        N: Number density (atoms/cm³)
        diameter_cm: Cylinder diameter (cm)
        
    Returns:
        Self-shielding factor G_ss (0 < G_ss <= 1)
    """
    Sigma_t = sigma_t_cm2 * N
    radius_cm = diameter_cm / 2.0
    x = Sigma_t * diameter_cm  # Optical thickness across diameter
    
    if x < 1e-10:
        return 1.0
    elif x > 50:
        # Thick limit approximation
        return 2.0 / (math.pi * x)
    else:
        # Use chord-averaging approximation
        # G_ss ≈ I_0(x) * exp(-x) for small x, with corrections
        # More accurate: numerical integration over chord lengths
        
        # Simplified Wigner rational approximation
        a = 1.0 / (1.0 + 0.4 * x)
        return a


def sphere_self_shielding_isotropic(
    sigma_t_cm2: float,
    N: float,
    diameter_cm: float,
) -> float:
    """
    Calculate self-shielding factor for a sphere in isotropic flux.
    
    Args:
        sigma_t_cm2: Total microscopic cross section (cm²)
        N: Number density (atoms/cm³)
        diameter_cm: Sphere diameter (cm)
        
    Returns:
        Self-shielding factor G_ss (0 < G_ss <= 1)
    """
    Sigma_t = sigma_t_cm2 * N
    x = Sigma_t * diameter_cm
    
    if x < 1e-10:
        return 1.0
    elif x > 50:
        return 3.0 / (2.0 * x)
    else:
        # Exact result for sphere
        # G_ss = 3 * [1/(2x²) - 1/(2x³) + exp(-2x)/(2x³) + exp(-2x)/x²]
        if x < 0.1:
            # Small x expansion
            return 1.0 - 2.0*x/3.0 + x**2/3.0 - 2.0*x**3/15.0
        else:
            exp_2x = math.exp(-2.0 * x)
            term1 = 1.0 / (2.0 * x**2)
            term2 = -1.0 / (2.0 * x**3)
            term3 = exp_2x / (2.0 * x**3)
            term4 = exp_2x / x**2
            return 3.0 * (term1 + term2 + term3 + term4)


def calculate_self_shielding_factor(
    sigma_t_cm2: float,
    material: MaterialProperties,
    geometry: MonitorGeometry,
    flux_type: FluxType = FluxType.ISOTROPIC,
) -> float:
    """
    Calculate self-shielding factor for given conditions.
    
    Args:
        sigma_t_cm2: Total microscopic cross section (cm²)
        material: Material properties
        geometry: Monitor geometry
        flux_type: Incident flux angular distribution
        
    Returns:
        Self-shielding factor G_ss (0 < G_ss <= 1)
    """
    N = material.number_density_per_cm3
    dim = geometry.characteristic_length_cm
    
    if geometry.geometry_type == Geometry.SLAB:
        if flux_type == FluxType.ISOTROPIC:
            return slab_self_shielding_isotropic(sigma_t_cm2, N, dim)
        else:
            return slab_self_shielding_beam(sigma_t_cm2, N, dim)
    
    elif geometry.geometry_type == Geometry.CYLINDER:
        # For cylinder, isotropic approximation is typically used
        return cylinder_self_shielding_isotropic(sigma_t_cm2, N, dim)
    
    else:  # SPHERE
        return sphere_self_shielding_isotropic(sigma_t_cm2, N, dim)


def calculate_group_self_shielding(
    sigma_t_fine: np.ndarray,
    energy_fine_ev: np.ndarray,
    flux_fine: np.ndarray,
    energy_group_bounds_ev: np.ndarray,
    material: MaterialProperties,
    geometry: MonitorGeometry,
    flux_type: FluxType = FluxType.ISOTROPIC,
) -> List[SelfShieldingFactor]:
    """
    Calculate self-shielding factors on group structure using fine grid.
    
    This implements the SHIELD-style methodology:
    1. Compute G_ss on a fine energy grid (resolving resonances)
    2. Flux-weight G_ss within each coarse group
    3. Return collapsed group-averaged G_ss values
    
    Args:
        sigma_t_fine: Total cross section on fine grid (barns)
        energy_fine_ev: Fine energy grid (eV)
        flux_fine: Flux on fine grid (per lethargy or per energy)
        energy_group_bounds_ev: Coarse group boundaries (eV)
        material: Material properties
        geometry: Monitor geometry
        flux_type: Flux angular distribution
        
    Returns:
        List of SelfShieldingFactor for each coarse group
    """
    # Convert barns to cm²
    sigma_t_cm2 = sigma_t_fine * 1e-24
    
    # Calculate G_ss on fine grid
    G_ss_fine = np.array([
        calculate_self_shielding_factor(s, material, geometry, flux_type)
        for s in sigma_t_cm2
    ])
    
    # Collapse to group structure
    n_groups = len(energy_group_bounds_ev) - 1
    results = []
    
    for g in range(n_groups):
        E_low = energy_group_bounds_ev[g]
        E_high = energy_group_bounds_ev[g + 1]
        
        # Find fine grid points in this group
        mask = (energy_fine_ev >= E_low) & (energy_fine_ev < E_high)
        
        if np.any(mask):
            # Flux-weighted average
            weights = flux_fine[mask]
            if np.sum(weights) > 0:
                G_ss_g = np.average(G_ss_fine[mask], weights=weights)
            else:
                G_ss_g = np.mean(G_ss_fine[mask])
            
            # Estimate uncertainty from variance within group
            if np.sum(mask) > 1:
                variance = np.var(G_ss_fine[mask])
                G_ss_unc = math.sqrt(variance / np.sum(mask))
            else:
                G_ss_unc = 0.1 * G_ss_g  # Default 10% uncertainty
        else:
            # No fine grid points in group, use midpoint
            E_mid = math.sqrt(E_low * E_high)
            # Interpolate cross section
            idx = np.searchsorted(energy_fine_ev, E_mid)
            if idx < len(sigma_t_cm2):
                G_ss_g = calculate_self_shielding_factor(
                    sigma_t_cm2[idx], material, geometry, flux_type
                )
            else:
                G_ss_g = 1.0
            G_ss_unc = 0.2 * G_ss_g  # Higher uncertainty for interpolated
        
        results.append(SelfShieldingFactor(
            energy_low_ev=E_low,
            energy_high_ev=E_high,
            G_ss=G_ss_g,
            G_ss_uncertainty=G_ss_unc,
        ))
    
    return results


@dataclass
class SelfShieldingLibrary:
    """
    Self-shielding library artifact (analogous to STAYSL sshldlib.dat).
    
    Contains energy-dependent self-shielding factors for each
    reaction/monitor that can be applied during response matrix construction.
    """
    
    reaction_id: str
    material: MaterialProperties
    geometry: MonitorGeometry
    flux_type: FluxType
    factors: List[SelfShieldingFactor] = field(default_factory=list)
    
    def get_factor(self, energy_ev: float) -> float:
        """Get self-shielding factor at given energy."""
        for f in self.factors:
            if f.energy_low_ev <= energy_ev < f.energy_high_ev:
                return f.G_ss
        return 1.0  # Default if outside range
    
    def get_group_factors(self) -> np.ndarray:
        """Get array of group self-shielding factors."""
        return np.array([f.G_ss for f in self.factors])
    
    def get_group_uncertainties(self) -> np.ndarray:
        """Get array of group self-shielding uncertainties."""
        return np.array([f.G_ss_uncertainty for f in self.factors])
    
    def to_dict(self) -> dict:
        """Export to dictionary for serialization."""
        return {
            "schema": "fluxforge.self_shielding_library.v1",
            "reaction_id": self.reaction_id,
            "material": {
                "name": self.material.name,
                "density_g_cm3": self.material.density_g_cm3,
                "atomic_mass_amu": self.material.atomic_mass_amu,
                "isotopic_abundance": self.material.isotopic_abundance,
                "number_density_per_cm3": self.material.number_density_per_cm3,
            },
            "geometry": {
                "type": self.geometry.geometry_type.value,
                "thickness_cm": self.geometry.thickness_cm,
                "width_cm": self.geometry.width_cm,
                "length_cm": self.geometry.length_cm,
            },
            "flux_type": self.flux_type.value,
            "n_groups": len(self.factors),
            "factors": [
                {
                    "energy_low_ev": f.energy_low_ev,
                    "energy_high_ev": f.energy_high_ev,
                    "G_ss": f.G_ss,
                    "G_ss_uncertainty": f.G_ss_uncertainty,
                }
                for f in self.factors
            ],
        }


def create_self_shielding_library(
    reaction_id: str,
    sigma_t_fine: np.ndarray,
    energy_fine_ev: np.ndarray,
    flux_fine: np.ndarray,
    energy_group_bounds_ev: np.ndarray,
    material: MaterialProperties,
    geometry: MonitorGeometry,
    flux_type: FluxType = FluxType.ISOTROPIC,
) -> SelfShieldingLibrary:
    """
    Create a self-shielding library for a reaction.
    
    Args:
        reaction_id: Unique reaction identifier
        sigma_t_fine: Total cross section on fine grid (barns)
        energy_fine_ev: Fine energy grid (eV)
        flux_fine: Flux on fine grid
        energy_group_bounds_ev: Coarse group boundaries
        material: Material properties
        geometry: Monitor geometry
        flux_type: Flux angular distribution
        
    Returns:
        SelfShieldingLibrary artifact
    """
    factors = calculate_group_self_shielding(
        sigma_t_fine, energy_fine_ev, flux_fine,
        energy_group_bounds_ev, material, geometry, flux_type
    )
    
    return SelfShieldingLibrary(
        reaction_id=reaction_id,
        material=material,
        geometry=geometry,
        flux_type=flux_type,
        factors=factors,
    )


# Common materials for flux monitors
STANDARD_MATERIALS = {
    "Au": MaterialProperties("Gold", 19.3, 196.97),
    "Co": MaterialProperties("Cobalt", 8.9, 58.93),
    "Ni": MaterialProperties("Nickel", 8.9, 58.69),
    "Fe": MaterialProperties("Iron", 7.87, 55.85),
    "Ti": MaterialProperties("Titanium", 4.5, 47.87),
    "Sc": MaterialProperties("Scandium", 2.99, 44.96),
    "In": MaterialProperties("Indium", 7.31, 114.82),
    "Cu": MaterialProperties("Copper", 8.96, 63.55),
    "Mn": MaterialProperties("Manganese", 7.44, 54.94),
    "Al": MaterialProperties("Aluminum", 2.7, 26.98),
    "Zr": MaterialProperties("Zirconium", 6.5, 91.22),
    "Nb": MaterialProperties("Niobium", 8.57, 92.91),
}


def get_standard_material(element: str) -> MaterialProperties:
    """Get standard material properties for common flux monitor elements."""
    if element in STANDARD_MATERIALS:
        return STANDARD_MATERIALS[element]
    raise ValueError(f"Unknown element: {element}. Available: {list(STANDARD_MATERIALS.keys())}")
