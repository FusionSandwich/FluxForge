"""Sample and geometry data models.

These are lightweight containers intended to make corrections and provenance
more explicit and consistent across the activation workflow.

Follows STAYSL PNNL conventions for flux wire/foil samples including:
- Material composition with isotopic abundances
- Cover specifications (Cd, Gd, B, Au)
- Sample geometry for self-shielding calculations
- Container/capsule specifications for gamma attenuation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple


# Geometry type literals matching STAYSL conventions
GeometryType = Literal["wire", "foil", "cylinder", "slab", "sphere", "disk", "unknown"]


class FluxEnvironment(Enum):
    """Neutron flux environment type (STAYSL convention)."""
    
    ISOTROPIC = "isotropic"  # Reactor core, isotropic flux
    BEAM = "beam"            # Beam flux normal to sample surface
    MIXED = "mixed"          # Mixed environment


@dataclass(frozen=True)
class MaterialComponent:
    """Single component of sample material."""
    
    nuclide: str
    atom_fraction: float
    mass_fraction: Optional[float] = None
    
    @classmethod
    def from_element(cls, element: str, abundance: float = 1.0) -> "MaterialComponent":
        """Create from natural element with abundance."""
        return cls(nuclide=element, atom_fraction=abundance)


@dataclass(frozen=True)
class IsotopicComposition:
    """Full isotopic composition for an element."""
    
    element: str
    isotopes: Tuple[Tuple[int, float], ...]  # (mass_number, atom_fraction) pairs
    
    def get_fraction(self, mass_number: int) -> float:
        """Get atom fraction for specific isotope."""
        for a, frac in self.isotopes:
            if a == mass_number:
                return frac
        return 0.0
    
    @classmethod
    def natural(cls, element: str) -> "IsotopicComposition":
        """Create natural isotopic composition from element."""
        # Common natural abundances for activation monitors
        NATURAL_ABUNDANCES = {
            "Au": ((197, 1.0),),
            "Co": ((59, 1.0),),
            "In": ((113, 0.0429), (115, 0.9571)),
            "Fe": ((54, 0.0585), (56, 0.9175), (57, 0.0212), (58, 0.0028)),
            "Ni": ((58, 0.6808), (60, 0.2622), (61, 0.0114), (62, 0.0363), (64, 0.0093)),
            "Ti": ((46, 0.0825), (47, 0.0744), (48, 0.7372), (49, 0.0541), (50, 0.0518)),
            "Sc": ((45, 1.0),),
            "Cu": ((63, 0.6915), (65, 0.3085)),
            "Mn": ((55, 1.0),),
            "Zr": ((90, 0.5145), (91, 0.1122), (92, 0.1715), (94, 0.1738), (96, 0.0280)),
            "Na": ((23, 1.0),),
            "Al": ((27, 1.0),),
            "Mg": ((24, 0.7899), (25, 0.1000), (26, 0.1101)),
        }
        isotopes = NATURAL_ABUNDANCES.get(element, ((0, 1.0),))
        return cls(element=element, isotopes=isotopes)


@dataclass(frozen=True)
class Cover:
    """
    Cover/shield specification following STAYSL conventions.
    
    Attributes:
        material: Cover material (Cd, Gd, B, Au, etc.)
        thickness_cm: Thickness in cm
        thickness_mil: Thickness in mils (1 mil = 0.001 inch = 0.00254 cm)
        density_g_cm3: Material density (uses standard if not provided)
    """
    
    material: str
    thickness_cm: float
    density_g_cm3: Optional[float] = None
    
    @property
    def thickness_mil(self) -> float:
        """Thickness in mils (STAYSL convention)."""
        return self.thickness_cm / 0.00254
    
    @classmethod
    def from_mil(cls, material: str, thickness_mil: float, density: Optional[float] = None) -> "Cover":
        """Create cover from mil thickness (STAYSL convention)."""
        thickness_cm = thickness_mil * 0.00254
        return cls(material=material, thickness_cm=thickness_cm, density_g_cm3=density)
    
    @classmethod
    def standard_cd(cls, thickness_mil: float = 20.0) -> "Cover":
        """Standard Cd cover (20 mil default)."""
        return cls.from_mil("Cd", thickness_mil, density=8.65)
    
    @classmethod
    def standard_gd(cls, thickness_mil: float = 5.0) -> "Cover":
        """Standard Gd cover."""
        return cls.from_mil("Gd", thickness_mil, density=7.90)


@dataclass(frozen=True)
class Container:
    """Sample container/capsule specification for gamma attenuation."""
    
    material: str
    thickness_cm: float
    density_g_cm3: Optional[float] = None
    inner_radius_cm: Optional[float] = None
    
    # Common container materials and densities
    MATERIAL_DENSITIES = {
        "polyethylene": 0.95,
        "polypropylene": 0.90,
        "quartz": 2.65,
        "aluminum": 2.70,
        "stainless_steel": 8.0,
    }
    
    def get_density(self) -> float:
        """Get density, using default for known materials."""
        if self.density_g_cm3 is not None:
            return self.density_g_cm3
        return self.MATERIAL_DENSITIES.get(self.material.lower(), 1.0)


@dataclass
class SampleDimensions:
    """
    Detailed sample dimensions for various geometries.
    
    For self-shielding calculations following STAYSL SHIELD conventions.
    """
    
    # Wire dimensions
    diameter_cm: Optional[float] = None
    length_cm: Optional[float] = None
    
    # Foil dimensions
    thickness_cm: Optional[float] = None
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    
    # Disk/cylinder
    radius_cm: Optional[float] = None
    
    # Sphere
    sphere_radius_cm: Optional[float] = None
    
    @property
    def diameter_mil(self) -> Optional[float]:
        """Wire diameter in mils."""
        if self.diameter_cm is not None:
            return self.diameter_cm / 0.00254
        return None
    
    @property
    def thickness_mil(self) -> Optional[float]:
        """Foil thickness in mils."""
        if self.thickness_cm is not None:
            return self.thickness_cm / 0.00254
        return None
    
    def mean_chord(self, geometry: GeometryType) -> float:
        """
        Calculate mean chord length for self-shielding (STAYSL convention).
        
        For wire: diameter
        For foil: 2 * thickness (for perpendicular beam) or thickness (isotropic average)
        """
        if geometry == "wire" and self.diameter_cm is not None:
            return self.diameter_cm
        elif geometry == "foil" and self.thickness_cm is not None:
            return self.thickness_cm
        elif geometry == "sphere" and self.sphere_radius_cm is not None:
            return 4.0 * self.sphere_radius_cm / 3.0  # Mean chord = 4R/3
        elif geometry == "cylinder" and self.diameter_cm is not None:
            return self.diameter_cm
        return 0.0
    
    @classmethod
    def wire(cls, diameter_cm: float, length_cm: Optional[float] = None) -> "SampleDimensions":
        """Create wire dimensions."""
        return cls(diameter_cm=diameter_cm, length_cm=length_cm)
    
    @classmethod
    def wire_mil(cls, diameter_mil: float, length_cm: Optional[float] = None) -> "SampleDimensions":
        """Create wire dimensions from mil diameter."""
        diameter_cm = diameter_mil * 0.00254
        return cls(diameter_cm=diameter_cm, length_cm=length_cm)
    
    @classmethod
    def foil(cls, thickness_cm: float, width_cm: float, height_cm: float) -> "SampleDimensions":
        """Create foil dimensions."""
        return cls(thickness_cm=thickness_cm, width_cm=width_cm, height_cm=height_cm)
    
    @classmethod
    def foil_mil(cls, thickness_mil: float, width_cm: float, height_cm: float) -> "SampleDimensions":
        """Create foil dimensions from mil thickness."""
        thickness_cm = thickness_mil * 0.00254
        return cls(thickness_cm=thickness_cm, width_cm=width_cm, height_cm=height_cm)


@dataclass
class Sample:
    """
    Represents a physical sample (foil/wire/etc.) and its immediate coverings.
    
    Complete sample specification for activation analysis following STAYSL conventions.
    
    Attributes:
        sample_id: Unique identifier
        geometry: Sample geometry type (wire, foil, cylinder, slab, sphere)
        mass_g: Sample mass in grams
        density_g_cm3: Material density
        dimensions: Detailed dimensions for self-shielding
        composition: Material composition (element/isotope fractions)
        covers: List of covers (Cd, Gd, B, Au)
        container: Optional container/capsule
        flux_environment: Flux type (isotropic/beam)
        temperature_K: Sample temperature during irradiation
        purity: Material purity fraction (0-1)
        metadata: Additional metadata
    """

    sample_id: str
    geometry: GeometryType = "unknown"
    mass_g: Optional[float] = None
    density_g_cm3: Optional[float] = None
    dimensions_cm: Dict[str, float] = field(default_factory=dict)
    dimensions: Optional[SampleDimensions] = None
    composition: List[MaterialComponent] = field(default_factory=list)
    isotopic_composition: Optional[IsotopicComposition] = None
    covers: List[Cover] = field(default_factory=list)
    container: Optional[Container] = None
    flux_environment: FluxEnvironment = FluxEnvironment.ISOTROPIC
    temperature_K: float = 300.0
    purity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_cover(self, material: str) -> bool:
        """Check if sample has a specific cover type."""
        return any(c.material.lower() == material.lower() for c in self.covers)
    
    def get_cover(self, material: str) -> Optional[Cover]:
        """Get cover by material type."""
        for cover in self.covers:
            if cover.material.lower() == material.lower():
                return cover
        return None
    
    @property
    def is_cd_covered(self) -> bool:
        """Check if sample has Cd cover."""
        return self.has_cover("Cd")
    
    @property
    def is_bare(self) -> bool:
        """Check if sample is bare (no covers)."""
        return len(self.covers) == 0
    
    def mean_chord_cm(self) -> float:
        """Get mean chord for self-shielding calculations."""
        if self.dimensions is not None:
            return self.dimensions.mean_chord(self.geometry)
        # Fallback to dimensions_cm dict
        if self.geometry == "wire" and "diameter" in self.dimensions_cm:
            return self.dimensions_cm["diameter"]
        if self.geometry == "foil" and "thickness" in self.dimensions_cm:
            return self.dimensions_cm["thickness"]
        return 0.0
    
    def atom_density(self, atomic_weight: float) -> float:
        """
        Calculate atom density for self-shielding (atoms/cm³).
        
        N = (N_A * density * purity) / A
        """
        if self.density_g_cm3 is None:
            return 0.0
        N_A = 6.02214076e23  # Avogadro's number
        return N_A * self.density_g_cm3 * self.purity / atomic_weight
    
    @classmethod
    def flux_wire(
        cls,
        sample_id: str,
        element: str,
        diameter_mil: float,
        mass_g: float,
        density_g_cm3: float,
        cd_covered: bool = False,
        cd_thickness_mil: float = 20.0,
    ) -> "Sample":
        """
        Create a standard flux wire sample.
        
        Parameters
        ----------
        sample_id : str
            Sample identifier
        element : str
            Primary element (Au, Co, In, Fe, Ni, Ti, Sc, Cu, Mn)
        diameter_mil : float
            Wire diameter in mils
        mass_g : float
            Sample mass in grams
        density_g_cm3 : float
            Material density
        cd_covered : bool
            Whether sample has Cd cover
        cd_thickness_mil : float
            Cd cover thickness in mils (default 20)
        """
        dimensions = SampleDimensions.wire_mil(diameter_mil)
        composition = [MaterialComponent.from_element(element)]
        isotopic = IsotopicComposition.natural(element)
        covers = [Cover.standard_cd(cd_thickness_mil)] if cd_covered else []
        
        return cls(
            sample_id=sample_id,
            geometry="wire",
            mass_g=mass_g,
            density_g_cm3=density_g_cm3,
            dimensions=dimensions,
            composition=composition,
            isotopic_composition=isotopic,
            covers=covers,
            flux_environment=FluxEnvironment.ISOTROPIC,
        )
    
    @classmethod  
    def activation_foil(
        cls,
        sample_id: str,
        element: str,
        thickness_mil: float,
        width_cm: float,
        height_cm: float,
        mass_g: float,
        density_g_cm3: float,
        cd_covered: bool = False,
    ) -> "Sample":
        """Create a standard activation foil sample."""
        dimensions = SampleDimensions.foil_mil(thickness_mil, width_cm, height_cm)
        composition = [MaterialComponent.from_element(element)]
        isotopic = IsotopicComposition.natural(element)
        covers = [Cover.standard_cd()] if cd_covered else []
        
        return cls(
            sample_id=sample_id,
            geometry="foil",
            mass_g=mass_g,
            density_g_cm3=density_g_cm3,
            dimensions=dimensions,
            composition=composition,
            isotopic_composition=isotopic,
            covers=covers,
            flux_environment=FluxEnvironment.ISOTROPIC,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "geometry": self.geometry,
            "mass_g": self.mass_g,
            "density_g_cm3": self.density_g_cm3,
            "dimensions_cm": dict(self.dimensions_cm),
            "dimensions": {
                "diameter_cm": self.dimensions.diameter_cm if self.dimensions else None,
                "thickness_cm": self.dimensions.thickness_cm if self.dimensions else None,
                "length_cm": self.dimensions.length_cm if self.dimensions else None,
            } if self.dimensions else None,
            "composition": [
                {"nuclide": c.nuclide, "atom_fraction": c.atom_fraction} for c in self.composition
            ],
            "covers": [
                {
                    "material": cover.material,
                    "thickness_cm": cover.thickness_cm,
                    "density_g_cm3": cover.density_g_cm3,
                }
                for cover in self.covers
            ],
            "container": None
            if self.container is None
            else {
                "material": self.container.material,
                "thickness_cm": self.container.thickness_cm,
                "density_g_cm3": self.container.density_g_cm3,
            },
            "flux_environment": self.flux_environment.value,
            "temperature_K": self.temperature_K,
            "purity": self.purity,
            "metadata": dict(self.metadata),
        }


# ==============================================================================
# Standard Wire/Foil Materials (from STAYSL Table 2)
# ==============================================================================

STANDARD_WIRE_MATERIALS = {
    "Au": {"density": 19.3, "atomic_weight": 196.967, "sigma_th": 98.65, "RI": 1550},
    "Co": {"density": 8.9, "atomic_weight": 58.933, "sigma_th": 37.18, "RI": 74.2},
    "In": {"density": 7.31, "atomic_weight": 114.818, "sigma_th": 193.8, "RI": 2640},
    "Fe": {"density": 7.874, "atomic_weight": 55.845, "sigma_th": 2.56, "RI": 1.26},
    "Ni": {"density": 8.908, "atomic_weight": 58.693, "sigma_th": 4.49, "RI": 2.1},
    "Ti": {"density": 4.506, "atomic_weight": 47.867, "sigma_th": 6.09, "RI": 3.1},
    "Sc": {"density": 2.989, "atomic_weight": 44.956, "sigma_th": 27.2, "RI": 12.1},
    "Cu": {"density": 8.96, "atomic_weight": 63.546, "sigma_th": 3.78, "RI": 4.97},
    "Mn": {"density": 7.21, "atomic_weight": 54.938, "sigma_th": 13.3, "RI": 14.0},
    "Na": {"density": 0.968, "atomic_weight": 22.990, "sigma_th": 0.53, "RI": 0.31},
    "Al": {"density": 2.70, "atomic_weight": 26.982, "sigma_th": 0.231, "RI": 0.17},
    "Mg": {"density": 1.738, "atomic_weight": 24.305, "sigma_th": 0.066, "RI": 0.038},
    "Zr": {"density": 6.52, "atomic_weight": 91.224, "sigma_th": 0.185, "RI": 0.95},
}
