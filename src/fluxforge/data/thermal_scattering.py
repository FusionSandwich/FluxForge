"""Thermal Scattering Law (S(α,β)) metadata and material mappings.

This module provides material-to-TSL identifier mappings for thermal
scattering treatment in NJOY processing and transport codes.

Thermal scattering is important for:
- Light moderators (H2O, D2O, graphite, Be, BeO)
- Materials at cryogenic temperatures
- Hydrogenous materials where bound scattering differs from free-gas

References:
    ENDF-102 Manual, Section 4.7 (MF7)
    MCNP6 User Manual, Appendix G
    OpenMC documentation on thermal scattering data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TSLLibrary(Enum):
    """Thermal scattering library identifiers."""
    
    # ENDF/B-VIII.0 TSL identifiers
    ENDF8_LWTR = "lwtr"    # Light water H
    ENDF8_HWTR = "hwtr"    # Heavy water D
    ENDF8_HZR = "hzr"      # H in ZrH
    ENDF8_DZR = "dzr"      # D in ZrD  
    ENDF8_GRPH = "grph"    # Graphite
    ENDF8_BE = "be"        # Beryllium metal
    ENDF8_BEO = "beo"      # Beryllium oxide
    ENDF8_POLY = "poly"    # Polyethylene CH2
    ENDF8_BENZ = "benz"    # Benzene C6H6
    ENDF8_METH = "meth"    # Methane CH4
    ENDF8_AL = "al"        # Aluminum
    ENDF8_FE = "fe"        # Iron
    ENDF8_SI = "si"        # Silicon
    ENDF8_UMET = "umet"    # Uranium metal
    ENDF8_UO2 = "uo2"      # Uranium dioxide
    
    # JEFF-3.3 identifiers
    JEFF_LWTR = "j_lwtr"
    JEFF_GRPH = "j_grph"
    
    # Custom/user libraries
    CUSTOM = "custom"


@dataclass
class ThermalScatteringData:
    """Thermal scattering data for a material-nuclide pair.
    
    Attributes
    ----------
    material : str
        Material name (e.g., "H2O", "graphite")
    bound_nuclide : str
        Nuclide treated with bound scattering (e.g., "H", "C")
    tsl_id : str
        Thermal scattering law identifier
    library : TSLLibrary
        Source library
    temperatures_K : list[float]
        Available temperatures in Kelvin
    mat_number : int
        ENDF MAT number for the TSL evaluation
    description : str
        Human-readable description
    """
    
    material: str
    bound_nuclide: str
    tsl_id: str
    library: TSLLibrary
    temperatures_K: list[float] = field(default_factory=list)
    mat_number: int = 0
    description: str = ""
    
    def __post_init__(self):
        """Sort temperatures after initialization."""
        self.temperatures_K = sorted(self.temperatures_K)
    
    def nearest_temperature(self, T: float) -> float:
        """Find nearest available temperature.
        
        Parameters
        ----------
        T : float
            Desired temperature in Kelvin
            
        Returns
        -------
        float
            Nearest available temperature
        """
        if not self.temperatures_K:
            return T
        
        return min(self.temperatures_K, key=lambda x: abs(x - T))
    
    def interpolation_temperatures(self, T: float) -> tuple[Optional[float], Optional[float]]:
        """Get bracketing temperatures for interpolation.
        
        Parameters
        ----------
        T : float
            Target temperature in Kelvin
            
        Returns
        -------
        tuple[float | None, float | None]
            (T_low, T_high) or None if outside range
        """
        if not self.temperatures_K:
            return None, None
        
        temps = self.temperatures_K
        if T <= temps[0]:
            return None, temps[0]
        if T >= temps[-1]:
            return temps[-1], None
        
        for i, t in enumerate(temps[:-1]):
            if t <= T <= temps[i + 1]:
                return t, temps[i + 1]
        
        return temps[-1], None


# Standard ENDF/B-VIII.0 thermal scattering data
# Temperatures from ENDF/B-VIII.0 release
ENDF8_TSL_DATA = {
    "H2O": ThermalScatteringData(
        material="H2O",
        bound_nuclide="H",
        tsl_id="lwtr",
        library=TSLLibrary.ENDF8_LWTR,
        mat_number=1,
        temperatures_K=[293.6, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 800.0],
        description="Hydrogen bound in light water",
    ),
    "D2O": ThermalScatteringData(
        material="D2O",
        bound_nuclide="D",
        tsl_id="hwtr",
        library=TSLLibrary.ENDF8_HWTR,
        mat_number=2,
        temperatures_K=[293.6, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0],
        description="Deuterium bound in heavy water",
    ),
    "graphite": ThermalScatteringData(
        material="graphite",
        bound_nuclide="C",
        tsl_id="grph",
        library=TSLLibrary.ENDF8_GRPH,
        mat_number=31,
        temperatures_K=[293.6, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0, 1200.0, 1600.0, 2000.0],
        description="Carbon in crystalline graphite",
    ),
    "Be": ThermalScatteringData(
        material="Be",
        bound_nuclide="Be",
        tsl_id="be",
        library=TSLLibrary.ENDF8_BE,
        mat_number=26,
        temperatures_K=[293.6, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0, 1200.0],
        description="Beryllium metal",
    ),
    "BeO": ThermalScatteringData(
        material="BeO",
        bound_nuclide="Be",
        tsl_id="beo",
        library=TSLLibrary.ENDF8_BEO,
        mat_number=27,
        temperatures_K=[293.6, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0],
        description="Beryllium in beryllium oxide",
    ),
    "ZrH": ThermalScatteringData(
        material="ZrH",
        bound_nuclide="H",
        tsl_id="hzr",
        library=TSLLibrary.ENDF8_HZR,
        mat_number=7,
        temperatures_K=[293.6, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0, 1200.0],
        description="Hydrogen bound in zirconium hydride",
    ),
    "polyethylene": ThermalScatteringData(
        material="polyethylene",
        bound_nuclide="H",
        tsl_id="poly",
        library=TSLLibrary.ENDF8_POLY,
        mat_number=37,
        temperatures_K=[293.6, 350.0, 400.0],
        description="Hydrogen bound in polyethylene CH2",
    ),
    "Al": ThermalScatteringData(
        material="Al",
        bound_nuclide="Al",
        tsl_id="al",
        library=TSLLibrary.ENDF8_AL,
        mat_number=13,
        temperatures_K=[20.0, 80.0, 293.6, 400.0, 500.0, 600.0],
        description="Aluminum metal",
    ),
    "Fe": ThermalScatteringData(
        material="Fe",
        bound_nuclide="Fe",
        tsl_id="fe",
        library=TSLLibrary.ENDF8_FE,
        mat_number=56,
        temperatures_K=[20.0, 80.0, 293.6, 400.0, 500.0, 600.0],
        description="Iron metal (alpha phase)",
    ),
    "UO2": ThermalScatteringData(
        material="UO2",
        bound_nuclide="U",
        tsl_id="uo2",
        library=TSLLibrary.ENDF8_UO2,
        mat_number=75,
        temperatures_K=[293.6, 500.0, 700.0, 900.0, 1200.0, 1500.0, 2000.0],
        description="Uranium in uranium dioxide",
    ),
}

# MCNP-style identifiers (SAB card suffixes)
MCNP_SAB_IDENTIFIERS = {
    "H2O": "lwtr",
    "D2O": "hwtr",
    "graphite": "grph",
    "Be": "be",
    "BeO": "beo",
    "ZrH": "h-zr",
    "polyethylene": "poly",
    "Al": "al27",
    "Fe": "fe56",
}

# OpenMC-style library names
OPENMC_TSL_NAMES = {
    "H2O": "c_H_in_H2O",
    "D2O": "c_D_in_D2O",
    "graphite": "c_C_in_graphite",
    "Be": "c_Be_in_Be",
    "BeO": "c_Be_in_BeO",
    "ZrH": "c_H_in_ZrH",
    "polyethylene": "c_H_in_CH2",
}


def get_tsl_for_material(
    material: str,
    temperature_K: float = 293.6,
) -> Optional[ThermalScatteringData]:
    """Get thermal scattering data for a material.
    
    Parameters
    ----------
    material : str
        Material name (e.g., "H2O", "graphite", "Be")
    temperature_K : float
        Operating temperature in Kelvin (for validation)
        
    Returns
    -------
    ThermalScatteringData or None
        TSL data if available
    """
    # Normalize material name
    material_key = material.strip()
    
    # Try exact match
    if material_key in ENDF8_TSL_DATA:
        return ENDF8_TSL_DATA[material_key]
    
    # Try case-insensitive match
    for key, data in ENDF8_TSL_DATA.items():
        if key.lower() == material_key.lower():
            return data
    
    # Try common aliases
    aliases = {
        "water": "H2O",
        "light_water": "H2O",
        "light water": "H2O",
        "heavy_water": "D2O",
        "heavy water": "D2O",
        "carbon": "graphite",
        "beryllium": "Be",
        "beryllia": "BeO",
        "beryllium_oxide": "BeO",
        "zirconium_hydride": "ZrH",
        "poly": "polyethylene",
        "CH2": "polyethylene",
        "aluminum": "Al",
        "aluminium": "Al",
        "iron": "Fe",
        "uranium_dioxide": "UO2",
    }
    
    if material_key.lower() in aliases:
        canonical = aliases[material_key.lower()]
        return ENDF8_TSL_DATA.get(canonical)
    
    return None


def requires_thermal_scattering(nuclide: str, material_context: Optional[str] = None) -> bool:
    """Check if a nuclide requires thermal scattering treatment.
    
    Parameters
    ----------
    nuclide : str
        Nuclide name (e.g., "H1", "C12", "Be9")
    material_context : str, optional
        Material context for determining if bound scattering is relevant
        
    Returns
    -------
    bool
        True if thermal scattering data should be used
    """
    # Elements that commonly require thermal scattering
    tsl_elements = {"H", "D", "C", "Be", "O", "Al", "Fe", "U", "Zr"}
    
    # Extract element from nuclide
    element = "".join(c for c in nuclide if c.isalpha())
    
    if element not in tsl_elements:
        return False
    
    # If material context provided, check if TSL available
    if material_context:
        tsl = get_tsl_for_material(material_context)
        if tsl and tsl.bound_nuclide == element:
            return True
        return False
    
    # Default: only H, D, C, Be require TSL without context
    return element in {"H", "D", "C", "Be"}


def get_mcnp_sab_card(material: str, temperature_K: float = 293.6) -> Optional[str]:
    """Get MCNP SAB card identifier for thermal scattering.
    
    Parameters
    ----------
    material : str
        Material name
    temperature_K : float
        Temperature in Kelvin (for selecting temperature suffix)
        
    Returns
    -------
    str or None
        MCNP SAB identifier (e.g., "lwtr.20t") or None
    """
    tsl = get_tsl_for_material(material, temperature_K)
    if tsl is None:
        return None
    
    base_id = MCNP_SAB_IDENTIFIERS.get(material, tsl.tsl_id)
    
    # Temperature suffix (approximate mapping)
    # MCNP uses suffixes like .10t, .12t, .14t, etc.
    T_nearest = tsl.nearest_temperature(temperature_K)
    
    # Map temperature to MCNP suffix (simplified)
    if T_nearest <= 300:
        suffix = ".10t"  # Room temperature
    elif T_nearest <= 400:
        suffix = ".12t"
    elif T_nearest <= 500:
        suffix = ".14t"
    elif T_nearest <= 600:
        suffix = ".16t"
    else:
        suffix = ".18t"
    
    return f"{base_id}{suffix}"


def get_openmc_tsl_name(material: str) -> Optional[str]:
    """Get OpenMC thermal scattering library name.
    
    Parameters
    ----------
    material : str
        Material name
        
    Returns
    -------
    str or None
        OpenMC library name (e.g., "c_H_in_H2O")
    """
    tsl = get_tsl_for_material(material)
    if tsl is None:
        return None
    
    return OPENMC_TSL_NAMES.get(material)


@dataclass
class ThermalScatteringConfig:
    """Configuration for thermal scattering treatment.
    
    Attributes
    ----------
    enabled : bool
        Whether to use thermal scattering libraries
    materials : dict[str, ThermalScatteringData]
        Material-to-TSL mappings
    temperature_K : float
        Default temperature for TSL selection
    interpolate : bool
        Whether to interpolate between temperature points
    """
    
    enabled: bool = True
    materials: dict[str, ThermalScatteringData] = field(default_factory=dict)
    temperature_K: float = 293.6
    interpolate: bool = False
    
    @classmethod
    def default(cls) -> ThermalScatteringConfig:
        """Create default configuration with standard materials."""
        return cls(
            enabled=True,
            materials=dict(ENDF8_TSL_DATA),
            temperature_K=293.6,
        )
    
    def add_material(self, material: str, tsl_id: Optional[str] = None) -> None:
        """Add a material to the configuration.
        
        Parameters
        ----------
        material : str
            Material name
        tsl_id : str, optional
            Custom TSL identifier
        """
        if tsl_id:
            self.materials[material] = ThermalScatteringData(
                material=material,
                bound_nuclide="",
                tsl_id=tsl_id,
                library=TSLLibrary.CUSTOM,
            )
        else:
            std_tsl = get_tsl_for_material(material)
            if std_tsl:
                self.materials[material] = std_tsl
    
    def get_njoy_thermr_inputs(self, material: str) -> dict:
        """Get NJOY THERMR module inputs for a material.
        
        Parameters
        ----------
        material : str
            Material name
            
        Returns
        -------
        dict
            NJOY THERMR input parameters
        """
        tsl = self.materials.get(material) or get_tsl_for_material(material)
        if tsl is None:
            return {}
        
        T_nearest = tsl.nearest_temperature(self.temperature_K)
        
        return {
            "matde": tsl.mat_number,
            "matdp": tsl.mat_number,
            "ntemp": 1,
            "tempr": [T_nearest],
            "tol": 0.001,
            "emax": 4.0,  # Maximum thermal energy in eV
        }


# Export main interface
__all__ = [
    "TSLLibrary",
    "ThermalScatteringData",
    "ThermalScatteringConfig",
    "get_tsl_for_material",
    "requires_thermal_scattering",
    "get_mcnp_sab_card",
    "get_openmc_tsl_name",
    "ENDF8_TSL_DATA",
]
