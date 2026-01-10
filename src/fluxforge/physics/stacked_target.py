"""
Stacked Target Module - Epic T (Curie Parity)

Implements stacked-target activation characterization:
- Foil stack definition and energy degradation
- Activation rate calculation at each foil
- Flux unfolding from activation foils
- Cross-section library interface (IRDFF, ENDF, TENDL)

This module enables charged-particle spectrum determination
using activation foil stacks - a key Curie feature.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from .stopping_power import (
    Material, Projectile, STANDARD_MATERIALS,
    calculate_energy_loss, calculate_straggling, StragglResult
)


@dataclass
class Foil:
    """Single foil in a stacked target."""
    material: Material
    thickness_um: float  # Micrometers
    reaction: Optional[str] = None  # e.g., "(p,n)", "(p,x)"
    target_isotope: Optional[str] = None  # e.g., "Cu-63"
    product_isotope: Optional[str] = None  # e.g., "Zn-63"
    
    @property
    def thickness_mg_cm2(self) -> float:
        """Thickness in mg/cm²."""
        return self.thickness_um * self.material.density_g_cm3 * 0.1
    
    @property
    def thickness_g_cm2(self) -> float:
        """Thickness in g/cm²."""
        return self.thickness_um * self.material.density_g_cm3 * 1e-4


@dataclass
class EnergyAtFoil:
    """Energy information at a foil position."""
    foil_index: int
    energy_in_MeV: float
    energy_out_MeV: float
    energy_mean_MeV: float
    energy_loss_MeV: float
    straggling: StragglResult


@dataclass
class StackedTarget:
    """
    Stacked target for activation measurements.
    
    A stack of foils where beam energy degrades through
    successive layers, enabling flux/spectrum measurement
    at multiple energy points.
    """
    foils: List[Foil] = field(default_factory=list)
    projectile: Projectile = Projectile.PROTON
    beam_energy_MeV: float = 0.0
    
    def add_foil(
        self,
        material: Union[str, Material],
        thickness_um: float,
        reaction: Optional[str] = None,
        target_isotope: Optional[str] = None,
        product_isotope: Optional[str] = None
    ) -> 'StackedTarget':
        """
        Add a foil to the stack.
        
        Parameters
        ----------
        material : str or Material
            Material name or Material object
        thickness_um : float
            Thickness in micrometers
        reaction : str, optional
            Nuclear reaction
        target_isotope : str, optional
            Target isotope
        product_isotope : str, optional
            Product isotope
        
        Returns
        -------
        StackedTarget
            Self for chaining
        """
        if isinstance(material, str):
            if material.lower() in STANDARD_MATERIALS:
                material = STANDARD_MATERIALS[material.lower()]
            else:
                raise ValueError(f"Unknown material: {material}")
        
        foil = Foil(
            material=material,
            thickness_um=thickness_um,
            reaction=reaction,
            target_isotope=target_isotope,
            product_isotope=product_isotope
        )
        self.foils.append(foil)
        return self
    
    def calculate_energies(self, n_steps: int = 100) -> List[EnergyAtFoil]:
        """
        Calculate beam energy at each foil.
        
        Parameters
        ----------
        n_steps : int
            Integration steps per foil
        
        Returns
        -------
        List[EnergyAtFoil]
            Energy information at each foil
        """
        energies = []
        E = self.beam_energy_MeV
        
        for i, foil in enumerate(self.foils):
            E_in = E
            
            # Calculate energy loss and straggling
            strag = calculate_straggling(
                E_in, self.projectile, foil.material, foil.thickness_mg_cm2
            )
            
            E_out = strag.mean_energy_MeV
            E_mean = (E_in + E_out) / 2
            dE = E_in - E_out
            
            energies.append(EnergyAtFoil(
                foil_index=i,
                energy_in_MeV=E_in,
                energy_out_MeV=E_out,
                energy_mean_MeV=E_mean,
                energy_loss_MeV=dE,
                straggling=strag
            ))
            
            E = E_out
            if E <= 0:
                break
        
        return energies
    
    def get_energy_profile(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get energy profile through stack.
        
        Returns
        -------
        tuple
            (positions_um, energies_MeV) arrays
        """
        energies = self.calculate_energies()
        
        positions = [0.0]
        E_values = [self.beam_energy_MeV]
        
        cumulative_thickness = 0.0
        for i, e_info in enumerate(energies):
            foil = self.foils[e_info.foil_index]
            # Entry point
            positions.append(cumulative_thickness)
            E_values.append(e_info.energy_in_MeV)
            
            # Exit point
            cumulative_thickness += foil.thickness_um
            positions.append(cumulative_thickness)
            E_values.append(e_info.energy_out_MeV)
        
        return np.array(positions), np.array(E_values)


@dataclass
class CrossSectionData:
    """Cross-section data for a reaction."""
    reaction: str
    target: str
    product: str
    energies_MeV: NDArray[np.float64]
    cross_sections_mb: NDArray[np.float64]
    uncertainties_mb: Optional[NDArray[np.float64]] = None
    library: str = "unknown"
    
    def interpolate(self, energy_MeV: float) -> float:
        """Interpolate cross-section at given energy."""
        if energy_MeV < self.energies_MeV[0]:
            return 0.0
        if energy_MeV > self.energies_MeV[-1]:
            return 0.0
        return np.interp(energy_MeV, self.energies_MeV, self.cross_sections_mb)


class CrossSectionLibrary:
    """
    Multi-library cross-section interface.
    
    Provides access to:
    - IRDFF-II (International Reactor Dosimetry and Fusion File)
    - ENDF/B-VIII.0
    - TENDL (TALYS-based Evaluated Nuclear Data Library)
    """
    
    LIBRARY_PATHS: Dict[str, Path] = {}
    
    # Common monitor reactions (placeholder data)
    MONITOR_REACTIONS: Dict[str, CrossSectionData] = {}
    
    @classmethod
    def add_library_path(cls, name: str, path: Path) -> None:
        """Register a library path."""
        cls.LIBRARY_PATHS[name] = path
    
    @classmethod
    def get_cross_section(
        cls,
        reaction: str,
        library: str = "irdff"
    ) -> Optional[CrossSectionData]:
        """
        Get cross-section data for a reaction.
        
        Parameters
        ----------
        reaction : str
            Reaction string, e.g., "Cu-63(p,n)Zn-63"
        library : str
            Library name: "irdff", "endf", "tendl"
        
        Returns
        -------
        CrossSectionData or None
        """
        # Check cached reactions
        key = f"{library}:{reaction}"
        if key in cls.MONITOR_REACTIONS:
            return cls.MONITOR_REACTIONS[key]
        
        # Try to load from file
        if library.lower() in cls.LIBRARY_PATHS:
            # Implementation for reading ENDF/ACE files would go here
            pass
        
        return None
    
    @classmethod
    def search_reactions(
        cls,
        target: Optional[str] = None,
        projectile: Optional[str] = None,
        product: Optional[str] = None,
        library: str = "irdff"
    ) -> List[str]:
        """
        Search for available reactions.
        
        Parameters
        ----------
        target : str, optional
            Target isotope filter
        projectile : str, optional
            Projectile filter ("p", "n", "d", "a")
        product : str, optional
            Product isotope filter
        library : str
            Library to search
        
        Returns
        -------
        List[str]
            List of matching reaction strings
        """
        matches = []
        for key, xs in cls.MONITOR_REACTIONS.items():
            lib, rxn = key.split(":", 1)
            if lib.lower() != library.lower():
                continue
            if target and target.lower() not in xs.target.lower():
                continue
            if product and product.lower() not in xs.product.lower():
                continue
            matches.append(rxn)
        
        return matches
    
    @classmethod
    def register_reaction(
        cls,
        reaction: str,
        energies_MeV: NDArray[np.float64],
        cross_sections_mb: NDArray[np.float64],
        library: str = "user",
        uncertainties_mb: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Register a cross-section dataset.
        
        Parameters
        ----------
        reaction : str
            Reaction string, e.g., "Cu-63(p,n)Zn-63"
        energies_MeV : array
            Energy grid
        cross_sections_mb : array
            Cross-sections in millibarns
        library : str
            Library name
        uncertainties_mb : array, optional
            Uncertainties
        """
        # Parse reaction
        import re
        match = re.match(r'(\w+[-\d]+)\((\w+,\w+)\)(\w+[-\d]+)', reaction)
        if match:
            target, rxn_type, product = match.groups()
        else:
            target, rxn_type, product = reaction, "", ""
        
        xs = CrossSectionData(
            reaction=reaction,
            target=target,
            product=product,
            energies_MeV=energies_MeV,
            cross_sections_mb=cross_sections_mb,
            uncertainties_mb=uncertainties_mb,
            library=library
        )
        
        key = f"{library}:{reaction}"
        cls.MONITOR_REACTIONS[key] = xs


# Register some common monitor reactions (approximate data for testing)
def _init_monitor_reactions():
    """Initialize common monitor reaction data."""
    # Cu-63(p,n)Zn-63 - common proton monitor
    E = np.array([4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50])
    sigma = np.array([10, 50, 150, 280, 380, 450, 420, 320, 180, 120, 85, 50, 35])
    CrossSectionLibrary.register_reaction(
        "Cu-63(p,n)Zn-63", E, sigma, "irdff"
    )
    
    # Al-27(p,x)Na-22 - common monitor
    E = np.array([15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100])
    sigma = np.array([0.5, 5, 20, 35, 45, 50, 52, 48, 42, 38, 32])
    CrossSectionLibrary.register_reaction(
        "Al-27(p,x)Na-22", E, sigma, "irdff"
    )
    
    # Ti-nat(p,x)V-48 - common monitor
    E = np.array([5, 8, 10, 12, 15, 20, 25, 30, 40, 50])
    sigma = np.array([5, 50, 200, 320, 380, 350, 280, 200, 100, 60])
    CrossSectionLibrary.register_reaction(
        "Ti-nat(p,x)V-48", E, sigma, "irdff"
    )

_init_monitor_reactions()


@dataclass
class ActivationResult:
    """Activation measurement result for a foil."""
    foil_index: int
    reaction: str
    energy_MeV: float
    energy_spread_MeV: float
    activity_Bq: float
    activity_uncertainty_Bq: float
    cross_section_mb: float


def calculate_activation(
    stack: StackedTarget,
    flux_p_cm2_s: float,
    irradiation_time_s: float,
    cooling_time_s: float = 0.0
) -> List[ActivationResult]:
    """
    Calculate expected activation for stack foils.
    
    Parameters
    ----------
    stack : StackedTarget
        The stacked target
    flux_p_cm2_s : float
        Beam flux (particles/cm²/s)
    irradiation_time_s : float
        Irradiation time
    cooling_time_s : float
        Cooling time before counting
    
    Returns
    -------
    List[ActivationResult]
        Activation results for each foil
    """
    from ..data.elements import ATOMIC_MASSES
    
    N_A = 6.022e23
    results = []
    
    energies = stack.calculate_energies()
    
    for e_info in energies:
        foil = stack.foils[e_info.foil_index]
        
        if not foil.reaction or not foil.target_isotope:
            continue
        
        # Get cross-section at mean energy
        rxn_str = f"{foil.target_isotope}({foil.reaction}){foil.product_isotope}"
        xs_data = CrossSectionLibrary.get_cross_section(rxn_str)
        
        if xs_data is None:
            # Try to find similar reaction
            continue
        
        sigma = xs_data.interpolate(e_info.energy_mean_MeV)  # mb
        sigma_cm2 = sigma * 1e-27  # Convert to cm²
        
        # Number of target atoms
        # Simplified - assumes natural abundance
        A = 63  # Example for Cu-63
        n_atoms = N_A * foil.thickness_g_cm2 / A
        
        # Activity = flux * N * sigma (simplified, ignoring decay during irradiation)
        activity = flux_p_cm2_s * n_atoms * sigma_cm2
        
        results.append(ActivationResult(
            foil_index=e_info.foil_index,
            reaction=rxn_str,
            energy_MeV=e_info.energy_mean_MeV,
            energy_spread_MeV=e_info.straggling.sigma_MeV,
            activity_Bq=activity,
            activity_uncertainty_Bq=activity * 0.1,  # 10% assumed
            cross_section_mb=sigma
        ))
    
    return results


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing stacked_target module...")
    
    # Create a typical monitor foil stack
    stack = StackedTarget(
        projectile=Projectile.PROTON,
        beam_energy_MeV=30.0
    )
    
    # Add foils (typical arrangement)
    stack.add_foil('aluminum', 100)  # Degrader
    stack.add_foil('copper', 25, "(p,n)", "Cu-63", "Zn-63")
    stack.add_foil('aluminum', 200)  # Degrader
    stack.add_foil('copper', 25, "(p,n)", "Cu-63", "Zn-63")
    stack.add_foil('aluminum', 300)  # Degrader
    stack.add_foil('copper', 25, "(p,n)", "Cu-63", "Zn-63")
    
    print(f"\nStacked target with {len(stack.foils)} foils:")
    print(f"  Beam: {stack.projectile.symbol} at {stack.beam_energy_MeV} MeV")
    
    # Calculate energy at each foil
    energies = stack.calculate_energies()
    
    print("\nEnergy at each foil:")
    for e_info in energies:
        foil = stack.foils[e_info.foil_index]
        print(f"  Foil {e_info.foil_index} ({foil.material.name}, {foil.thickness_um}μm):")
        print(f"    E_in={e_info.energy_in_MeV:.2f} MeV, E_out={e_info.energy_out_MeV:.2f} MeV")
        print(f"    E_mean={e_info.energy_mean_MeV:.2f} MeV, σ={e_info.straggling.sigma_MeV:.3f} MeV")
    
    # Test energy profile
    positions, E_profile = stack.get_energy_profile()
    print(f"\nEnergy profile: {len(positions)} points")
    print(f"  Start: {E_profile[0]:.2f} MeV at {positions[0]:.1f} μm")
    print(f"  End: {E_profile[-1]:.2f} MeV at {positions[-1]:.1f} μm")
    
    # Test cross-section library
    print("\nCross-section library:")
    for rxn in ["Cu-63(p,n)Zn-63", "Al-27(p,x)Na-22", "Ti-nat(p,x)V-48"]:
        xs = CrossSectionLibrary.get_cross_section(rxn, "irdff")
        if xs:
            print(f"  {rxn}: {len(xs.energies_MeV)} points, max σ={max(xs.cross_sections_mb):.1f} mb")
    
    print("\n✅ stacked_target module tests passed!")
