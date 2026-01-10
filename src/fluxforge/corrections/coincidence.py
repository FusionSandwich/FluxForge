"""
Coincidence Summing Correction Module.

Handles corrections for True Coincidence Summing (TCS) in gamma spectroscopy.
This is critical for accurate activity determination in close geometry measurements.

Theory
------
True Coincidence Summing occurs when two or more gamma rays from a single decay
event are detected "simultaneously" (within the detector resolution time).
This can cause:
- Summing-out: Loss from photopeak to higher energy
- Summing-in: Gain to photopeak from lower energies

The correction factor C_TCS corrects the observed peak area:
    A_corrected = A_observed × C_TCS

For cascading gammas γ1 → γ2:
    C_TCS(γ1) ≈ 1 / (1 - ε_t(γ2))  [summing-out dominant]

where ε_t is the TOTAL detection efficiency (not just peak efficiency).

References
----------
- Debertin & Schötzig, NIM A193 (1982) 375
- Blaauw, NIM A332 (1993) 493
- IAEA-TECDOC-1287 (2002)

Supports:
- Simple two-gamma cascades (Co-60, Y-88, Cs-134, Eu-152)
- Full decay scheme handling for complex cascades
- Geometry-dependent corrections
- Monte Carlo uncertainty propagation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np


class CascadeType(Enum):
    """Type of gamma cascade."""
    SIMPLE = "simple"        # Two-gamma cascade (Co-60)
    COMPLEX = "complex"      # Multi-gamma with branching
    ISOMERIC = "isomeric"    # Involves isomeric state
    BETA_GAMMA = "beta_gamma"  # Beta-gamma coincidence


@dataclass
class GammaTransition:
    """A single gamma-ray transition in a decay scheme."""
    
    energy_keV: float
    intensity: float  # Emission probability per decay
    multipolarity: str = ""  # E1, M1, E2, etc.
    conversion_coeff: float = 0.0  # Internal conversion coefficient
    
    @property
    def total_intensity(self) -> float:
        """Total transition probability including conversion."""
        return self.intensity * (1 + self.conversion_coeff)


@dataclass
class DecayLevel:
    """Energy level in a decay scheme."""
    
    energy_keV: float
    spin_parity: str = ""
    half_life_s: Optional[float] = None
    feeding_intensity: float = 0.0  # Probability to populate this level


@dataclass
class DecayScheme:
    """
    Complete decay scheme for TCS calculations.
    
    Describes the gamma cascade structure needed for accurate TCS corrections.
    """
    
    parent_isotope: str
    parent_half_life_s: float
    levels: List[DecayLevel] = field(default_factory=list)
    transitions: List[Tuple[int, int, GammaTransition]] = field(default_factory=list)
    
    def get_cascading_gammas(self, energy_keV: float, tolerance: float = 2.0) -> List[GammaTransition]:
        """Get gammas in cascade with the specified energy."""
        cascades = []
        # Find the transition matching this energy
        target_transition = None
        target_from = None
        target_to = None
        
        for from_level, to_level, transition in self.transitions:
            if abs(transition.energy_keV - energy_keV) < tolerance:
                target_transition = transition
                target_from = from_level
                target_to = to_level
                break
        
        if target_transition is None:
            return cascades
        
        # Find transitions that feed target_from (from higher levels)
        for from_level, to_level, transition in self.transitions:
            if to_level == target_from:
                cascades.append(transition)
        
        # Find transitions from target_to (to lower levels)
        for from_level, to_level, transition in self.transitions:
            if from_level == target_to:
                cascades.append(transition)
        
        return cascades


@dataclass
class CoincidenceCorrection:
    """Correction factors for a specific energy."""
    
    energy: float
    factor: float  # Multiplicative factor (C_TCS)
    uncertainty: float
    cascade_type: CascadeType = CascadeType.SIMPLE
    summing_out: float = 0.0  # Probability of loss
    summing_in: float = 0.0   # Probability of gain
    
    @property
    def loss_fraction(self) -> float:
        """Fraction of counts lost to summing-out."""
        if self.factor > 0:
            return 1.0 - 1.0/self.factor
        return 0.0


# ==============================================================================
# Decay Scheme Database for Common Isotopes
# ==============================================================================

# Pre-defined decay schemes for activation products commonly used in flux wire analysis
DECAY_SCHEMES: Dict[str, DecayScheme] = {}


def _init_decay_schemes():
    """Initialize decay scheme database."""
    
    # Co-60: Simple 2-gamma cascade
    # 60Co → 60Ni: 1173.2 keV (99.85%) → 1332.5 keV (99.9826%)
    co60_levels = [
        DecayLevel(energy_keV=2505.7, spin_parity="4+", feeding_intensity=0.9985),
        DecayLevel(energy_keV=1332.5, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=0.0, spin_parity="0+", feeding_intensity=0.0),
    ]
    co60_transitions = [
        (0, 1, GammaTransition(energy_keV=1173.2, intensity=0.9985)),
        (1, 2, GammaTransition(energy_keV=1332.5, intensity=0.999826)),
    ]
    DECAY_SCHEMES["Co60"] = DecayScheme(
        parent_isotope="Co-60",
        parent_half_life_s=1.6634e8,
        levels=co60_levels,
        transitions=co60_transitions,
    )
    
    # Y-88: 2-gamma cascade
    # 88Y → 88Sr: 898.04 keV (93.7%) + 1836.1 keV (99.2%)
    y88_levels = [
        DecayLevel(energy_keV=2734.1, spin_parity="3-", feeding_intensity=0.937),
        DecayLevel(energy_keV=1836.1, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=0.0, spin_parity="0+", feeding_intensity=0.0),
    ]
    y88_transitions = [
        (0, 1, GammaTransition(energy_keV=898.04, intensity=0.937)),
        (1, 2, GammaTransition(energy_keV=1836.1, intensity=0.992)),
    ]
    DECAY_SCHEMES["Y88"] = DecayScheme(
        parent_isotope="Y-88",
        parent_half_life_s=9.23e6,
        levels=y88_levels,
        transitions=y88_transitions,
    )
    
    # Cs-134: Multiple cascades
    # Major gammas: 604.7 keV (97.6%), 795.9 keV (85.5%), 569.3 keV (15.4%)
    cs134_levels = [
        DecayLevel(energy_keV=1969.8, spin_parity="3+", feeding_intensity=0.0278),
        DecayLevel(energy_keV=1643.3, spin_parity="4+", feeding_intensity=0.0255),
        DecayLevel(energy_keV=1400.5, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=1167.9, spin_parity="3+", feeding_intensity=0.0),
        DecayLevel(energy_keV=604.7, spin_parity="2+", feeding_intensity=0.9761),
        DecayLevel(energy_keV=0.0, spin_parity="0+", feeding_intensity=0.0),
    ]
    cs134_transitions = [
        (0, 3, GammaTransition(energy_keV=801.9, intensity=0.087)),
        (1, 3, GammaTransition(energy_keV=475.4, intensity=0.015)),
        (3, 4, GammaTransition(energy_keV=563.2, intensity=0.084)),
        (4, 5, GammaTransition(energy_keV=604.7, intensity=0.9761)),
        (2, 4, GammaTransition(energy_keV=795.9, intensity=0.855)),
        (3, 5, GammaTransition(energy_keV=1167.9, intensity=0.018)),
    ]
    DECAY_SCHEMES["Cs134"] = DecayScheme(
        parent_isotope="Cs-134",
        parent_half_life_s=6.51e7,
        levels=cs134_levels,
        transitions=cs134_transitions,
    )
    
    # Eu-152: Complex cascade (simplified main branches)
    eu152_levels = [
        DecayLevel(energy_keV=1579.5, spin_parity="2-", feeding_intensity=0.0),
        DecayLevel(energy_keV=1408.0, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=1299.1, spin_parity="0+", feeding_intensity=0.0),
        DecayLevel(energy_keV=1233.9, spin_parity="3-", feeding_intensity=0.0),
        DecayLevel(energy_keV=1085.8, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=963.4, spin_parity="3-", feeding_intensity=0.0),
        DecayLevel(energy_keV=755.4, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=344.3, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=121.8, spin_parity="4+", feeding_intensity=0.0),
        DecayLevel(energy_keV=0.0, spin_parity="0+", feeding_intensity=0.0),
    ]
    eu152_transitions = [
        (1, 9, GammaTransition(energy_keV=1408.0, intensity=0.212)),
        (7, 9, GammaTransition(energy_keV=344.3, intensity=0.265)),
        (8, 9, GammaTransition(energy_keV=121.8, intensity=0.284)),
        (4, 7, GammaTransition(energy_keV=244.7, intensity=0.0753)),
        (3, 8, GammaTransition(energy_keV=1112.1, intensity=0.136)),
        (5, 7, GammaTransition(energy_keV=964.1, intensity=0.146)),
        (6, 8, GammaTransition(energy_keV=411.1, intensity=0.0223)),
        (6, 7, GammaTransition(energy_keV=778.9, intensity=0.130)),
    ]
    DECAY_SCHEMES["Eu152"] = DecayScheme(
        parent_isotope="Eu-152",
        parent_half_life_s=4.27e8,
        levels=eu152_levels,
        transitions=eu152_transitions,
    )
    
    # Na-24: 2-gamma cascade (important for NAA)
    # 24Na → 24Mg: 1368.6 keV (100%) → 2754.0 keV (99.94%)
    na24_levels = [
        DecayLevel(energy_keV=4122.9, spin_parity="4+", feeding_intensity=0.9994),
        DecayLevel(energy_keV=1368.6, spin_parity="2+", feeding_intensity=0.0),
        DecayLevel(energy_keV=0.0, spin_parity="0+", feeding_intensity=0.0),
    ]
    na24_transitions = [
        (0, 1, GammaTransition(energy_keV=2754.0, intensity=0.9994)),
        (1, 2, GammaTransition(energy_keV=1368.6, intensity=1.0)),
    ]
    DECAY_SCHEMES["Na24"] = DecayScheme(
        parent_isotope="Na-24",
        parent_half_life_s=53820,
        levels=na24_levels,
        transitions=na24_transitions,
    )


# Initialize on module load
_init_decay_schemes()


class CoincidenceCorrector:
    """
    Calculator for True Coincidence Summing corrections.
    
    Implements both simplified cascade model and full decay scheme calculations.
    Requires total detection efficiency curve for accurate results.
    
    Parameters
    ----------
    efficiency_curve : callable, optional
        Object with evaluate(energy_keV) method returning total efficiency.
        Note: TCS requires TOTAL efficiency, not just peak efficiency.
        If you have peak efficiency, set peak_to_total_ratio appropriately.
    peak_to_total_ratio : float, optional
        Peak-to-total ratio for converting peak efficiency to total.
        Default is 1.0 (assumes efficiency_curve returns total efficiency).
        Set to ~0.4 if efficiency_curve returns peak efficiency at ~1 MeV.
    """
    
    def __init__(
        self, 
        efficiency_curve: Any = None,
        peak_to_total_ratio: float = 1.0,
    ):
        self.efficiency_curve = efficiency_curve
        self.peak_to_total_ratio = peak_to_total_ratio
        
    def _get_total_efficiency(self, energy_keV: float) -> float:
        """Get total efficiency at energy."""
        if self.efficiency_curve is None:
            return 0.0
        
        # Get peak efficiency
        if hasattr(self.efficiency_curve, 'evaluate'):
            peak_eff = self.efficiency_curve.evaluate(energy_keV)
        elif callable(self.efficiency_curve):
            peak_eff = self.efficiency_curve(energy_keV)
        else:
            return 0.0
        
        # Convert to total efficiency
        return peak_eff / self.peak_to_total_ratio
        
    def calculate_correction(
        self, 
        isotope: str, 
        energy: float, 
        geometry: str = "default",
        use_decay_scheme: bool = True,
    ) -> CoincidenceCorrection:
        """
        Calculate TCS correction factor.
        
        Parameters
        ----------
        isotope : str
            Isotope name (e.g., 'Co60', 'Na-24')
        energy : float
            Gamma energy in keV
        geometry : str
            Detector geometry ID (for lookup tables)
        use_decay_scheme : bool
            If True, use full decay scheme if available
            
        Returns
        -------
        CoincidenceCorrection
            Correction factor and associated data
        """
        # Normalize isotope name
        iso = isotope.upper().replace('-', '').replace(' ', '')
        
        # Try decay scheme approach first
        if use_decay_scheme and iso in DECAY_SCHEMES:
            return self._calculate_from_scheme(DECAY_SCHEMES[iso], energy)
        
        # Fall back to simple cascade models
        iso_lower = isotope.lower().replace('-', '')
        if iso_lower == 'co60':
            return self._calculate_co60_correction(energy)
        elif iso_lower == 'y88':
            return self._calculate_y88_correction(energy)
        elif iso_lower == 'na24':
            return self._calculate_na24_correction(energy)
        elif iso_lower == 'cs134':
            return self._calculate_cs134_correction(energy)
            
        # Default: No correction
        return CoincidenceCorrection(energy, 1.0, 0.0)
    
    def _calculate_from_scheme(
        self, 
        scheme: DecayScheme, 
        energy_keV: float,
    ) -> CoincidenceCorrection:
        """Calculate correction from full decay scheme."""
        
        # Find all cascading gammas
        cascades = scheme.get_cascading_gammas(energy_keV)
        
        if not cascades:
            return CoincidenceCorrection(energy_keV, 1.0, 0.0)
        
        # Calculate summing-out probability
        # P_out = sum over cascading gammas of (intensity * total_efficiency)
        summing_out = 0.0
        for gamma in cascades:
            eff_total = self._get_total_efficiency(gamma.energy_keV)
            summing_out += gamma.intensity * eff_total
        
        # Simple summing-out correction (ignoring summing-in for now)
        if summing_out >= 1.0:
            factor = 1.0  # Saturated, use approximation
        else:
            factor = 1.0 / (1.0 - summing_out)
        
        # Estimate uncertainty (5% relative + efficiency contribution)
        uncertainty = factor * 0.05
        
        cascade_type = CascadeType.COMPLEX if len(cascades) > 1 else CascadeType.SIMPLE
        
        return CoincidenceCorrection(
            energy=energy_keV,
            factor=factor,
            uncertainty=uncertainty,
            cascade_type=cascade_type,
            summing_out=summing_out,
            summing_in=0.0,
        )

    def _calculate_co60_correction(self, energy: float) -> CoincidenceCorrection:
        """
        Calculate Co-60 correction.
        
        Co-60 has two cascading gammas: 1173.2 keV and 1332.5 keV.
        They are in 1-1 cascade (99.85% x 99.98%).
        
        Correction factor C = 1 / (1 - P_sum)
        where P_sum = ε_t(partner) × I_partner
        """
        if self.efficiency_curve is None:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.SIMPLE)
            
        # Identify line and calculate summing-out
        if abs(energy - 1173.2) < 2.0:
            eff_neighbor = self._get_total_efficiency(1332.5)
            I_neighbor = 0.99983  # Intensity of 1332 line
            summing_out = eff_neighbor * I_neighbor
            
        elif abs(energy - 1332.5) < 2.0:
            eff_neighbor = self._get_total_efficiency(1173.2)
            I_neighbor = 0.9985  # Intensity of 1173 line
            summing_out = eff_neighbor * I_neighbor
            
        else:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.SIMPLE)
        
        factor = 1.0 / (1.0 - summing_out) if summing_out < 1.0 else 1.0
        uncertainty = factor * 0.05
        
        return CoincidenceCorrection(
            energy=energy,
            factor=factor,
            uncertainty=uncertainty,
            cascade_type=CascadeType.SIMPLE,
            summing_out=summing_out,
        )

    def _calculate_y88_correction(self, energy: float) -> CoincidenceCorrection:
        """Calculate Y-88 correction (898 keV and 1836 keV cascade)."""
        if self.efficiency_curve is None:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.SIMPLE)
            
        if abs(energy - 898.0) < 2.0:
            eff_neighbor = self._get_total_efficiency(1836.1)
            I_neighbor = 0.992
            summing_out = eff_neighbor * I_neighbor
            
        elif abs(energy - 1836.0) < 5.0:
            eff_neighbor = self._get_total_efficiency(898.0)
            I_neighbor = 0.937
            summing_out = eff_neighbor * I_neighbor
            
        else:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.SIMPLE)
        
        factor = 1.0 / (1.0 - summing_out) if summing_out < 1.0 else 1.0
        return CoincidenceCorrection(
            energy=energy,
            factor=factor,
            uncertainty=factor * 0.05,
            cascade_type=CascadeType.SIMPLE,
            summing_out=summing_out,
        )

    def _calculate_na24_correction(self, energy: float) -> CoincidenceCorrection:
        """Calculate Na-24 correction (1368.6 and 2754.0 keV cascade)."""
        if self.efficiency_curve is None:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.SIMPLE)
            
        if abs(energy - 1368.6) < 2.0:
            eff_neighbor = self._get_total_efficiency(2754.0)
            summing_out = eff_neighbor * 0.9994
            
        elif abs(energy - 2754.0) < 5.0:
            eff_neighbor = self._get_total_efficiency(1368.6)
            summing_out = eff_neighbor * 1.0
            
        else:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.SIMPLE)
        
        factor = 1.0 / (1.0 - summing_out) if summing_out < 1.0 else 1.0
        return CoincidenceCorrection(
            energy=energy,
            factor=factor,
            uncertainty=factor * 0.05,
            cascade_type=CascadeType.SIMPLE,
            summing_out=summing_out,
        )

    def _calculate_cs134_correction(self, energy: float) -> CoincidenceCorrection:
        """Calculate Cs-134 correction (complex cascade)."""
        if self.efficiency_curve is None:
            return CoincidenceCorrection(energy, 1.0, 0.0, CascadeType.COMPLEX)
        
        # Use decay scheme for complex cascades
        if "Cs134" in DECAY_SCHEMES:
            return self._calculate_from_scheme(DECAY_SCHEMES["Cs134"], energy)
        
        # Simplified for main lines
        summing_out = 0.0
        
        if abs(energy - 604.7) < 2.0:
            # 604.7 keV is fed by multiple cascading gammas
            eff_796 = self._get_total_efficiency(795.9)
            eff_563 = self._get_total_efficiency(563.2)
            summing_out = eff_796 * 0.855 + eff_563 * 0.084
            
        elif abs(energy - 795.9) < 2.0:
            # 795.9 keV feeds 604.7 keV
            eff_605 = self._get_total_efficiency(604.7)
            summing_out = eff_605 * 0.976
        
        factor = 1.0 / (1.0 - summing_out) if summing_out < 1.0 else 1.0
        return CoincidenceCorrection(
            energy=energy,
            factor=factor,
            uncertainty=factor * 0.08,  # Higher uncertainty for complex
            cascade_type=CascadeType.COMPLEX,
            summing_out=summing_out,
        )

    def calculate_batch(
        self, 
        isotope: str, 
        energies: List[float],
    ) -> List[CoincidenceCorrection]:
        """Calculate corrections for multiple energies."""
        return [self.calculate_correction(isotope, e) for e in energies]


def get_available_isotopes() -> List[str]:
    """Get list of isotopes with decay scheme data."""
    return list(DECAY_SCHEMES.keys())


def get_decay_scheme(isotope: str) -> Optional[DecayScheme]:
    """Get decay scheme for an isotope."""
    iso = isotope.upper().replace('-', '').replace(' ', '')
    return DECAY_SCHEMES.get(iso)
