"""
k0-Standardization Workflow for TRIGA Reactors
==============================================

This module implements the k0-standardization method (k0-NAA) for 
elemental concentration determination in TRIGA reactors, following
the methods described in:

- De Corte et al. (1987) "The k0-standardisation method"
- Di Luzio et al. (2017) "Measurement of neutron flux parameters f and α"
- Menezes et al. (2023) "Characterization of irradiation channels"
- Guesmia et al. (2025) "k0-NAA at NUR research reactor"

Theory
------
The k0-NAA method relates element concentration to measured peak areas:

    ρ = [Np/(tm·S·D·C·W)]_a / [Np/(tm·S·D·C·w)]_c 
        × 1/k0 × [f + Q0,c(α)]·εp,c / [f + Q0,a(α)]·εp,a

Where:
    - ρ: Mass fraction (mg/kg)
    - Np: Net peak area
    - S, D, C: Saturation, decay, counting factors
    - W, w: Sample and comparator weights
    - k0: Comparator k0 factor
    - f: Thermal-to-epithermal flux ratio
    - Q0(α): α-corrected resonance integral ratio
    - εp: Full-energy peak efficiency

The module supports multiple methods for determining f and α:
1. Cd-ratio multi-monitor method (bare vs Cd-covered)
2. Bare triple-monitor method (Zr-94, Zr-96, Au-197)
3. Single monitor with known f and α

References
----------
- IAEA k0 database (https://www.kayzero.com)
- IRDFF-II dosimetry reaction library
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
from pathlib import Path

# Import from sister modules
try:
    from .cd_ratio import (
        CdRatioMeasurement,
        FluxParameters,
        CdRatioAnalyzer,
        STANDARD_MONITORS,
        calculate_Q0_alpha,
        estimate_f,
    )
except ImportError:
    # Standalone import
    from cd_ratio import (
        CdRatioMeasurement,
        FluxParameters,
        CdRatioAnalyzer,
        STANDARD_MONITORS,
        calculate_Q0_alpha,
        estimate_f,
    )


# ==============================================================================
# Westcott g(T) Factors
# ==============================================================================

@dataclass
class WestcottFactors:
    """
    Westcott g(T) and s(T) factors for non-1/v cross section correction.
    
    The Westcott convention parameterizes the effective cross section as:
        σ_eff = σ_0 × [g(T) + r × s(T)]
    
    where:
        g(T) = thermal component factor (departure from 1/v)
        s(T) = epithermal component factor
        r = epithermal index = φ_epi/φ_th × sqrt(π·T/T_0)/2
        T_0 = 293.6 K (reference temperature)
    
    For 1/v absorbers: g(T) = 1.0 (independent of T)
    For non-1/v absorbers: g(T) varies with temperature
    
    Attributes:
        isotope: Target isotope
        reaction: Reaction type (capture, fission)
        g_coefficients: Polynomial coefficients for g(T) = Σ a_n × (T/T_0)^n
        s_coefficients: Polynomial coefficients for s(T)
        T_ref: Reference temperature (K)
        valid_range: Valid temperature range (K_min, K_max)
    """
    
    isotope: str
    reaction: str
    g_coefficients: Tuple[float, ...] = (1.0,)  # Default: 1/v behavior
    s_coefficients: Tuple[float, ...] = (0.0,)
    T_ref: float = 293.6  # K (20.45°C)
    valid_range: Tuple[float, float] = (200.0, 1000.0)
    
    def g(self, T_K: float) -> float:
        """
        Calculate g(T) factor at temperature T.
        
        Parameters
        ----------
        T_K : float
            Temperature in Kelvin
            
        Returns
        -------
        float
            g(T) factor
        """
        T_ratio = T_K / self.T_ref
        return sum(a * (T_ratio ** n) for n, a in enumerate(self.g_coefficients))
    
    def s(self, T_K: float) -> float:
        """Calculate s(T) factor at temperature T."""
        T_ratio = T_K / self.T_ref
        return sum(a * (T_ratio ** n) for n, a in enumerate(self.s_coefficients))
    
    def effective_factor(self, T_K: float, r: float = 0.0) -> float:
        """
        Calculate total Westcott factor: g(T) + r × s(T).
        
        Parameters
        ----------
        T_K : float
            Temperature in Kelvin
        r : float
            Epithermal index
            
        Returns
        -------
        float
            Total Westcott factor
        """
        return self.g(T_K) + r * self.s(T_K)


# Westcott g(T) data for common isotopes
# Coefficients from Westcott tables and IAEA compilations
# g(T) = a0 + a1*(T/T0) + a2*(T/T0)^2 + ...
WESTCOTT_DATA: Dict[str, WestcottFactors] = {}


def _init_westcott_data():
    """Initialize Westcott factor database."""
    
    # 1/v absorbers: g(T) = 1.0 always
    one_v_isotopes = [
        "Au-197", "Co-59", "Mn-55", "Na-23", "V-51", "Sc-45",
        "Fe-58", "Cu-63", "Zr-94", "Zr-96", "Mg-26", "Al-27",
    ]
    for iso in one_v_isotopes:
        WESTCOTT_DATA[iso] = WestcottFactors(
            isotope=iso,
            reaction="(n,g)",
            g_coefficients=(1.0,),
        )
    
    # Lu-176: Strong non-1/v behavior
    # g(T) polynomial fit from Westcott tables
    WESTCOTT_DATA["Lu-176"] = WestcottFactors(
        isotope="Lu-176",
        reaction="(n,g)",
        g_coefficients=(1.0, -0.0023, 0.000012),
        valid_range=(200, 600),
    )
    
    # Sm-149: Very strong non-1/v (resonance at 0.0973 eV)
    WESTCOTT_DATA["Sm-149"] = WestcottFactors(
        isotope="Sm-149",
        reaction="(n,g)",
        g_coefficients=(1.0, 0.0047, 0.000055),
        valid_range=(200, 500),
    )
    
    # Eu-151: Non-1/v with resonance at 0.461 eV
    WESTCOTT_DATA["Eu-151"] = WestcottFactors(
        isotope="Eu-151",
        reaction="(n,g)",
        g_coefficients=(1.0, 0.0015, 0.000008),
        valid_range=(200, 600),
    )
    
    # Cd-113: Very strong non-1/v (cutoff ~0.5 eV)
    WESTCOTT_DATA["Cd-113"] = WestcottFactors(
        isotope="Cd-113",
        reaction="(n,g)",
        g_coefficients=(1.0, -0.0031, 0.000018),
        valid_range=(200, 500),
    )
    
    # In-115: Resonance at 1.457 eV
    WESTCOTT_DATA["In-115"] = WestcottFactors(
        isotope="In-115",
        reaction="(n,g)",
        g_coefficients=(1.0, 0.0008, 0.000003),
        valid_range=(200, 600),
    )
    
    # Rh-103: Strong resonance at 1.26 eV
    WESTCOTT_DATA["Rh-103"] = WestcottFactors(
        isotope="Rh-103",
        reaction="(n,g)",
        g_coefficients=(1.0, 0.0012, 0.000006),
        valid_range=(200, 600),
    )
    
    # Pu-239: Fission, non-1/v
    WESTCOTT_DATA["Pu-239"] = WestcottFactors(
        isotope="Pu-239",
        reaction="(n,f)",
        g_coefficients=(1.055, -0.0018, 0.000009),
        valid_range=(200, 1000),
    )
    
    # U-235: Fission, slight non-1/v
    WESTCOTT_DATA["U-235"] = WestcottFactors(
        isotope="U-235",
        reaction="(n,f)",
        g_coefficients=(0.9780, 0.00015, 0.0000008),
        valid_range=(200, 1000),
    )
    
    # U-233: Fission
    WESTCOTT_DATA["U-233"] = WestcottFactors(
        isotope="U-233",
        reaction="(n,f)",
        g_coefficients=(0.9957, 0.00008, 0.0000004),
        valid_range=(200, 1000),
    )


# Initialize on module load
_init_westcott_data()


def get_westcott_g(isotope: str, T_K: float = 293.6) -> float:
    """
    Get Westcott g(T) factor for an isotope.
    
    Parameters
    ----------
    isotope : str
        Isotope identifier (e.g., "Au-197", "Lu-176")
    T_K : float
        Temperature in Kelvin (default: 293.6 K = 20.45°C)
        
    Returns
    -------
    float
        g(T) factor (1.0 for 1/v absorbers)
    """
    iso = isotope.replace(' ', '-')
    if iso in WESTCOTT_DATA:
        return WESTCOTT_DATA[iso].g(T_K)
    
    # Assume 1/v if unknown
    return 1.0


def get_westcott_factors(isotope: str) -> Optional[WestcottFactors]:
    """Get full Westcott factor data for an isotope."""
    iso = isotope.replace(' ', '-')
    return WESTCOTT_DATA.get(iso)


def list_non_1v_isotopes() -> List[str]:
    """List isotopes with significant non-1/v behavior."""
    return [iso for iso, data in WESTCOTT_DATA.items() 
            if len(data.g_coefficients) > 1 or data.g_coefficients[0] != 1.0]


# ==============================================================================
# Irradiation Parameters
# ==============================================================================

@dataclass
class TRIGAIrradiationParams:
    """
    Standard irradiation parameters for TRIGA reactor NAA.
    
    Attributes
    ----------
    irradiation_time_s : float
        Total irradiation time (seconds)
    decay_time_s : float
        Time between end of irradiation and start of counting (seconds)
    counting_time_s : float
        Duration of gamma spectrum acquisition (seconds)
    live_time_s : float, optional
        Detector live time (seconds), default = counting_time
    dead_time_fraction : float
        Fractional dead time (0-1)
    reactor_power_kW : float
        Reactor operating power (kW)
    position : str
        Irradiation position identifier
    flux_params : FluxParameters, optional
        Characterized flux parameters for this position
    """
    irradiation_time_s: float
    decay_time_s: float
    counting_time_s: float
    live_time_s: Optional[float] = None
    dead_time_fraction: float = 0.0
    reactor_power_kW: float = 1000.0  # 1 MW default for TRIGA
    position: str = ""
    flux_params: Optional[FluxParameters] = None
    
    def __post_init__(self):
        """Set defaults and validate."""
        if self.live_time_s is None:
            self.live_time_s = self.counting_time_s * (1 - self.dead_time_fraction)
    
    @property
    def irradiation_time_h(self) -> float:
        """Irradiation time in hours."""
        return self.irradiation_time_s / 3600.0
    
    @property
    def decay_time_h(self) -> float:
        """Decay time in hours."""
        return self.decay_time_s / 3600.0
    
    @property
    def counting_time_h(self) -> float:
        """Counting time in hours."""
        return self.counting_time_s / 3600.0


@dataclass 
class TripleMonitorResult:
    """
    Result from bare triple-monitor method for f and α determination.
    
    Uses Zr-94, Zr-96, and Au-197 to determine f and α simultaneously
    without requiring Cd covers (bare multi-monitor method).
    
    References: Guesmia et al. (2025), De Corte compilations
    """
    f: float
    f_uncertainty: float
    alpha: float
    alpha_uncertainty: float
    phi_thermal: float
    phi_epithermal: float
    phi_fast: Optional[float] = None
    monitors_used: List[str] = field(default_factory=list)
    position: str = ""
    convergence_iterations: int = 0
    
    def to_flux_parameters(self) -> FluxParameters:
        """Convert to FluxParameters object."""
        return FluxParameters(
            f=self.f,
            f_uncertainty=self.f_uncertainty,
            alpha=self.alpha,
            alpha_uncertainty=self.alpha_uncertainty,
            phi_thermal=self.phi_thermal,
            phi_epithermal=self.phi_epithermal,
            measurement_position=self.position,
        )


# Nuclear data for triple-monitor method (Zr-94, Zr-96, Au-197)
TRIPLE_MONITOR_DATA = {
    'Zr-94': {
        'reaction': 'Zr-94(n,g)Zr-95',
        'product': 'Zr95',
        'E_gamma_keV': 756.7,
        'half_life_s': 64.02 * 24 * 3600,  # 64.02 days
        'k0_Au': 1.10e-4,
        'Q0': 5.31,
        'E_res': 6260.0,  # eV
        'sigma_0': 0.0494,  # barns
    },
    'Zr-96': {
        'reaction': 'Zr-96(n,g)Zr-97',
        'product': 'Zr97',
        'E_gamma_keV': 743.3,
        'half_life_s': 16.9 * 3600,  # 16.9 hours
        'k0_Au': 1.24e-5,
        'Q0': 251.6,
        'E_res': 338.0,  # eV
        'sigma_0': 0.0229,  # barns
    },
    'Au-197': {
        'reaction': 'Au-197(n,g)Au-198',
        'product': 'Au198',
        'E_gamma_keV': 411.8,
        'half_life_s': 2.695 * 24 * 3600,  # 2.695 days
        'k0_Au': 1.0,  # Reference
        'Q0': 15.7,
        'E_res': 5.65,  # eV
        'sigma_0': 98.65,  # barns
    },
}

# Additional monitors for complete characterization
EXTENDED_MONITOR_DATA = {
    'Co-59': {
        'reaction': 'Co-59(n,g)Co-60',
        'product': 'Co60',
        'E_gamma_keV': 1332.5,
        'half_life_s': 5.2714 * 365.25 * 24 * 3600,  # 5.27 years
        'k0_Au': 1.320,
        'Q0': 1.99,
        'E_res': 136.0,  # eV
        'sigma_0': 37.18,  # barns
        'FCd': 1.0,
    },
    'Rb-85': {
        'reaction': 'Rb-85(n,g)Rb-86',
        'product': 'Rb86',
        'E_gamma_keV': 1077.0,
        'half_life_s': 18.63 * 24 * 3600,  # 18.63 days
        'k0_Au': 7.65e-4,
        'Q0': 14.8,
        'E_res': 839.0,  # eV
        'sigma_0': 0.485,  # barns
        'FCd': 1.0,
    },
    'Fe-58': {
        'reaction': 'Fe-58(n,g)Fe-59',
        'product': 'Fe59',
        'E_gamma_keV': 1099.2,
        'half_life_s': 44.5 * 24 * 3600,  # 44.5 days
        'k0_Au': 3.92e-5,
        'Q0': 0.78,
        'E_res': 357.0,  # eV
        'sigma_0': 1.28,  # barns
    },
    'Zn-64': {
        'reaction': 'Zn-64(n,g)Zn-65',
        'product': 'Zn65',
        'E_gamma_keV': 1115.5,
        'half_life_s': 244.3 * 24 * 3600,  # 244.3 days
        'k0_Au': 3.04e-4,
        'Q0': 1.91,
        'E_res': 2560.0,  # eV
        'sigma_0': 0.76,  # barns
    },
    # Fast neutron monitors
    'Fe-54': {
        'reaction': 'Fe-54(n,p)Mn-54',
        'product': 'Mn54',
        'E_gamma_keV': 834.8,
        'half_life_s': 312.29 * 24 * 3600,  # 312.29 days
        'threshold_MeV': 2.0,
        'is_threshold': True,
    },
    'Al-27': {
        'reaction': 'Al-27(n,p)Mg-27',
        'product': 'Mg27',
        'E_gamma_keV': 843.76,
        'half_life_s': 9.46 * 60,  # 9.46 min
        'threshold_MeV': 3.25,
        'is_threshold': True,
    },
}


def calculate_sdc_factors(
    half_life_s: float,
    t_irr_s: float,
    t_decay_s: float,
    t_count_s: float,
) -> Tuple[float, float, float]:
    """
    Calculate saturation (S), decay (D), and counting (C) factors.
    
    From the k0-NAA formalism:
        S = 1 - exp(-λ·t_irr)
        D = exp(-λ·t_decay)
        C = (1 - exp(-λ·t_count)) / (λ·t_count)
    
    Parameters
    ----------
    half_life_s : float
        Half-life of product isotope (seconds)
    t_irr_s : float
        Irradiation time (seconds)
    t_decay_s : float
        Decay time (seconds)
    t_count_s : float
        Counting time (seconds)
        
    Returns
    -------
    tuple
        (S, D, C) factors
    """
    if half_life_s <= 0:
        return 0.0, 0.0, 0.0
    
    lambda_decay = np.log(2) / half_life_s
    
    # Saturation factor
    S = 1.0 - np.exp(-lambda_decay * t_irr_s)
    
    # Decay factor
    D = np.exp(-lambda_decay * t_decay_s)
    
    # Counting factor (avoid divide by zero)
    if lambda_decay * t_count_s < 1e-6:
        C = 1.0  # Limit for short-lived isotopes
    else:
        C = (1.0 - np.exp(-lambda_decay * t_count_s)) / (lambda_decay * t_count_s)
    
    return S, D, C


def triple_monitor_method(
    activities: Dict[str, float],
    irradiation_params: TRIGAIrradiationParams,
    initial_alpha: float = 0.0,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> TripleMonitorResult:
    """
    Determine f and α using bare triple-monitor method.
    
    Uses Zr-94, Zr-96, and Au-197 monitors without Cd covers to
    simultaneously determine the thermal-to-epithermal ratio f
    and the epithermal shape parameter α.
    
    The method relies on the different Q0 and Er values of the
    three monitors to solve for both parameters iteratively.
    
    Parameters
    ----------
    activities : dict
        Dictionary of measured activities: {'Zr-94': A1, 'Zr-96': A2, 'Au-197': A3}
    irradiation_params : TRIGAIrradiationParams
        Irradiation parameters
    initial_alpha : float
        Initial guess for α
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance for α
        
    Returns
    -------
    TripleMonitorResult
        Characterized flux parameters
        
    References
    ----------
    Guesmia et al. (2025) J. Radioanal. Nucl. Chem. 334:7009-7018
    """
    # Validate inputs
    required = ['Zr-94', 'Zr-96', 'Au-197']
    for monitor in required:
        if monitor not in activities:
            raise ValueError(f"Missing required monitor: {monitor}")
    
    alpha = initial_alpha
    converged = False
    
    for iteration in range(max_iterations):
        # Calculate Q0(α) for each monitor
        Q0_Zr94 = calculate_Q0_alpha(
            TRIPLE_MONITOR_DATA['Zr-94']['Q0'],
            TRIPLE_MONITOR_DATA['Zr-94']['E_res'],
            alpha
        )
        Q0_Zr96 = calculate_Q0_alpha(
            TRIPLE_MONITOR_DATA['Zr-96']['Q0'],
            TRIPLE_MONITOR_DATA['Zr-96']['E_res'],
            alpha
        )
        Q0_Au = calculate_Q0_alpha(
            TRIPLE_MONITOR_DATA['Au-197']['Q0'],
            TRIPLE_MONITOR_DATA['Au-197']['E_res'],
            alpha
        )
        
        # Get activities and k0 factors
        A_Zr94 = activities['Zr-94']
        A_Zr96 = activities['Zr-96']
        A_Au = activities['Au-197']
        
        k0_Zr94 = TRIPLE_MONITOR_DATA['Zr-94']['k0_Au']
        k0_Zr96 = TRIPLE_MONITOR_DATA['Zr-96']['k0_Au']
        
        # Solve for f using ratio method
        # From the activity ratios, we can derive f
        # Using the k0 formalism: A ∝ k0 × (f + Q0(α)) × ε × φ_th
        
        # Ratio of Zr-94 to Au-197
        R1 = (A_Zr94 / k0_Zr94) / A_Au
        
        # Ratio of Zr-96 to Au-197
        R2 = (A_Zr96 / k0_Zr96) / A_Au
        
        # From these ratios, solve for f
        # R1 = (f + Q0_Zr94) / (f + Q0_Au)
        # R2 = (f + Q0_Zr96) / (f + Q0_Au)
        
        # This gives us:
        # f = (Q0_Au × R1 - Q0_Zr94) / (1 - R1) for Zr-94
        # f = (Q0_Au × R2 - Q0_Zr96) / (1 - R2) for Zr-96
        
        if abs(1 - R1) > 1e-10:
            f_Zr94 = (Q0_Au * R1 - Q0_Zr94) / (1 - R1)
        else:
            f_Zr94 = 20.0  # Default
            
        if abs(1 - R2) > 1e-10:
            f_Zr96 = (Q0_Au * R2 - Q0_Zr96) / (1 - R2)
        else:
            f_Zr96 = 20.0
        
        # Average f
        f = 0.5 * (f_Zr94 + f_Zr96)
        
        # Update α using the ratio of ratios
        # log(R1/R2) is related to α through the E_res values
        E_res_ratio = TRIPLE_MONITOR_DATA['Zr-94']['E_res'] / TRIPLE_MONITOR_DATA['Zr-96']['E_res']
        
        if R1 > 0 and R2 > 0 and E_res_ratio != 1:
            new_alpha = np.log(R1 / R2) / np.log(E_res_ratio)
            # Damped update
            new_alpha = 0.5 * alpha + 0.5 * new_alpha
            # Bound to reasonable range
            new_alpha = np.clip(new_alpha, -0.3, 0.3)
        else:
            new_alpha = alpha
        
        # Check convergence
        if abs(new_alpha - alpha) < tolerance:
            converged = True
            alpha = new_alpha
            break
            
        alpha = new_alpha
    
    # Calculate thermal flux from Au monitor
    # φ_th = A_Au / (N × σ_0 × (1 + Q0(α)/f) × S × D)
    # For normalized calculation, use relative values
    
    # Thermal flux (relative, would need absolute calibration)
    sigma_0_Au = TRIPLE_MONITOR_DATA['Au-197']['sigma_0'] * 1e-24  # cm²
    
    S, D, C = calculate_sdc_factors(
        TRIPLE_MONITOR_DATA['Au-197']['half_life_s'],
        irradiation_params.irradiation_time_s,
        irradiation_params.decay_time_s,
        irradiation_params.counting_time_s,
    )
    
    # Simplified flux calculation (needs sample mass for absolute)
    phi_thermal = A_Au / (S * D * sigma_0_Au * (1 + Q0_Au / f)) if (S * D) > 0 else 0
    phi_epithermal = phi_thermal / f if f > 0 else 0
    
    # Uncertainty estimation
    f_uncertainty = abs(f_Zr94 - f_Zr96) / 2 if abs(f_Zr94 - f_Zr96) > 0 else f * 0.1
    alpha_uncertainty = 0.02 if converged else 0.05
    
    return TripleMonitorResult(
        f=f,
        f_uncertainty=f_uncertainty,
        alpha=alpha,
        alpha_uncertainty=alpha_uncertainty,
        phi_thermal=phi_thermal,
        phi_epithermal=phi_epithermal,
        monitors_used=['Zr-94', 'Zr-96', 'Au-197'],
        position=irradiation_params.position,
        convergence_iterations=iteration + 1,
    )


def validate_triga_flux_params(
    params: FluxParameters,
    reactor_type: str = "TRIGA",
) -> Dict[str, Union[bool, str]]:
    """
    Validate flux parameters against expected TRIGA ranges.
    
    Parameters
    ----------
    params : FluxParameters
        Measured flux parameters
    reactor_type : str
        Reactor type for expected ranges
        
    Returns
    -------
    dict
        Validation results and warnings
    """
    results = {'valid': True, 'warnings': [], 'info': []}
    
    # Expected ranges for TRIGA reactors
    # Based on literature values from Pavia, Brazilian, ENEA TRIGAs
    if reactor_type.upper() == "TRIGA":
        # f typically 10-50 for TRIGA, can be lower in central thimble
        if params.f < 2:
            results['warnings'].append(
                f"f = {params.f:.1f} is very low; may indicate hard spectrum"
            )
        elif params.f > 100:
            results['warnings'].append(
                f"f = {params.f:.1f} is very high; unusual for TRIGA"
            )
        else:
            results['info'].append(f"f = {params.f:.1f} is within typical TRIGA range")
        
        # α typically -0.1 to +0.1 for TRIGA
        if abs(params.alpha) > 0.2:
            results['warnings'].append(
                f"α = {params.alpha:.3f} is larger than typical; check monitors"
            )
        else:
            results['info'].append(f"α = {params.alpha:.3f} is within typical range")
        
        # Check self-consistency
        if params.f_uncertainty / params.f > 0.5 if params.f > 0 else True:
            results['warnings'].append(
                "High relative uncertainty in f; results may not be reliable"
            )
    
    # Set overall validity
    results['valid'] = len(results['warnings']) == 0
    
    return results


class TRIGAk0Workflow:
    """
    Complete k0-standardization workflow for TRIGA reactor NAA.
    
    This class provides a comprehensive k0-NAA analysis pipeline:
    1. Flux characterization (f, α) using Cd-ratio or bare multi-monitor
    2. Peak area to activity conversion with SDC corrections
    3. Concentration calculation using k0 factors
    4. Uncertainty propagation and validation
    
    Example
    -------
    >>> workflow = TRIGAk0Workflow(position="Lazy Susan #27")
    >>> workflow.set_irradiation_params(t_irr_h=4.0, t_decay_h=24.0, t_count_h=2.0)
    >>> workflow.characterize_flux_cd_ratio(bare_activities, cd_activities)
    >>> results = workflow.analyze_sample(peak_areas, sample_mass_mg=100)
    
    References
    ----------
    - Di Luzio et al. (2017) J. Radioanal. Nucl. Chem. 312:75-80
    - Menezes et al. (2023) J. Radioanal. Nucl. Chem. 332:3445-3456
    """
    
    def __init__(self, position: str = "", reactor_power_kW: float = 1000.0):
        """
        Initialize the workflow.
        
        Parameters
        ----------
        position : str
            Irradiation position identifier
        reactor_power_kW : float
            Reactor operating power
        """
        self.position = position
        self.reactor_power_kW = reactor_power_kW
        self.irradiation_params: Optional[TRIGAIrradiationParams] = None
        self.flux_params: Optional[FluxParameters] = None
        self.cd_analyzer: Optional[CdRatioAnalyzer] = None
        self._results: List[Dict] = []
    
    def set_irradiation_params(
        self,
        t_irr_s: Optional[float] = None,
        t_irr_h: Optional[float] = None,
        t_decay_s: Optional[float] = None,
        t_decay_h: Optional[float] = None,
        t_count_s: Optional[float] = None,
        t_count_h: Optional[float] = None,
        dead_time_fraction: float = 0.0,
    ) -> None:
        """
        Set irradiation timing parameters.
        
        Times can be provided in seconds or hours.
        """
        # Convert hours to seconds if provided
        if t_irr_h is not None:
            t_irr_s = t_irr_h * 3600
        if t_decay_h is not None:
            t_decay_s = t_decay_h * 3600
        if t_count_h is not None:
            t_count_s = t_count_h * 3600
        
        if t_irr_s is None or t_decay_s is None or t_count_s is None:
            raise ValueError("Must provide irradiation, decay, and counting times")
        
        self.irradiation_params = TRIGAIrradiationParams(
            irradiation_time_s=t_irr_s,
            decay_time_s=t_decay_s,
            counting_time_s=t_count_s,
            dead_time_fraction=dead_time_fraction,
            reactor_power_kW=self.reactor_power_kW,
            position=self.position,
        )
    
    def characterize_flux_cd_ratio(
        self,
        bare_activities: Dict[str, float],
        cd_activities: Dict[str, float],
        uncertainties_bare: Optional[Dict[str, float]] = None,
        uncertainties_cd: Optional[Dict[str, float]] = None,
    ) -> FluxParameters:
        """
        Characterize flux using Cd-ratio multi-monitor method.
        
        Parameters
        ----------
        bare_activities : dict
            Activities of bare monitors {element: activity_Bq}
        cd_activities : dict
            Activities of Cd-covered monitors {element: activity_Bq}
        uncertainties_bare : dict, optional
            Relative uncertainties in bare activities
        uncertainties_cd : dict, optional
            Relative uncertainties in Cd activities
            
        Returns
        -------
        FluxParameters
            Characterized f and α with uncertainties
        """
        self.cd_analyzer = CdRatioAnalyzer(position=self.position)
        
        # Add measurements for matching elements
        for element in bare_activities:
            if element in cd_activities and element in STANDARD_MONITORS:
                unc_bare = uncertainties_bare.get(element, 0.05) if uncertainties_bare else 0.05
                unc_cd = uncertainties_cd.get(element, 0.07) if uncertainties_cd else 0.07
                
                self.cd_analyzer.add_measurement(
                    element=element,
                    activity_bare=bare_activities[element],
                    activity_cd=cd_activities[element],
                    uncertainty_bare=unc_bare,
                    uncertainty_cd=unc_cd,
                )
        
        self.flux_params = self.cd_analyzer.characterize_flux()
        
        if self.irradiation_params:
            self.irradiation_params.flux_params = self.flux_params
        
        return self.flux_params
    
    def characterize_flux_triple_monitor(
        self,
        activities: Dict[str, float],
    ) -> FluxParameters:
        """
        Characterize flux using bare triple-monitor method.
        
        Uses Zr-94, Zr-96, and Au-197 without Cd covers.
        
        Parameters
        ----------
        activities : dict
            Measured activities {isotope: activity_Bq}
            
        Returns
        -------
        FluxParameters
            Characterized f and α
        """
        if self.irradiation_params is None:
            raise ValueError("Set irradiation parameters first")
        
        result = triple_monitor_method(activities, self.irradiation_params)
        self.flux_params = result.to_flux_parameters()
        self.irradiation_params.flux_params = self.flux_params
        
        return self.flux_params
    
    def set_flux_params(
        self,
        f: float,
        alpha: float = 0.0,
        f_uncertainty: float = 1.0,
        alpha_uncertainty: float = 0.01,
    ) -> None:
        """
        Set flux parameters directly (if known from previous characterization).
        """
        self.flux_params = FluxParameters(
            f=f,
            f_uncertainty=f_uncertainty,
            alpha=alpha,
            alpha_uncertainty=alpha_uncertainty,
            measurement_position=self.position,
        )
        if self.irradiation_params:
            self.irradiation_params.flux_params = self.flux_params
    
    def calculate_concentration(
        self,
        peak_area: float,
        peak_area_unc: float,
        element: str,
        isotope: str,
        k0_factor: float,
        Q0: float,
        E_res: float,
        half_life_s: float,
        efficiency: float,
        efficiency_unc: float,
        sample_mass_g: float,
        comparator_peak_area: float,
        comparator_mass_ug: float,
        comparator_efficiency: float,
    ) -> Dict[str, float]:
        """
        Calculate element concentration using k0-NAA formalism.
        
        Implements the Høgdahl convention:
            ρ = [Np/(tm·S·D·C·W)]_a / [Np/(tm·S·D·C·w)]_c 
                × 1/k0 × [f + Q0,c(α)]·εp,c / [f + Q0,a(α)]·εp,a
        
        Returns
        -------
        dict
            {'concentration_mg_kg': ..., 'uncertainty': ..., 'LOD': ...}
        """
        if self.flux_params is None:
            raise ValueError("Characterize flux first")
        if self.irradiation_params is None:
            raise ValueError("Set irradiation parameters first")
        
        f = self.flux_params.f
        alpha = self.flux_params.alpha
        
        # Calculate Q0(α)
        Q0_alpha = calculate_Q0_alpha(Q0, E_res, alpha)
        
        # Comparator (Au) Q0(α)
        Q0_Au = calculate_Q0_alpha(15.7, 5.65, alpha)  # Au-198 values
        
        # SDC factors for analyte
        S_a, D_a, C_a = calculate_sdc_factors(
            half_life_s,
            self.irradiation_params.irradiation_time_s,
            self.irradiation_params.decay_time_s,
            self.irradiation_params.counting_time_s,
        )
        
        # SDC for comparator (Au-198)
        S_c, D_c, C_c = calculate_sdc_factors(
            2.695 * 24 * 3600,  # Au-198 half-life
            self.irradiation_params.irradiation_time_s,
            self.irradiation_params.decay_time_s,
            self.irradiation_params.counting_time_s,
        )
        
        # Counting time (same for both)
        tm = self.irradiation_params.live_time_s
        
        # Calculate concentration (mg/kg)
        numerator = peak_area / (tm * S_a * D_a * C_a * sample_mass_g)
        denominator = comparator_peak_area / (tm * S_c * D_c * C_c * comparator_mass_ug * 1e-6)
        
        flux_term = (f + Q0_Au) * comparator_efficiency / ((f + Q0_alpha) * efficiency)
        
        concentration = numerator / denominator * flux_term / k0_factor * 1e6  # mg/kg
        
        # Uncertainty propagation (simplified)
        rel_unc = np.sqrt(
            (peak_area_unc / peak_area)**2 +
            (efficiency_unc / efficiency)**2 +
            (self.flux_params.f_uncertainty / f)**2 * 0.1  # Partial sensitivity
        )
        uncertainty = concentration * rel_unc
        
        # Limit of detection (3σ above background)
        LOD = 3 * uncertainty
        
        result = {
            'element': element,
            'isotope': isotope,
            'concentration_mg_kg': concentration,
            'uncertainty': uncertainty,
            'LOD': LOD,
            'k0_factor': k0_factor,
            'Q0_alpha': Q0_alpha,
            'SDC_product': S_a * D_a * C_a,
        }
        
        self._results.append(result)
        return result
    
    def summary(self) -> str:
        """Generate a text summary of the analysis."""
        lines = [
            "=" * 60,
            "TRIGA k0-NAA Workflow Summary",
            "=" * 60,
            f"Position: {self.position}",
        ]
        
        if self.irradiation_params:
            lines.extend([
                f"Reactor power: {self.reactor_power_kW} kW",
                f"Irradiation: {self.irradiation_params.irradiation_time_h:.2f} h",
                f"Decay: {self.irradiation_params.decay_time_h:.2f} h",
                f"Counting: {self.irradiation_params.counting_time_h:.2f} h",
            ])
        
        if self.flux_params:
            lines.extend([
                "",
                "Flux Parameters:",
                f"  f = {self.flux_params.f:.1f} ± {self.flux_params.f_uncertainty:.1f}",
                f"  α = {self.flux_params.alpha:.4f} ± {self.flux_params.alpha_uncertainty:.4f}",
                f"  Spectrum: {self.flux_params.spectrum_description}",
            ])
        
        if self._results:
            lines.extend([
                "",
                "Concentration Results:",
                "-" * 40,
            ])
            for r in self._results:
                lines.append(
                    f"  {r['element']:6s}: {r['concentration_mg_kg']:.2f} ± "
                    f"{r['uncertainty']:.2f} mg/kg"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def to_dataframe(self):
        """Export results to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        return pd.DataFrame(self._results)
    
    def validate(self) -> Dict[str, Union[bool, List[str]]]:
        """Validate the workflow and flux parameters."""
        if self.flux_params is None:
            return {'valid': False, 'warnings': ['Flux not characterized']}
        
        return validate_triga_flux_params(self.flux_params)
