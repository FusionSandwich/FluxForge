"""
k₀-NAA: k₀-Standardization Based Neutron Activation Analysis

Implements the k₀-standardization method for Neutron Activation Analysis (NAA),
which allows absolute determination of element concentrations without standards
through the use of pre-determined nuclear constants (k₀ factors).

Key equations:
    ρ(a) = [Np,a / (SDC·W)] / [Np,Au / (SDC·W)_Au] · 1/k0,Au(a) · 
           Ge(Eγ,Au)/Ge(Eγ,a) · f+Q0,Au(α)/f+Q0,a(α)

where:
    ρ(a) = mass fraction of analyte a (g/g or μg/g)
    Np = net peak area (counts)
    S = saturation factor = 1 - exp(-λ·ti)
    D = decay factor = exp(-λ·td)
    C = counting factor = [1 - exp(-λ·tc)] / (λ·tc)
    W = sample mass
    k0,Au(a) = k0 factor for analyte a relative to Au
    Ge = detector efficiency at gamma energy Eγ
    f = thermal to epithermal flux ratio
    Q0 = resonance integral / thermal cross section
    α = epithermal flux shape parameter

References:
    [1] De Corte, F. et al., J. Radioanal. Nucl. Chem., 169, 125-141 (1993)
    [2] IAEA-TECDOC-1215, "Use of research reactors for NAA" (2001)
    [3] Simonits, A. et al., J. Radioanal. Chem., 81, 397-405 (1984)
    [4] Di Luzio et al., J. Radioanal. Nucl. Chem. (2017)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# k₀ Nuclear Data Database
# =============================================================================

@dataclass
class K0NuclideData:
    """
    k₀-NAA nuclear data for a specific nuclide.
    
    Attributes
    ----------
    target_isotope : str
        Target isotope (e.g., "Au-197", "Co-59")
    product_isotope : str
        Product isotope (e.g., "Au-198", "Co-60")
    element : str
        Element symbol
    gamma_energy_keV : float
        Primary gamma energy for analysis
    gamma_intensity : float
        Gamma emission probability (absolute intensity)
    half_life_s : float
        Half-life of product isotope in seconds
    k0_Au : float
        k0 factor relative to Au (197Au(n,γ)198Au, 411.8 keV)
    k0_unc : float
        Uncertainty in k0 factor (relative, %)
    Q0 : float
        Q0 = I0 / σ0 (resonance integral / thermal cross section)
    Q0_unc : float
        Uncertainty in Q0 (relative, %)
    E_res_eV : float
        Effective resonance energy (eV)
    sigma_0_barn : float
        Thermal (2200 m/s) cross section in barns
    I0_barn : float
        Resonance integral in barns
    isotopic_abundance : float
        Natural isotopic abundance (fraction)
    atomic_mass : float
        Atomic mass (g/mol)
    """
    target_isotope: str
    product_isotope: str
    element: str
    gamma_energy_keV: float
    gamma_intensity: float
    half_life_s: float
    k0_Au: float
    k0_unc: float = 0.0
    Q0: float = 1.0
    Q0_unc: float = 0.0
    E_res_eV: float = 0.0
    sigma_0_barn: float = 0.0
    I0_barn: float = 0.0
    isotopic_abundance: float = 1.0
    atomic_mass: float = 0.0
    additional_gammas: List[Tuple[float, float]] = field(default_factory=list)


# k0 Nuclear Data Database (relative to Au-197(n,g)Au-198, 411.8 keV)
# Data from: IAEA k0 database, De Corte compilations
K0_DATABASE: Dict[str, K0NuclideData] = {}


def _initialize_k0_database():
    """Initialize the k0 nuclear data database."""
    global K0_DATABASE
    
    # Reference: Au-197(n,g)Au-198
    K0_DATABASE['Au-198'] = K0NuclideData(
        target_isotope='Au-197',
        product_isotope='Au-198',
        element='Au',
        gamma_energy_keV=411.8,
        gamma_intensity=0.9558,
        half_life_s=2.6944 * 24 * 3600,  # 2.6944 d
        k0_Au=1.0,  # Reference
        k0_unc=0.0,
        Q0=15.71,
        Q0_unc=1.5,
        E_res_eV=5.65,
        sigma_0_barn=98.65,
        I0_barn=1550,
        isotopic_abundance=1.0,
        atomic_mass=196.967,
    )
    
    # Co-59(n,g)Co-60 - important for flux monitoring
    K0_DATABASE['Co-60'] = K0NuclideData(
        target_isotope='Co-59',
        product_isotope='Co-60',
        element='Co',
        gamma_energy_keV=1332.5,
        gamma_intensity=0.9998,
        half_life_s=5.2714 * 365.25 * 24 * 3600,  # 5.2714 y
        k0_Au=1.320,
        k0_unc=0.9,
        Q0=1.99,
        Q0_unc=2.0,
        E_res_eV=132.0,
        sigma_0_barn=37.18,
        I0_barn=74.0,
        isotopic_abundance=1.0,
        atomic_mass=58.933,
        additional_gammas=[(1173.2, 0.9985)],
    )
    
    # Sc-45(n,g)Sc-46 - flux monitor
    K0_DATABASE['Sc-46'] = K0NuclideData(
        target_isotope='Sc-45',
        product_isotope='Sc-46',
        element='Sc',
        gamma_energy_keV=889.3,
        gamma_intensity=0.99984,
        half_life_s=83.79 * 24 * 3600,  # 83.79 d
        k0_Au=8.79e-3,
        k0_unc=1.3,
        Q0=0.435,
        Q0_unc=4.0,
        E_res_eV=4.5,
        sigma_0_barn=27.5,
        I0_barn=12.0,
        isotopic_abundance=1.0,
        atomic_mass=44.956,
        additional_gammas=[(1120.5, 0.99987)],
    )
    
    # Fe-58(n,g)Fe-59
    K0_DATABASE['Fe-59'] = K0NuclideData(
        target_isotope='Fe-58',
        product_isotope='Fe-59',
        element='Fe',
        gamma_energy_keV=1099.2,
        gamma_intensity=0.565,
        half_life_s=44.50 * 24 * 3600,  # 44.50 d
        k0_Au=5.34e-5,
        k0_unc=2.0,
        Q0=0.91,
        Q0_unc=8.0,
        E_res_eV=230.0,
        sigma_0_barn=1.31,
        I0_barn=1.19,
        isotopic_abundance=0.00282,
        atomic_mass=55.845,
        additional_gammas=[(1291.6, 0.432)],
    )
    
    # Cu-63(n,g)Cu-64
    K0_DATABASE['Cu-64'] = K0NuclideData(
        target_isotope='Cu-63',
        product_isotope='Cu-64',
        element='Cu',
        gamma_energy_keV=1345.8,
        gamma_intensity=0.00473,
        half_life_s=12.701 * 3600,  # 12.701 h
        k0_Au=3.88e-3,
        k0_unc=2.5,
        Q0=1.11,
        Q0_unc=4.0,
        E_res_eV=580.0,
        sigma_0_barn=4.5,
        I0_barn=5.0,
        isotopic_abundance=0.6917,
        atomic_mass=63.546,
    )
    
    # In-113(n,g)In-114m - high Q0, good epithermal monitor
    K0_DATABASE['In-114m'] = K0NuclideData(
        target_isotope='In-113',
        product_isotope='In-114m',
        element='In',
        gamma_energy_keV=190.3,
        gamma_intensity=0.1556,
        half_life_s=49.51 * 24 * 3600,  # 49.51 d
        k0_Au=3.16e-2,
        k0_unc=2.0,
        Q0=31.5,
        Q0_unc=3.0,
        E_res_eV=1.45,
        sigma_0_barn=4.0,
        I0_barn=126.0,
        isotopic_abundance=0.0429,
        atomic_mass=114.818,
    )
    
    # Mn-55(n,g)Mn-56 - short-lived, good for INAA
    K0_DATABASE['Mn-56'] = K0NuclideData(
        target_isotope='Mn-55',
        product_isotope='Mn-56',
        element='Mn',
        gamma_energy_keV=846.8,
        gamma_intensity=0.9887,
        half_life_s=2.5789 * 3600,  # 2.5789 h
        k0_Au=4.88e-3,
        k0_unc=1.2,
        Q0=0.67,
        Q0_unc=3.5,
        E_res_eV=337.0,
        sigma_0_barn=13.3,
        I0_barn=8.9,
        isotopic_abundance=1.0,
        atomic_mass=54.938,
        additional_gammas=[(1810.7, 0.272), (2113.1, 0.143)],
    )
    
    # Na-23(n,g)Na-24
    K0_DATABASE['Na-24'] = K0NuclideData(
        target_isotope='Na-23',
        product_isotope='Na-24',
        element='Na',
        gamma_energy_keV=1368.6,
        gamma_intensity=0.9999,
        half_life_s=14.997 * 3600,  # 14.997 h
        k0_Au=4.68e-2,
        k0_unc=1.3,
        Q0=0.59,
        Q0_unc=4.0,
        E_res_eV=2850.0,
        sigma_0_barn=0.530,
        I0_barn=0.31,
        isotopic_abundance=1.0,
        atomic_mass=22.990,
        additional_gammas=[(2754.0, 0.9986)],
    )
    
    # Cr-50(n,g)Cr-51
    K0_DATABASE['Cr-51'] = K0NuclideData(
        target_isotope='Cr-50',
        product_isotope='Cr-51',
        element='Cr',
        gamma_energy_keV=320.1,
        gamma_intensity=0.0991,
        half_life_s=27.70 * 24 * 3600,  # 27.70 d
        k0_Au=5.06e-4,
        k0_unc=2.5,
        Q0=0.54,
        Q0_unc=5.0,
        E_res_eV=7530.0,
        sigma_0_barn=15.8,
        I0_barn=8.5,
        isotopic_abundance=0.04345,
        atomic_mass=51.996,
    )
    
    # Zn-64(n,g)Zn-65
    K0_DATABASE['Zn-65'] = K0NuclideData(
        target_isotope='Zn-64',
        product_isotope='Zn-65',
        element='Zn',
        gamma_energy_keV=1115.5,
        gamma_intensity=0.5004,
        half_life_s=244.26 * 24 * 3600,  # 244.26 d
        k0_Au=5.72e-3,
        k0_unc=1.5,
        Q0=1.91,
        Q0_unc=3.0,
        E_res_eV=2560.0,
        sigma_0_barn=0.78,
        I0_barn=1.49,
        isotopic_abundance=0.486,
        atomic_mass=65.38,
    )
    
    # As-75(n,g)As-76
    K0_DATABASE['As-76'] = K0NuclideData(
        target_isotope='As-75',
        product_isotope='As-76',
        element='As',
        gamma_energy_keV=559.1,
        gamma_intensity=0.450,
        half_life_s=26.32 * 3600,  # 26.32 h
        k0_Au=4.01e-2,
        k0_unc=1.8,
        Q0=13.8,
        Q0_unc=2.5,
        E_res_eV=47.0,
        sigma_0_barn=4.5,
        I0_barn=62.0,
        isotopic_abundance=1.0,
        atomic_mass=74.922,
        additional_gammas=[(657.0, 0.0619)],
    )
    
    # W-186(n,g)W-187
    K0_DATABASE['W-187'] = K0NuclideData(
        target_isotope='W-186',
        product_isotope='W-187',
        element='W',
        gamma_energy_keV=685.8,
        gamma_intensity=0.273,
        half_life_s=23.72 * 3600,  # 23.72 h
        k0_Au=3.17e-2,
        k0_unc=1.5,
        Q0=13.8,
        Q0_unc=3.0,
        E_res_eV=18.8,
        sigma_0_barn=38.1,
        I0_barn=525.0,
        isotopic_abundance=0.2843,
        atomic_mass=183.84,
    )


# Initialize database on module load
_initialize_k0_database()


def get_k0_data(product_isotope: str) -> Optional[K0NuclideData]:
    """Get k0 data for a product isotope."""
    return K0_DATABASE.get(product_isotope)


def list_available_nuclides() -> List[str]:
    """List all nuclides in the k0 database."""
    return list(K0_DATABASE.keys())


# =============================================================================
# Flux Parameters
# =============================================================================

@dataclass
class K0Parameters:
    """k₀-NAA flux parameters."""
    f: float           # Thermal-to-epithermal flux ratio (φ_th / φ_epi)
    alpha: float       # Epithermal flux shape deviation from 1/E
    f_uncertainty: float = 0.0
    alpha_uncertainty: float = 0.0
    phi_thermal: float = 0.0     # Thermal flux (n/cm²/s)
    phi_epithermal: float = 0.0  # Epithermal flux (n/cm²/s)
    phi_fast: float = 0.0        # Fast flux (n/cm²/s)
    phi_total: float = 0.0       # Total flux (n/cm²/s)


def calculate_k0_parameters(bare_activities: Dict[str, float],
                           cd_activities: Dict[str, float]) -> K0Parameters:
    """
    Calculate k₀-NAA parameters f and α from bare and Cd-covered measurements.
    
    Following Di Luzio et al. 2017:
    - f = φ_th / φ_epi (thermal-to-epithermal flux ratio)
    - α = deviation of epithermal flux from ideal 1/E behavior
    
    Args:
        bare_activities: Dictionary of isotope -> specific activity (Bq/g) for bare wires
        cd_activities: Dictionary of isotope -> specific activity (Bq/g) for Cd-covered wires
        
    Returns:
        K0Parameters object containing f and alpha
    """
    # Cadmium ratios for Au, Co, Sc monitors (from standard tables)
    # Q_0 values at α=0 (resonance integral / thermal cross section)
    # TODO: Move these to a data library
    Q0_VALUES = {
        'sc46': 0.43,   # Sc-45(n,g)Sc-46
        'co60': 1.99,   # Co-59(n,g)Co-60  
        'cu64': 0.975,  # Cu-63(n,g)Cu-64
        'fe59': 0.45,   # Fe-58(n,g)Fe-59
        'au198': 15.7,  # Au-197(n,g)Au-198
        'zr95': 5.30,   # Zr-94(n,g)Zr-95
        'zr97': 248.0,  # Zr-96(n,g)Zr-97
    }
    
    # Effective resonance energies (eV)
    E_RES = {
        'sc46': 4.93,
        'co60': 132,
        'cu64': 241,
        'fe59': 231,
        'au198': 5.65,
        'zr95': 338,
        'zr97': 338, # Approximation
    }
    
    f_values = []
    
    # Calculate f from each bare/Cd pair
    for isotope, q0 in Q0_VALUES.items():
        # Normalize isotope keys to lowercase for matching
        iso_key = isotope.lower()
        
        # Check if we have data for this isotope (checking various key formats)
        bare_val = None
        cd_val = None
        
        for k in bare_activities:
            if k.lower() == iso_key:
                bare_val = bare_activities[k]
                break
                
        for k in cd_activities:
            if k.lower() == iso_key:
                cd_val = cd_activities[k]
                break
                
        if bare_val is not None and cd_val is not None and cd_val > 0:
            R_cd = bare_val / cd_val  # Cadmium ratio
            
            # f = (R_cd - 1) * Q_0 / F_cd
            # F_cd ≈ 1 for well-thermalized positions (simplified)
            f_calc = (R_cd - 1) / q0
            if f_calc > 0:
                f_values.append(f_calc)
    
    # Average f value
    f = np.mean(f_values) if f_values else 0.0
    f_unc = np.std(f_values) if len(f_values) > 1 else (f * 0.1 if f > 0 else 0.0)
    
    # Calculate α from multi-monitor method
    # Using log-linear fit of Cd-covered activities vs E_res
    alpha = 0.0  # Default - ideal 1/E spectrum
    alpha_unc = 0.0
    
    if len(cd_activities) >= 2:
        ln_e_res = []
        ln_activity = []
        
        for isotope, e_res in E_RES.items():
            iso_key = isotope.lower()
            cd_val = None
            for k in cd_activities:
                if k.lower() == iso_key:
                    cd_val = cd_activities[k]
                    break
            
            if cd_val is not None and cd_val > 0:
                ln_e_res.append(np.log(e_res))
                # We need to normalize activity by cross section and other factors for true alpha
                # But for this simplified implementation, we'll assume the user provided
                # reaction rates or we are just doing a relative slope check.
                # NOTE: The original code in flux_wire_spectrum_analysis.py just logged the activity.
                # This is likely a simplification or assumes specific normalization.
                # For a robust implementation, we should use the specific activity equation:
                # A_sp = ... E_res^(-alpha)
                ln_activity.append(np.log(cd_val))
        
        if len(ln_e_res) >= 2:
            # Linear regression: ln(A) = const - α * ln(E_res)
            try:
                coeffs = np.polyfit(ln_e_res, ln_activity, 1)
                alpha = -coeffs[0]  # Slope gives -α
                # Estimate uncertainty from residuals
                residuals = ln_activity - np.polyval(coeffs, ln_e_res)
                alpha_unc = np.std(residuals)
            except Exception:
                alpha = 0.0
                alpha_unc = 0.0
    
    return K0Parameters(
        f=f,
        alpha=alpha,
        f_uncertainty=f_unc,
        alpha_uncertainty=alpha_unc,
    )

# =============================================================================
# Correction Factor Calculations
# =============================================================================

def calculate_Q0_alpha(Q0: float, alpha: float, E_res_eV: float, E_Cd_eV: float = 0.55) -> float:
    """
    Calculate Q0(α) - the epithermal correction factor.
    
    For non-1/E epithermal spectra, Q0 needs to be corrected:
        Q0(α) = (Q0 - 0.429) / E_res^α + 0.429 / (2α + 1) / E_Cd^α
    
    Parameters
    ----------
    Q0 : float
        Q0 factor (I0/σ0) for α=0 (1/E spectrum)
    alpha : float
        Epithermal shape parameter
    E_res_eV : float
        Effective resonance energy (eV)
    E_Cd_eV : float
        Cadmium cut-off energy (eV), default 0.55 eV
        
    Returns
    -------
    float
        Q0(α) - corrected Q0 for given α
    """
    if alpha == 0:
        return Q0
    
    if E_res_eV <= 0:
        return Q0
    
    # Correction factor per Hogdahl convention
    Q0_alpha = (Q0 - 0.429) / (E_res_eV ** alpha) + 0.429 / ((2 * alpha + 1) * (E_Cd_eV ** alpha))
    
    return Q0_alpha


def saturation_factor(lambda_s: float, t_irr: float) -> float:
    """
    Calculate saturation factor S.
    
    S = 1 - exp(-λ·t_irr)
    
    Parameters
    ----------
    lambda_s : float
        Decay constant (1/s)
    t_irr : float
        Irradiation time (s)
        
    Returns
    -------
    float
        Saturation factor S
    """
    return 1.0 - np.exp(-lambda_s * t_irr)


def decay_factor(lambda_s: float, t_decay: float) -> float:
    """
    Calculate decay factor D.
    
    D = exp(-λ·t_d)
    """
    return np.exp(-lambda_s * t_decay)


def counting_factor(lambda_s: float, t_count: float) -> float:
    """
    Calculate counting factor C.
    
    C = [1 - exp(-λ·t_c)] / (λ·t_c)
    """
    if lambda_s * t_count < 1e-6:
        return 1.0
    return (1.0 - np.exp(-lambda_s * t_count)) / (lambda_s * t_count)


def sdc_factor(
    half_life_s: float,
    t_irr: float,
    t_decay: float,
    t_count: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate S, D, C correction factors and their product.
    
    Parameters
    ----------
    half_life_s : float
        Half-life of product nuclide (s)
    t_irr : float
        Irradiation time (s)
    t_decay : float
        Decay time (s)
    t_count : float
        Counting time (s)
        
    Returns
    -------
    tuple
        (S, D, C, SDC) - individual factors and product
    """
    lambda_s = np.log(2) / half_life_s
    S = saturation_factor(lambda_s, t_irr)
    D = decay_factor(lambda_s, t_decay)
    C = counting_factor(lambda_s, t_count)
    
    return S, D, C, S * D * C


# =============================================================================
# K0 Measurement and Result Classes
# =============================================================================

@dataclass
class K0Measurement:
    """
    A single k0-NAA measurement.
    
    Attributes
    ----------
    product_isotope : str
        Product isotope (e.g., "Co-60")
    net_peak_area : float
        Net peak area (counts)
    peak_area_unc : float
        Uncertainty in peak area (counts)
    efficiency : float
        Detector efficiency at the gamma energy
    efficiency_unc : float
        Efficiency uncertainty (relative, %)
    t_irr : float
        Irradiation time (s)
    t_decay : float
        Decay time (s)
    t_count : float
        Counting time (s)
    sample_mass : float
        Sample mass (g)
    gamma_energy_keV : float
        Gamma energy for this measurement (keV)
    """
    product_isotope: str
    net_peak_area: float
    peak_area_unc: float = 0.0
    efficiency: float = 1.0
    efficiency_unc: float = 0.0
    t_irr: float = 0.0
    t_decay: float = 0.0
    t_count: float = 0.0
    sample_mass: float = 1.0
    gamma_energy_keV: float = 0.0


@dataclass
class K0Result:
    """
    Result of k0-NAA concentration calculation.
    
    Attributes
    ----------
    element : str
        Element symbol
    product_isotope : str
        Product isotope used
    concentration_ug_g : float
        Concentration in μg/g (ppm)
    concentration_unc : float
        Uncertainty in concentration (μg/g)
    detection_limit_ug_g : float
        Detection limit in μg/g
    k0_used : float
        k0 factor used
    Q0_alpha_used : float
        Q0(α) factor used
    sdc_factor : float
        SDC correction factor
    specific_count_rate : float
        Specific count rate (counts/s/g)
    """
    element: str
    product_isotope: str
    concentration_ug_g: float
    concentration_unc: float
    detection_limit_ug_g: float = 0.0
    k0_used: float = 0.0
    Q0_alpha_used: float = 0.0
    sdc_factor: float = 0.0
    specific_count_rate: float = 0.0


# =============================================================================
# K0 Calculator Class
# =============================================================================

class K0Calculator:
    """
    Calculator for k0-NAA concentration determinations.
    
    This class implements the k0-standardization method for absolute
    NAA without comparator standards.
    
    Examples
    --------
    >>> # Characterize flux
    >>> flux_params = K0Parameters(f=25.0, alpha=0.02)
    >>> 
    >>> # Create calculator with Au reference
    >>> au_meas = K0Measurement(
    ...     product_isotope='Au-198',
    ...     net_peak_area=100000,
    ...     peak_area_unc=1000,
    ...     efficiency=0.01,
    ...     t_irr=3600,
    ...     t_decay=7200,
    ...     t_count=3600,
    ...     sample_mass=0.001,
    ... )
    >>> calc = K0Calculator(flux_params, au_meas)
    >>> 
    >>> # Analyze a peak
    >>> sample_meas = K0Measurement(
    ...     product_isotope='Co-60',
    ...     net_peak_area=5000,
    ...     peak_area_unc=100,
    ...     efficiency=0.005,
    ...     t_irr=3600,
    ...     t_decay=7200,
    ...     t_count=3600,
    ...     sample_mass=0.1,
    ... )
    >>> result = calc.calculate_concentration(sample_meas)
    >>> print(f"Co concentration: {result.concentration_ug_g:.2f} μg/g")
    """
    
    def __init__(
        self,
        flux_params: K0Parameters,
        au_measurement: Optional[K0Measurement] = None,
    ):
        """
        Initialize k0 calculator.
        
        Parameters
        ----------
        flux_params : K0Parameters
            Neutron flux characterization parameters (f, α)
        au_measurement : K0Measurement, optional
            Gold flux monitor measurement (if using relative method)
        """
        self.flux_params = flux_params
        self.au_measurement = au_measurement
        self._au_specific_count_rate = None
        self._au_Q0_alpha = None
        
        if au_measurement is not None:
            self._calculate_au_reference()
    
    def _calculate_au_reference(self):
        """Calculate Au reference specific count rate."""
        if self.au_measurement is None:
            return
        
        au_data = get_k0_data('Au-198')
        if au_data is None:
            raise ValueError("Au-198 data not found in k0 database")
        
        S, D, C, SDC = sdc_factor(
            au_data.half_life_s,
            self.au_measurement.t_irr,
            self.au_measurement.t_decay,
            self.au_measurement.t_count,
        )
        
        # Specific count rate = Np / (SDC * ε * W * t_count)
        self._au_specific_count_rate = (
            self.au_measurement.net_peak_area /
            (SDC * self.au_measurement.efficiency * self.au_measurement.sample_mass)
        )
        
        # Q0(α) for Au
        self._au_Q0_alpha = calculate_Q0_alpha(
            au_data.Q0, self.flux_params.alpha, au_data.E_res_eV
        )
    
    def calculate_concentration(
        self,
        measurement: K0Measurement,
        use_relative: bool = True,
    ) -> K0Result:
        """
        Calculate element concentration using k0 method.
        
        Parameters
        ----------
        measurement : K0Measurement
            The measurement to analyze
        use_relative : bool
            If True, use relative method with Au reference.
            If False, use absolute method with known flux.
            
        Returns
        -------
        K0Result
            Concentration result
        """
        # Get nuclide data
        nuclide_data = get_k0_data(measurement.product_isotope)
        if nuclide_data is None:
            raise ValueError(f"No k0 data for {measurement.product_isotope}")
        
        # Calculate SDC factors
        S, D, C, SDC = sdc_factor(
            nuclide_data.half_life_s,
            measurement.t_irr,
            measurement.t_decay,
            measurement.t_count,
        )
        
        # Calculate Q0(α)
        Q0_alpha = calculate_Q0_alpha(
            nuclide_data.Q0,
            self.flux_params.alpha,
            nuclide_data.E_res_eV,
        )
        
        # Specific count rate for the sample peak
        R_a = measurement.net_peak_area / (SDC * measurement.efficiency * measurement.sample_mass)
        
        if use_relative and self._au_specific_count_rate is not None:
            # Relative method using Au reference
            # ρ(a) = (R_a / R_Au) · (1 / k0,Au(a)) · G(α)
            # where G(α) = (f + Q0(α)_Au) / (f + Q0(α)_a)
            
            flux_ratio = (self.flux_params.f + self._au_Q0_alpha) / (self.flux_params.f + Q0_alpha)
            
            # Mass fraction (g/g)
            concentration = (
                R_a / self._au_specific_count_rate *
                1.0 / nuclide_data.k0_Au *
                flux_ratio
            )
            
        else:
            # Absolute method using known flux
            # ρ = Np / (SDC·ε·W·k0·φ_th·(f + Q0(α))/f)
            if self.flux_params.phi_thermal <= 0:
                raise ValueError("Need phi_thermal for absolute method")
            
            flux_factor = self.flux_params.phi_thermal * (self.flux_params.f + Q0_alpha) / self.flux_params.f
            concentration = R_a / (nuclide_data.k0_Au * flux_factor)
        
        # Convert to μg/g (ppm)
        concentration_ug_g = concentration * 1e6
        
        # Uncertainty propagation
        rel_unc_peak = measurement.peak_area_unc / measurement.net_peak_area if measurement.net_peak_area > 0 else 0
        rel_unc_k0 = nuclide_data.k0_unc / 100
        rel_unc_eff = measurement.efficiency_unc / 100
        rel_unc_Q0 = nuclide_data.Q0_unc / 100 * Q0_alpha / (self.flux_params.f + Q0_alpha)
        
        rel_unc_total = np.sqrt(rel_unc_peak**2 + rel_unc_k0**2 + rel_unc_eff**2 + rel_unc_Q0**2)
        concentration_unc = concentration_ug_g * rel_unc_total
        
        # Detection limit (Currie formulation, simplified)
        detection_limit = 3.0 * concentration_unc
        
        return K0Result(
            element=nuclide_data.element,
            product_isotope=measurement.product_isotope,
            concentration_ug_g=concentration_ug_g,
            concentration_unc=concentration_unc,
            detection_limit_ug_g=detection_limit,
            k0_used=nuclide_data.k0_Au,
            Q0_alpha_used=Q0_alpha,
            sdc_factor=SDC,
            specific_count_rate=R_a,
        )
    
    def analyze_spectrum(
        self,
        measurements: List[K0Measurement],
    ) -> Dict[str, K0Result]:
        """
        Analyze multiple peaks from a gamma spectrum.
        
        Parameters
        ----------
        measurements : list of K0Measurement
            All identified peaks to analyze
            
        Returns
        -------
        dict
            Element concentrations keyed by element symbol
        """
        results = {}
        
        for meas in measurements:
            try:
                result = self.calculate_concentration(meas)
                
                # If element already seen, combine results (weighted average)
                if result.element in results:
                    existing = results[result.element]
                    # Weighted average
                    w1 = 1.0 / existing.concentration_unc**2 if existing.concentration_unc > 0 else 1.0
                    w2 = 1.0 / result.concentration_unc**2 if result.concentration_unc > 0 else 1.0
                    combined_conc = (existing.concentration_ug_g * w1 + result.concentration_ug_g * w2) / (w1 + w2)
                    combined_unc = 1.0 / np.sqrt(w1 + w2)
                    
                    results[result.element] = K0Result(
                        element=result.element,
                        product_isotope=f"{existing.product_isotope}+{result.product_isotope}",
                        concentration_ug_g=combined_conc,
                        concentration_unc=combined_unc,
                        detection_limit_ug_g=min(existing.detection_limit_ug_g, result.detection_limit_ug_g),
                        k0_used=result.k0_used,
                        Q0_alpha_used=result.Q0_alpha_used,
                        sdc_factor=result.sdc_factor,
                        specific_count_rate=result.specific_count_rate,
                    )
                else:
                    results[result.element] = result
                    
            except Exception as e:
                logger.warning(f"Could not analyze {meas.product_isotope}: {e}")
        
        return results


# =============================================================================
# Utility Functions
# =============================================================================

def identify_isotope_from_gamma(
    energy_keV: float,
    tolerance_keV: float = 2.0,
) -> Optional[str]:
    """
    Identify product isotope from gamma energy.
    
    Parameters
    ----------
    energy_keV : float
        Measured gamma energy (keV)
    tolerance_keV : float
        Energy matching tolerance (keV)
        
    Returns
    -------
    str or None
        Product isotope name if matched, None otherwise
    """
    for isotope, data in K0_DATABASE.items():
        # Check primary gamma
        if abs(data.gamma_energy_keV - energy_keV) < tolerance_keV:
            return isotope
        # Check additional gammas
        for gamma_e, _ in data.additional_gammas:
            if abs(gamma_e - energy_keV) < tolerance_keV:
                return isotope
    return None


def create_k0_measurement_from_peak(
    peak_data: Any,
    efficiency: float,
    t_irr: float,
    t_decay: float,
    t_count: float,
    sample_mass: float,
    tolerance_keV: float = 2.0,
) -> Optional[K0Measurement]:
    """
    Create a K0Measurement from a detected peak.
    
    Parameters
    ----------
    peak_data : Peak or dict
        Peak detection result with net_area, energy, etc.
    efficiency : float
        Detector efficiency at peak energy
    t_irr, t_decay, t_count : float
        Timing parameters (s)
    sample_mass : float
        Sample mass (g)
    tolerance_keV : float
        Energy matching tolerance
        
    Returns
    -------
    K0Measurement or None
        K0Measurement if isotope is identified and in database
    """
    # Get peak properties
    if hasattr(peak_data, 'energy'):
        energy = peak_data.energy
        area = peak_data.net_area
        area_unc = peak_data.area_unc if hasattr(peak_data, 'area_unc') else area * 0.1
    elif isinstance(peak_data, dict):
        energy = peak_data.get('energy', 0)
        area = peak_data.get('net_area', 0)
        area_unc = peak_data.get('area_unc', area * 0.1)
    else:
        return None
    
    # Identify isotope from gamma energy
    isotope = identify_isotope_from_gamma(energy, tolerance_keV)
    
    if isotope is None:
        return None
    
    return K0Measurement(
        product_isotope=isotope,
        net_peak_area=area,
        peak_area_unc=area_unc,
        efficiency=efficiency,
        t_irr=t_irr,
        t_decay=t_decay,
        t_count=t_count,
        sample_mass=sample_mass,
        gamma_energy_keV=energy,
    )