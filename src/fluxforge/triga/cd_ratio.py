"""
Cd-Ratio Analysis for TRIGA Reactor Flux Characterization
==========================================================

This module implements the Cd-ratio method for characterizing neutron
fluxes in TRIGA reactors. The method uses paired bare and Cd-covered
flux monitors to determine:

- f: The thermal-to-epithermal flux ratio
- α: The epithermal flux shape parameter (deviation from 1/E)

Theory
------
The Cd-ratio is defined as:

    R_Cd = A_bare / A_Cd = (φ_th + φ_epi × Q_0) / (F_Cd × φ_epi × Q_0)

Where:
    - A_bare: Activity of bare monitor
    - A_Cd: Activity of Cd-covered monitor
    - φ_th: Thermal neutron flux
    - φ_epi: Epithermal neutron flux
    - Q_0: Resonance integral to σ_0 ratio
    - F_Cd: Cadmium transmission factor (typically ~1.0 for monitors)

For typical thermal reactors:
    f = φ_th / φ_epi = Q_0 × (R_Cd - 1) / F_Cd

The epithermal flux is modeled as:
    φ(E) = φ_epi × E^(-1-α)

For α ≠ 0, Q_0 must be corrected:
    Q_0(α) = (Q_0 - 0.429) / (E_r)^α + 0.429 × (2α + 1) / (E_Cd)^α

References
----------
- De Corte, F., et al. "The k0-standardisation method: a move to the
  optimization of neutron activation analysis" (1987)
- Simonits, A., et al. "α and f for 1/E^(1+α) epithermal neutron spectra
  determination" (1976)
- Westcott conventions for reactor neutron spectra (1960)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import warnings


@dataclass
class CdRatioMeasurement:
    """
    Data for a single Cd-ratio measurement pair.
    
    Attributes
    ----------
    element : str
        Element symbol (e.g., 'Co', 'Au', 'In')
    isotope : str
        Product isotope identifier (e.g., 'Co60', 'Au198')
    activity_bare : float
        Activity of bare monitor (Bq)
    activity_cd : float
        Activity of Cd-covered monitor (Bq)
    Q0 : float
        Resonance integral to thermal cross section ratio
    E_res : float
        Effective resonance energy (eV)
    sigma_0 : float, optional
        Thermal (2200 m/s) cross section (barns)
    half_life_s : float, optional
        Half-life of product isotope (seconds)
    uncertainty_bare : float, optional
        Uncertainty in bare activity (relative, fraction)
    uncertainty_cd : float, optional
        Uncertainty in Cd activity (relative, fraction)
    position : str, optional
        Irradiation position identifier
    """
    element: str
    isotope: str
    activity_bare: float
    activity_cd: float
    Q0: float
    E_res: float
    sigma_0: float = 0.0
    half_life_s: float = 0.0
    uncertainty_bare: float = 0.05
    uncertainty_cd: float = 0.07
    position: str = ""
    
    @property
    def cd_ratio(self) -> float:
        """Calculate the Cd-ratio (R_Cd)."""
        if self.activity_cd <= 0:
            return np.inf
        return self.activity_bare / self.activity_cd
    
    @property
    def cd_ratio_uncertainty(self) -> float:
        """Calculate uncertainty in Cd-ratio (propagated)."""
        # Relative uncertainty adds in quadrature
        rel_unc = np.sqrt(self.uncertainty_bare**2 + self.uncertainty_cd**2)
        return self.cd_ratio * rel_unc
    
    def __repr__(self):
        return (f"CdRatioMeasurement({self.element}, R_Cd={self.cd_ratio:.2f}±"
                f"{self.cd_ratio_uncertainty:.2f})")


@dataclass
class FluxParameters:
    """
    Neutron flux characterization parameters.
    
    Attributes
    ----------
    f : float
        Thermal-to-epithermal flux ratio
    f_uncertainty : float
        Uncertainty in f
    alpha : float
        Epithermal flux shape parameter (deviation from 1/E)
    alpha_uncertainty : float
        Uncertainty in α
    phi_thermal : float, optional
        Thermal flux (n/cm²/s) if absolute measurement available
    phi_epithermal : float, optional
        Epithermal flux (n/cm²/s)
    measurement_position : str, optional
        Position identifier
    cd_ratios : Dict[str, float], optional
        Individual Cd-ratios by element
    """
    f: float
    f_uncertainty: float
    alpha: float = 0.0
    alpha_uncertainty: float = 0.01
    phi_thermal: Optional[float] = None
    phi_epithermal: Optional[float] = None
    measurement_position: str = ""
    cd_ratios: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if self.f <= 0:
            warnings.warn(f"f = {self.f} is non-physical; should be > 0")
        if not -0.5 < self.alpha < 0.5:
            warnings.warn(f"α = {self.alpha} is unusual; typically -0.1 to 0.1")
    
    @property
    def is_well_thermalized(self) -> bool:
        """Check if spectrum is well-thermalized (f > 15)."""
        return self.f > 15
    
    @property
    def spectrum_description(self) -> str:
        """Describe the spectrum character."""
        if self.f > 30:
            return "Highly thermalized (graphite-moderated or peripheral)"
        elif self.f > 15:
            return "Well thermalized (typical TRIGA)"
        elif self.f > 5:
            return "Moderately hard spectrum"
        else:
            return "Hard/epithermal-rich spectrum"
    
    def __repr__(self):
        return f"FluxParameters(f={self.f:.1f}±{self.f_uncertainty:.1f}, α={self.alpha:.3f})"


# Standard nuclear data for Cd-ratio monitors
STANDARD_MONITORS: Dict[str, Dict[str, float]] = {
    'Au': {
        'Q0': 15.7,
        'E_res': 4.906,  # eV
        'sigma_0': 98.65,  # barns
        'E_Cd': 0.55,  # Cd cutoff energy (eV)
        'half_life_s': 232675.2,  # 2.694 days
        'isotope': 'Au198',
        'reaction': 'Au-197(n,g)Au-198',
    },
    'Co': {
        'Q0': 1.99,
        'E_res': 132.0,  # eV
        'sigma_0': 37.18,  # barns
        'E_Cd': 0.55,
        'half_life_s': 166344192.0,  # 5.271 years
        'isotope': 'Co60',
        'reaction': 'Co-59(n,g)Co-60',
    },
    'Sc': {
        'Q0': 0.43,
        'E_res': 4000.0,  # eV (high resonance)
        'sigma_0': 27.15,  # barns
        'E_Cd': 0.55,
        'half_life_s': 7239456.0,  # 83.79 days
        'isotope': 'Sc46',
        'reaction': 'Sc-45(n,g)Sc-46',
    },
    'In': {
        'Q0': 17.2,  # For In-113 thermal capture
        'E_res': 1.457,  # eV (main resonance)
        'sigma_0': 12.0,  # barns
        'E_Cd': 0.55,
        'half_life_s': 4280544.0,  # 49.51 days
        'isotope': 'In114m',
        'reaction': 'In-113(n,g)In-114m',
    },
    'Cu': {
        'Q0': 1.11,
        'E_res': 579.0,  # eV
        'sigma_0': 4.52,  # barns for Cu-63
        'E_Cd': 0.55,
        'half_life_s': 45835.2,  # 12.7 hours
        'isotope': 'Cu64',
        'reaction': 'Cu-63(n,g)Cu-64',
    },
    'Fe': {
        'Q0': 0.78,
        'E_res': 357.0,  # eV
        'sigma_0': 1.28,  # barns for Fe-58
        'E_Cd': 0.55,
        'half_life_s': 3845664.0,  # 44.5 days
        'isotope': 'Fe59',
        'reaction': 'Fe-58(n,g)Fe-59',
    },
    'Zr': {
        'Q0': 5.89,
        'E_res': 338.0,  # eV
        'sigma_0': 1.03,  # barns for Zr-94
        'E_Cd': 0.55,
        'half_life_s': 5529600.0,  # 64 days
        'isotope': 'Zr95',
        'reaction': 'Zr-94(n,g)Zr-95',
    },
}


def calculate_cd_ratio(
    activity_bare: float,
    activity_cd: float,
    uncertainty_bare: float = 0.05,
    uncertainty_cd: float = 0.07,
) -> Tuple[float, float]:
    """
    Calculate Cd-ratio and its uncertainty.
    
    Parameters
    ----------
    activity_bare : float
        Activity of bare monitor
    activity_cd : float
        Activity of Cd-covered monitor
    uncertainty_bare : float
        Relative uncertainty in bare activity
    uncertainty_cd : float
        Relative uncertainty in Cd activity
        
    Returns
    -------
    tuple
        (R_Cd, uncertainty)
    """
    if activity_cd <= 0:
        return np.inf, np.inf
    
    R_Cd = activity_bare / activity_cd
    rel_unc = np.sqrt(uncertainty_bare**2 + uncertainty_cd**2)
    
    return R_Cd, R_Cd * rel_unc


def calculate_Q0_alpha(Q0: float, E_res: float, alpha: float, E_Cd: float = 0.55) -> float:
    """
    Calculate α-corrected resonance integral ratio.
    
    For non-ideal 1/E epithermal spectra:
        Q_0(α) = (Q_0 - 0.429) × (E_r)^(-α) + 0.429 × (2α + 1) × (0.55/E_Cd)^α
    
    Parameters
    ----------
    Q0 : float
        Tabulated Q_0 value (at α=0)
    E_res : float
        Effective resonance energy (eV)
    alpha : float
        Epithermal shape parameter
    E_Cd : float
        Cadmium cutoff energy (default 0.55 eV)
        
    Returns
    -------
    float
        Q_0(α) corrected value
    """
    if alpha == 0:
        return Q0
    
    # De Corte formula
    term1 = (Q0 - 0.429) * (E_res ** (-alpha))
    term2 = 0.429 * (2 * alpha + 1) * ((0.55 / E_Cd) ** alpha)
    
    return term1 + term2


def estimate_f(R_Cd: float, Q0: float, alpha: float = 0.0, 
               E_res: float = 100.0, F_Cd: float = 1.0) -> float:
    """
    Estimate thermal-to-epithermal flux ratio f from Cd-ratio.
    
    From the relation:
        R_Cd = 1 + f / (Q_0(α) × F_Cd)
    
    We get:
        f = Q_0(α) × F_Cd × (R_Cd - 1)
    
    Parameters
    ----------
    R_Cd : float
        Measured Cd-ratio
    Q0 : float
        Resonance integral ratio (or Q_0(α) if alpha provided)
    alpha : float
        Epithermal shape parameter
    E_res : float
        Effective resonance energy (eV)
    F_Cd : float
        Cd transmission factor
        
    Returns
    -------
    float
        Estimated f value
    """
    Q0_eff = calculate_Q0_alpha(Q0, E_res, alpha) if alpha != 0 else Q0
    
    if R_Cd <= 1:
        warnings.warn(f"R_Cd = {R_Cd} <= 1 is non-physical")
        return 0.0
    
    return Q0_eff * F_Cd * (R_Cd - 1)


def estimate_alpha_multi(
    measurements: List[CdRatioMeasurement],
    initial_alpha: float = 0.0,
    tolerance: float = 1e-4,
    max_iterations: int = 20,
) -> Tuple[float, float]:
    """
    Estimate α using multiple Cd-ratio measurements with iterative refinement.
    
    Uses pairs of monitors with different E_res to determine α from:
        (R_Cd_1 - 1)/(R_Cd_2 - 1) = Q_0,2(α)/Q_0,1(α)
    
    Parameters
    ----------
    measurements : list
        CdRatioMeasurement objects
    initial_alpha : float
        Starting guess for α
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations
        
    Returns
    -------
    tuple
        (alpha, uncertainty)
    """
    if len(measurements) < 2:
        return initial_alpha, 0.05  # Default uncertainty
    
    # Sort by E_res
    sorted_meas = sorted(measurements, key=lambda m: m.E_res)
    
    alpha = initial_alpha
    alpha_values = []
    
    for iteration in range(max_iterations):
        new_alpha_estimates = []
        
        # Use all pairs
        for i in range(len(sorted_meas)):
            for j in range(i + 1, len(sorted_meas)):
                m1, m2 = sorted_meas[i], sorted_meas[j]
                
                R1, R2 = m1.cd_ratio, m2.cd_ratio
                
                if R1 <= 1 or R2 <= 1:
                    continue
                
                Q1 = calculate_Q0_alpha(m1.Q0, m1.E_res, alpha)
                Q2 = calculate_Q0_alpha(m2.Q0, m2.E_res, alpha)
                
                # Ratio method
                ratio = (R1 - 1) / (R2 - 1) * Q2 / Q1
                
                if ratio > 0:
                    E_ratio = m2.E_res / m1.E_res
                    if E_ratio != 1:
                        new_alpha = np.log(ratio) / np.log(E_ratio)
                        if -0.5 < new_alpha < 0.5:
                            new_alpha_estimates.append(new_alpha)
        
        if not new_alpha_estimates:
            break
            
        new_alpha = np.mean(new_alpha_estimates)
        alpha_values.extend(new_alpha_estimates)
        
        if abs(new_alpha - alpha) < tolerance:
            break
            
        alpha = new_alpha
    
    # Calculate uncertainty from spread
    if alpha_values:
        alpha_uncertainty = np.std(alpha_values) if len(alpha_values) > 1 else 0.02
    else:
        alpha_uncertainty = 0.05
    
    return alpha, alpha_uncertainty


class CdRatioAnalyzer:
    """
    Complete Cd-ratio analysis for TRIGA flux characterization.
    
    This class provides a comprehensive workflow for:
    1. Loading/parsing Cd-ratio measurements
    2. Determining flux parameters (f, α)
    3. Validating results against expected TRIGA values
    4. Generating reports and visualizations
    
    Example
    -------
    >>> analyzer = CdRatioAnalyzer()
    >>> analyzer.add_measurement('Co', 2035.0, 234.95)
    >>> analyzer.add_measurement('Sc', 1607650.0, 159063.0)
    >>> params = analyzer.characterize_flux()
    >>> print(params)
    FluxParameters(f=15.2±3.1, α=-0.02)
    """
    
    def __init__(self, position: str = ""):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        position : str
            Irradiation position identifier
        """
        self.position = position
        self.measurements: List[CdRatioMeasurement] = []
        self._flux_params: Optional[FluxParameters] = None
    
    def add_measurement(
        self,
        element: str,
        activity_bare: float,
        activity_cd: float,
        uncertainty_bare: float = 0.05,
        uncertainty_cd: float = 0.07,
    ) -> None:
        """
        Add a Cd-ratio measurement.
        
        Parameters
        ----------
        element : str
            Element symbol (must be in STANDARD_MONITORS)
        activity_bare : float
            Bare monitor activity (Bq)
        activity_cd : float
            Cd-covered monitor activity (Bq)
        uncertainty_bare : float
            Relative uncertainty in bare activity
        uncertainty_cd : float
            Relative uncertainty in Cd activity
        """
        if element not in STANDARD_MONITORS:
            raise ValueError(
                f"Unknown monitor element: {element}. "
                f"Available: {list(STANDARD_MONITORS.keys())}"
            )
        
        data = STANDARD_MONITORS[element]
        
        measurement = CdRatioMeasurement(
            element=element,
            isotope=data['isotope'],
            activity_bare=activity_bare,
            activity_cd=activity_cd,
            Q0=data['Q0'],
            E_res=data['E_res'],
            sigma_0=data['sigma_0'],
            half_life_s=data['half_life_s'],
            uncertainty_bare=uncertainty_bare,
            uncertainty_cd=uncertainty_cd,
            position=self.position,
        )
        
        self.measurements.append(measurement)
        self._flux_params = None  # Reset cached parameters
    
    def add_measurement_object(self, measurement: CdRatioMeasurement) -> None:
        """Add a pre-constructed CdRatioMeasurement."""
        self.measurements.append(measurement)
        self._flux_params = None
    
    def characterize_flux(self, alpha_method: str = 'multi') -> FluxParameters:
        """
        Determine flux parameters from all measurements.
        
        Parameters
        ----------
        alpha_method : str
            Method for α determination:
            - 'multi': Use multiple monitors (iterative)
            - 'default': Use default α=0
            - float value: Use specified α
            
        Returns
        -------
        FluxParameters
            Characterized flux parameters
        """
        if not self.measurements:
            raise ValueError("No measurements added")
        
        # Determine α
        if alpha_method == 'multi' and len(self.measurements) >= 2:
            alpha, alpha_unc = estimate_alpha_multi(self.measurements)
        elif alpha_method == 'default':
            alpha, alpha_unc = 0.0, 0.01
        elif isinstance(alpha_method, (int, float)):
            alpha, alpha_unc = float(alpha_method), 0.01
        else:
            alpha, alpha_unc = 0.0, 0.01
        
        # Calculate f for each monitor
        f_values = []
        cd_ratios = {}
        
        for m in self.measurements:
            R_Cd = m.cd_ratio
            cd_ratios[m.element] = R_Cd
            
            f_i = estimate_f(R_Cd, m.Q0, alpha, m.E_res)
            if f_i > 0:
                f_values.append(f_i)
        
        if not f_values:
            raise ValueError("Could not determine f from any measurement")
        
        # Use weighted average (prefer low-Q0 monitors for f)
        weights = [1.0 / (m.Q0 + 0.1) for m in self.measurements if m.cd_ratio > 1]
        if weights:
            f = np.average(f_values, weights=weights)
        else:
            f = np.mean(f_values)
        
        f_uncertainty = np.std(f_values) if len(f_values) > 1 else f * 0.1
        
        self._flux_params = FluxParameters(
            f=f,
            f_uncertainty=f_uncertainty,
            alpha=alpha,
            alpha_uncertainty=alpha_unc,
            measurement_position=self.position,
            cd_ratios=cd_ratios,
        )
        
        return self._flux_params
    
    def summary(self) -> str:
        """Generate a text summary of the analysis."""
        if self._flux_params is None:
            self.characterize_flux()
        
        lines = [
            "=" * 60,
            "Cd-Ratio Analysis Summary",
            "=" * 60,
            f"Position: {self.position or 'Not specified'}",
            f"Number of monitors: {len(self.measurements)}",
            "",
            "Individual Measurements:",
            "-" * 40,
        ]
        
        for m in self.measurements:
            lines.append(
                f"  {m.element:5s}: R_Cd = {m.cd_ratio:6.2f} ± {m.cd_ratio_uncertainty:5.2f} "
                f"(Q0={m.Q0:.2f}, E_res={m.E_res:.1f} eV)"
            )
        
        lines.extend([
            "",
            "Flux Parameters:",
            "-" * 40,
            f"  f = {self._flux_params.f:.1f} ± {self._flux_params.f_uncertainty:.1f}",
            f"  α = {self._flux_params.alpha:.4f} ± {self._flux_params.alpha_uncertainty:.4f}",
            "",
            f"Spectrum: {self._flux_params.spectrum_description}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def validate_triga(self) -> Dict[str, bool]:
        """
        Validate that results are consistent with TRIGA reactor physics.
        
        Returns
        -------
        dict
            Validation results with explanations
        """
        if self._flux_params is None:
            self.characterize_flux()
        
        p = self._flux_params
        results = {}
        
        # f should be in reasonable range for TRIGA
        results['f_in_range'] = 2.0 < p.f < 100.0
        
        # α should be small
        results['alpha_in_range'] = -0.2 < p.alpha < 0.2
        
        # Cd-ratios should be > 1 for all monitors
        if p.cd_ratios:
            results['all_cd_ratios_valid'] = all(r > 1 for r in p.cd_ratios.values())
        else:
            results['all_cd_ratios_valid'] = True
        
        # Check for internal consistency
        if len(self.measurements) >= 2:
            f_values = []
            for m in self.measurements:
                f_i = estimate_f(m.cd_ratio, m.Q0, p.alpha, m.E_res)
                if f_i > 0:
                    f_values.append(f_i)
            
            if len(f_values) >= 2:
                cv = np.std(f_values) / np.mean(f_values) if np.mean(f_values) > 0 else 1
                results['internally_consistent'] = cv < 0.5
            else:
                results['internally_consistent'] = True
        else:
            results['internally_consistent'] = True
        
        return results
    
    def to_dataframe(self):
        """Export measurements to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")
        
        data = []
        for m in self.measurements:
            data.append({
                'Element': m.element,
                'Isotope': m.isotope,
                'Activity_Bare_Bq': m.activity_bare,
                'Activity_Cd_Bq': m.activity_cd,
                'Cd_Ratio': m.cd_ratio,
                'Cd_Ratio_Unc': m.cd_ratio_uncertainty,
                'Q0': m.Q0,
                'E_res_eV': m.E_res,
                'Position': m.position,
            })
        
        return pd.DataFrame(data)
