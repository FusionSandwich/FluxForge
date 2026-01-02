"""
Detector Efficiency Module for HPGe Gamma Spectrometry

Provides detector efficiency curves and calibration tools for HPGe detectors.
Includes polynomial, functional, and empirical efficiency models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, optimize


@dataclass
class EfficiencyCurve:
    """
    Detector efficiency curve representation.
    
    Supports multiple efficiency models:
    - Polynomial in log-log space
    - Empirical points with interpolation
    - Custom functional forms
    
    Attributes
    ----------
    model_type : str
        Type of efficiency model: 'polynomial', 'empirical', 'functional'
    parameters : Dict[str, Any]
        Model parameters (coefficients, points, etc.)
    energy_range : Tuple[float, float]
        Valid energy range in keV
    detector_id : str
        Detector identifier
    calibration_date : str
        Date of efficiency calibration
    calibration_sources : List[str]
        Source(s) used for calibration
    geometry : Dict[str, Any]
        Detector-sample geometry (distance, container, etc.)
    uncertainty_model : Optional[Dict]
        Uncertainty model parameters
        
    Examples
    --------
    >>> curve = EfficiencyCurve.from_polynomial(
    ...     coefficients=[-5.2, -0.8, 0.02, -0.005],
    ...     energy_range=(50, 3000),
    ...     detector_id="HPGe-01"
    ... )
    >>> eff = curve.efficiency(661.7)  # At Cs-137 peak
    >>> print(f"Efficiency at 661.7 keV: {eff:.4f}")
    """
    
    model_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    energy_range: Tuple[float, float] = (0.0, 10000.0)
    detector_id: str = ""
    calibration_date: str = ""
    calibration_sources: List[str] = field(default_factory=list)
    geometry: Dict[str, Any] = field(default_factory=dict)
    uncertainty_model: Optional[Dict[str, Any]] = None
    
    # Internal interpolator for empirical model
    _interpolator: Optional[Callable] = field(default=None, repr=False)
    
    def efficiency(self, energy: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate detection efficiency at given energy.
        
        Parameters
        ----------
        energy : float or np.ndarray
            Energy in keV
        
        Returns
        -------
        float or np.ndarray
            Detection efficiency (fractional)
        """
        energy = np.atleast_1d(energy)
        
        if self.model_type == 'polynomial':
            return self._efficiency_polynomial(energy)
        elif self.model_type == 'empirical':
            return self._efficiency_empirical(energy)
        elif self.model_type == 'functional':
            return self._efficiency_functional(energy)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _efficiency_polynomial(self, energy: np.ndarray) -> np.ndarray:
        """
        Polynomial efficiency model in log-log space.
        
        ln(ε) = Σ aᵢ * (ln E)^i
        
        This is the standard form used in gamma spectrometry.
        """
        coefficients = self.parameters.get('coefficients', [0.0])
        
        log_e = np.log(energy)
        log_eff = np.zeros_like(log_e)
        
        for i, coeff in enumerate(coefficients):
            log_eff += coeff * (log_e ** i)
        
        efficiency = np.exp(log_eff)
        
        # Clip to valid range
        efficiency = np.clip(efficiency, 0.0, 1.0)
        
        return efficiency.squeeze()
    
    def _efficiency_empirical(self, energy: np.ndarray) -> np.ndarray:
        """
        Empirical efficiency from calibration points with interpolation.
        """
        if self._interpolator is None:
            cal_energies = np.array(self.parameters.get('energies', []))
            cal_efficiencies = np.array(self.parameters.get('efficiencies', []))
            
            if len(cal_energies) == 0:
                return np.zeros_like(energy)
            
            # Use log-log interpolation for physical behavior
            log_e = np.log(cal_energies)
            log_eff = np.log(np.clip(cal_efficiencies, 1e-20, 1.0))
            
            kind = self.parameters.get('interpolation', 'cubic')
            self._interpolator = interpolate.interp1d(
                log_e, log_eff,
                kind=kind,
                bounds_error=False,
                fill_value=(log_eff[0], log_eff[-1])
            )
        
        log_result = self._interpolator(np.log(energy))
        efficiency = np.exp(log_result)
        
        return np.clip(efficiency, 0.0, 1.0).squeeze()
    
    def _efficiency_functional(self, energy: np.ndarray) -> np.ndarray:
        """
        Custom functional form efficiency model.
        
        Supported forms:
        - 'gray': ε = exp(a + b*lnE + c*(lnE)² + d/E)
        - 'dual_polynomial': Separate polynomials for low/high energy
        - 'custom': User-provided function string
        """
        form = self.parameters.get('form', 'gray')
        
        if form == 'gray':
            # Gray & Ahmad model
            a = self.parameters.get('a', 0.0)
            b = self.parameters.get('b', 0.0)
            c = self.parameters.get('c', 0.0)
            d = self.parameters.get('d', 0.0)
            
            log_e = np.log(energy)
            log_eff = a + b * log_e + c * log_e**2 + d / energy
            efficiency = np.exp(log_eff)
        
        elif form == 'dual_polynomial':
            # Separate polynomials for low and high energy
            transition = self.parameters.get('transition_energy', 150.0)
            low_coeffs = self.parameters.get('low_energy_coefficients', [])
            high_coeffs = self.parameters.get('high_energy_coefficients', [])
            
            efficiency = np.zeros_like(energy)
            
            low_mask = energy < transition
            high_mask = ~low_mask
            
            # Low energy
            log_e_low = np.log(energy[low_mask])
            log_eff_low = sum(c * log_e_low**i for i, c in enumerate(low_coeffs))
            efficiency[low_mask] = np.exp(log_eff_low)
            
            # High energy
            log_e_high = np.log(energy[high_mask])
            log_eff_high = sum(c * log_e_high**i for i, c in enumerate(high_coeffs))
            efficiency[high_mask] = np.exp(log_eff_high)
        
        else:
            raise ValueError(f"Unknown functional form: {form}")
        
        return np.clip(efficiency, 0.0, 1.0).squeeze()
    
    def efficiency_uncertainty(
        self,
        energy: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate efficiency uncertainty at given energy.
        
        Parameters
        ----------
        energy : float or np.ndarray
            Energy in keV
        
        Returns
        -------
        float or np.ndarray
            Relative uncertainty (fractional, e.g., 0.05 for 5%)
        """
        if self.uncertainty_model is None:
            # Default: 5% relative uncertainty
            return 0.05 * np.ones_like(np.atleast_1d(energy)).squeeze()
        
        model_type = self.uncertainty_model.get('type', 'constant')
        
        if model_type == 'constant':
            value = self.uncertainty_model.get('value', 0.05)
            return value * np.ones_like(np.atleast_1d(energy)).squeeze()
        
        elif model_type == 'polynomial':
            # Uncertainty varies with energy
            coeffs = self.uncertainty_model.get('coefficients', [0.05])
            log_e = np.log(np.atleast_1d(energy))
            rel_unc = sum(c * log_e**i for i, c in enumerate(coeffs))
            return np.clip(rel_unc, 0.01, 0.5).squeeze()
        
        elif model_type == 'empirical':
            # Interpolate from calibration point uncertainties
            cal_energies = np.array(self.uncertainty_model.get('energies', []))
            cal_uncertainties = np.array(self.uncertainty_model.get('uncertainties', []))
            
            if len(cal_energies) == 0:
                return 0.05 * np.ones_like(np.atleast_1d(energy)).squeeze()
            
            interp = interpolate.interp1d(
                cal_energies, cal_uncertainties,
                kind='linear',
                bounds_error=False,
                fill_value=(cal_uncertainties[0], cal_uncertainties[-1])
            )
            return interp(np.atleast_1d(energy)).squeeze()
        
        return 0.05 * np.ones_like(np.atleast_1d(energy)).squeeze()
    
    @classmethod
    def from_polynomial(
        cls,
        coefficients: List[float],
        energy_range: Tuple[float, float] = (50.0, 3000.0),
        **kwargs
    ) -> 'EfficiencyCurve':
        """
        Create efficiency curve from polynomial coefficients.
        
        Parameters
        ----------
        coefficients : list of float
            Polynomial coefficients [a₀, a₁, a₂, ...] where
            ln(ε) = a₀ + a₁*ln(E) + a₂*(ln(E))² + ...
        energy_range : tuple
            Valid energy range (keV)
        **kwargs
            Additional EfficiencyCurve attributes
        
        Returns
        -------
        EfficiencyCurve
        """
        return cls(
            model_type='polynomial',
            parameters={'coefficients': coefficients},
            energy_range=energy_range,
            **kwargs
        )
    
    @classmethod
    def from_calibration_points(
        cls,
        energies: List[float],
        efficiencies: List[float],
        uncertainties: Optional[List[float]] = None,
        interpolation: str = 'cubic',
        **kwargs
    ) -> 'EfficiencyCurve':
        """
        Create efficiency curve from calibration points.
        
        Parameters
        ----------
        energies : list of float
            Calibration energies (keV)
        efficiencies : list of float
            Measured efficiencies at each energy
        uncertainties : list of float, optional
            Efficiency uncertainties
        interpolation : str
            Interpolation method: 'linear', 'cubic', etc.
        **kwargs
            Additional EfficiencyCurve attributes
        
        Returns
        -------
        EfficiencyCurve
        """
        parameters = {
            'energies': list(energies),
            'efficiencies': list(efficiencies),
            'interpolation': interpolation,
        }
        
        uncertainty_model = None
        if uncertainties is not None:
            uncertainty_model = {
                'type': 'empirical',
                'energies': list(energies),
                'uncertainties': list(uncertainties),
            }
        
        energy_range = (min(energies), max(energies))
        
        return cls(
            model_type='empirical',
            parameters=parameters,
            energy_range=energy_range,
            uncertainty_model=uncertainty_model,
            **kwargs
        )
    
    @classmethod
    def fit_from_points(
        cls,
        energies: List[float],
        efficiencies: List[float],
        uncertainties: Optional[List[float]] = None,
        polynomial_order: int = 4,
        **kwargs
    ) -> 'EfficiencyCurve':
        """
        Fit polynomial efficiency curve to calibration points.
        
        Parameters
        ----------
        energies : list of float
            Calibration energies (keV)
        efficiencies : list of float
            Measured efficiencies
        uncertainties : list of float, optional
            Efficiency uncertainties (for weighted fit)
        polynomial_order : int
            Order of polynomial fit
        **kwargs
            Additional EfficiencyCurve attributes
        
        Returns
        -------
        EfficiencyCurve
        """
        energies = np.array(energies)
        efficiencies = np.array(efficiencies)
        
        # Fit in log-log space
        log_e = np.log(energies)
        log_eff = np.log(np.clip(efficiencies, 1e-20, 1.0))
        
        if uncertainties is not None:
            # Weighted fit
            weights = 1.0 / (np.array(uncertainties) / efficiencies)
        else:
            weights = None
        
        # Polynomial fit
        coefficients = np.polyfit(log_e, log_eff, polynomial_order, w=weights)
        
        # Reverse to match convention (lowest order first)
        coefficients = list(coefficients[::-1])
        
        energy_range = (min(energies), max(energies))
        
        return cls(
            model_type='polynomial',
            parameters={'coefficients': coefficients},
            energy_range=energy_range,
            calibration_sources=kwargs.pop('calibration_sources', []),
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type,
            'parameters': self.parameters,
            'energy_range': list(self.energy_range),
            'detector_id': self.detector_id,
            'calibration_date': self.calibration_date,
            'calibration_sources': self.calibration_sources,
            'geometry': self.geometry,
            'uncertainty_model': self.uncertainty_model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EfficiencyCurve':
        """Create EfficiencyCurve from dictionary."""
        return cls(
            model_type=data['model_type'],
            parameters=data.get('parameters', {}),
            energy_range=tuple(data.get('energy_range', (0, 10000))),
            detector_id=data.get('detector_id', ''),
            calibration_date=data.get('calibration_date', ''),
            calibration_sources=data.get('calibration_sources', []),
            geometry=data.get('geometry', {}),
            uncertainty_model=data.get('uncertainty_model'),
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save efficiency curve to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'EfficiencyCurve':
        """Load efficiency curve from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# Standard Calibration Sources
# ============================================================================

# Gamma-ray energies and emission probabilities for common calibration sources
CALIBRATION_SOURCES = {
    'Am-241': [
        {'energy': 26.34, 'intensity': 0.024, 'uncertainty': 0.001},
        {'energy': 59.54, 'intensity': 0.359, 'uncertainty': 0.004},
    ],
    'Cd-109': [
        {'energy': 88.03, 'intensity': 0.037, 'uncertainty': 0.002},
    ],
    'Co-57': [
        {'energy': 122.06, 'intensity': 0.856, 'uncertainty': 0.006},
        {'energy': 136.47, 'intensity': 0.107, 'uncertainty': 0.001},
    ],
    'Ce-139': [
        {'energy': 165.86, 'intensity': 0.800, 'uncertainty': 0.006},
    ],
    'Cr-51': [
        {'energy': 320.08, 'intensity': 0.0986, 'uncertainty': 0.001},
    ],
    'Sn-113': [
        {'energy': 391.70, 'intensity': 0.649, 'uncertainty': 0.006},
    ],
    'Sr-85': [
        {'energy': 514.01, 'intensity': 0.960, 'uncertainty': 0.004},
    ],
    'Cs-137': [
        {'energy': 661.66, 'intensity': 0.851, 'uncertainty': 0.002},
    ],
    'Mn-54': [
        {'energy': 834.85, 'intensity': 0.9998, 'uncertainty': 0.001},
    ],
    'Y-88': [
        {'energy': 898.04, 'intensity': 0.937, 'uncertainty': 0.003},
        {'energy': 1836.06, 'intensity': 0.992, 'uncertainty': 0.003},
    ],
    'Co-60': [
        {'energy': 1173.23, 'intensity': 0.9985, 'uncertainty': 0.0003},
        {'energy': 1332.49, 'intensity': 0.9998, 'uncertainty': 0.0001},
    ],
    'Na-22': [
        {'energy': 511.00, 'intensity': 1.798, 'uncertainty': 0.002},  # Annihilation
        {'energy': 1274.54, 'intensity': 0.9994, 'uncertainty': 0.0002},
    ],
    'Eu-152': [
        {'energy': 121.78, 'intensity': 0.286, 'uncertainty': 0.003},
        {'energy': 244.70, 'intensity': 0.076, 'uncertainty': 0.001},
        {'energy': 344.28, 'intensity': 0.266, 'uncertainty': 0.002},
        {'energy': 411.12, 'intensity': 0.0224, 'uncertainty': 0.0003},
        {'energy': 443.97, 'intensity': 0.0312, 'uncertainty': 0.0004},
        {'energy': 778.90, 'intensity': 0.129, 'uncertainty': 0.001},
        {'energy': 867.38, 'intensity': 0.0422, 'uncertainty': 0.0004},
        {'energy': 964.13, 'intensity': 0.146, 'uncertainty': 0.001},
        {'energy': 1085.84, 'intensity': 0.102, 'uncertainty': 0.001},
        {'energy': 1112.08, 'intensity': 0.137, 'uncertainty': 0.001},
        {'energy': 1408.01, 'intensity': 0.210, 'uncertainty': 0.002},
    ],
    'Ba-133': [
        {'energy': 80.99, 'intensity': 0.329, 'uncertainty': 0.003},
        {'energy': 276.40, 'intensity': 0.0716, 'uncertainty': 0.0005},
        {'energy': 302.85, 'intensity': 0.1834, 'uncertainty': 0.0013},
        {'energy': 356.01, 'intensity': 0.6205, 'uncertainty': 0.0019},
        {'energy': 383.85, 'intensity': 0.0894, 'uncertainty': 0.0006},
    ],
}


def calculate_efficiency_from_source(
    measured_counts: float,
    live_time: float,
    source_activity: float,
    emission_probability: float,
    geometry_factor: float = 1.0,
    count_uncertainty: Optional[float] = None,
    activity_uncertainty: Optional[float] = None,
    probability_uncertainty: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate detection efficiency from calibration measurement.
    
    Parameters
    ----------
    measured_counts : float
        Net counts in peak (background subtracted)
    live_time : float
        Measurement live time in seconds
    source_activity : float
        Source activity in Bq
    emission_probability : float
        Gamma emission probability
    geometry_factor : float
        Correction factor for sample geometry differences
    count_uncertainty : float, optional
        Uncertainty in measured counts
    activity_uncertainty : float, optional
        Relative uncertainty in source activity
    probability_uncertainty : float, optional
        Absolute uncertainty in emission probability
    
    Returns
    -------
    efficiency : float
        Detection efficiency
    uncertainty : float
        Efficiency uncertainty
    """
    # Count rate
    count_rate = measured_counts / live_time
    
    # Expected emission rate
    emission_rate = source_activity * emission_probability
    
    # Efficiency
    efficiency = count_rate / (emission_rate * geometry_factor)
    
    # Uncertainty propagation
    rel_unc_squared = 0.0
    
    if count_uncertainty is not None:
        rel_unc_squared += (count_uncertainty / measured_counts) ** 2
    else:
        # Poisson uncertainty
        rel_unc_squared += 1.0 / measured_counts
    
    if activity_uncertainty is not None:
        rel_unc_squared += activity_uncertainty ** 2
    
    if probability_uncertainty is not None:
        rel_unc_squared += (probability_uncertainty / emission_probability) ** 2
    
    uncertainty = efficiency * np.sqrt(rel_unc_squared)
    
    return efficiency, uncertainty


def distance_correction(
    d1: float,
    d2: float,
    detector_radius: Optional[float] = None
) -> float:
    """
    Calculate geometric efficiency correction for different source-detector distances.
    
    For point sources at different distances, scales approximately as 1/d².
    For extended sources or very close distances, a more complex model is needed.
    
    Parameters
    ----------
    d1 : float
        Original calibration distance (cm)
    d2 : float
        New measurement distance (cm)
    detector_radius : float, optional
        Detector crystal radius for close-geometry correction (cm)
    
    Returns
    -------
    float
        Efficiency correction factor (multiply new efficiency by this)
    """
    if detector_radius is not None and (d1 < 3 * detector_radius or d2 < 3 * detector_radius):
        # Use solid angle ratio for close geometry
        omega1 = 2 * np.pi * (1 - d1 / np.sqrt(d1**2 + detector_radius**2))
        omega2 = 2 * np.pi * (1 - d2 / np.sqrt(d2**2 + detector_radius**2))
        return omega1 / omega2
    else:
        # Simple inverse square law
        return (d1 / d2) ** 2
