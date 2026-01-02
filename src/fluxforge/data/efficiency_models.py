"""
Enhanced Efficiency Models

Multiple detector efficiency model implementations based on common
formulations used in gamma spectroscopy analysis software.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np


@dataclass 
class EfficiencyModelParams:
    """Parameters for an efficiency model."""
    
    model_type: str  # 'log_poly', 'log10_poly', 'labsocs', 'custom'
    coefficients: Dict[str, float]
    description: str = ""


class EfficiencyModel:
    """
    Detector efficiency model supporting multiple formulations.
    
    Supported model types:
    
    1. **log_poly** (LabSOCS natural log):
       ε = DetModel × (C1 + C2×ln(E) + C3×ln(E)² + C4×ln(E)³)
    
    2. **log10_poly** (log10 polynomial):
       ε = 10^(C1 + C2×log₁₀(E) + C3×log₁₀(E)² + C4×log₁₀(E)³ + A×log₁₀(E)⁴)
    
    3. **log_log_poly** (standard log-log polynomial):
       ln(ε) = Σ aᵢ(ln E)^i
    
    4. **custom**: User-provided function
    """
    
    def __init__(
        self,
        model_type: str,
        coefficients: Optional[Dict[str, float]] = None,
        custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        description: str = "",
    ):
        """
        Initialize efficiency model.
        
        Parameters
        ----------
        model_type : str
            One of 'log_poly', 'log10_poly', 'log_log_poly', 'custom'
        coefficients : dict, optional
            Model coefficients (required for non-custom models)
        custom_func : callable, optional
            Custom function E (keV) -> efficiency (required for 'custom')
        description : str
            Description of the model
        """
        self.model_type = model_type
        self.coefficients = coefficients or {}
        self.custom_func = custom_func
        self.description = description
        
        if model_type == 'custom' and custom_func is None:
            raise ValueError("custom_func required for 'custom' model type")
    
    def __call__(self, energy_keV: np.ndarray) -> np.ndarray:
        """Calculate efficiency for given energies."""
        return self.calculate(energy_keV)
    
    def calculate(self, energy_keV: np.ndarray) -> np.ndarray:
        """
        Calculate detector efficiency for given energies.
        
        Parameters
        ----------
        energy_keV : ndarray
            Gamma energies in keV
        
        Returns
        -------
        ndarray
            Efficiency values (NaN for invalid energies)
        """
        E = np.asarray(energy_keV)
        
        if self.model_type == 'log_poly':
            return self._calc_log_poly(E)
        elif self.model_type == 'log10_poly':
            return self._calc_log10_poly(E)
        elif self.model_type == 'log_log_poly':
            return self._calc_log_log_poly(E)
        elif self.model_type == 'custom':
            return self.custom_func(E)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _calc_log_poly(self, E: np.ndarray) -> np.ndarray:
        """
        LabSOCS-style natural log polynomial.
        
        ε = DetModel × (C1 + C2×ln(E) + C3×ln(E)² + C4×ln(E)³)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            logE = np.log(np.where(E > 0, E, np.nan))
        
        c = self.coefficients
        det = c.get('DetModel', c.get('det_model', 1.0))
        c1 = c.get('C1', 0.0)
        c2 = c.get('C2', 0.0)
        c3 = c.get('C3', 0.0)
        c4 = c.get('C4', 0.0)
        
        eps = det * (c1 + c2 * logE + c3 * logE**2 + c4 * logE**3)
        
        # Mask invalid values
        return np.where(np.isfinite(eps) & (eps > 0), eps, np.nan)
    
    def _calc_log10_poly(self, E: np.ndarray) -> np.ndarray:
        """
        Log10 polynomial (as in v4_gamma_spec.py).
        
        ε = 10^(C1 + C2×log₁₀(E) + C3×log₁₀(E)² + C4×log₁₀(E)³ + A×log₁₀(E)⁴)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.log10(np.where(E > 0, E, np.nan))
        
        c = self.coefficients
        c1 = c.get('C1', 0.0)
        c2 = c.get('C2', 0.0)
        c3 = c.get('C3', 0.0)
        c4 = c.get('C4', 0.0)
        a = c.get('A', 0.0)
        
        log_eps = c1 + c2 * x + c3 * x**2 + c4 * x**3 + a * x**4
        eps = np.power(10.0, log_eps, where=~np.isnan(log_eps))
        
        # Mask invalid/negative
        return np.where(np.isfinite(eps) & (eps > 0), eps, np.nan)
    
    def _calc_log_log_poly(self, E: np.ndarray) -> np.ndarray:
        """
        Standard log-log polynomial.
        
        ln(ε) = a₀ + a₁×ln(E) + a₂×ln(E)² + ...
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.log(np.where(E > 0, E, np.nan))
        
        c = self.coefficients
        
        # Get ordered coefficients
        log_eps = np.zeros_like(x)
        for i in range(10):  # Support up to 10 terms
            key = f'a{i}'
            if key in c:
                log_eps += c[key] * (x ** i)
        
        eps = np.exp(log_eps)
        return np.where(np.isfinite(eps) & (eps > 0), eps, np.nan)
    
    @classmethod
    def from_labsocs_csv(cls, csv_path: str) -> 'EfficiencyModel':
        """
        Create model from LabSOCS efficiency CSV.
        
        Expected format: header row with C1, C2, C3, C4 (and optional DetModel)
        followed by one data row.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file
        
        Returns
        -------
        EfficiencyModel
            Configured model
        """
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = [h.strip() for h in f.readline().strip().split(',')]
            values = [v.strip() for v in f.readline().strip().split(',')]
        
        hmap = {h.lower(): h for h in header}
        
        def get_val(name: str, default: float = 0.0) -> float:
            if name.lower() not in hmap:
                return default
            return float(values[header.index(hmap[name.lower()])])
        
        coefficients = {
            'C1': get_val('C1'),
            'C2': get_val('C2'),
            'C3': get_val('C3'),
            'C4': get_val('C4'),
        }
        
        # Try various DetModel key names
        for key in ['DetModel', 'det_model', 'Det.Model', 'detector_model']:
            if key.lower() in hmap:
                coefficients['DetModel'] = get_val(key, 1.0)
                break
        else:
            coefficients['DetModel'] = 1.0
        
        return cls(
            model_type='log_poly',
            coefficients=coefficients,
            description=f'LabSOCS model from {csv_path}',
        )
    
    @classmethod
    def from_v4_csv(cls, csv_path: str) -> 'EfficiencyModel':
        """
        Create log10 polynomial model from v4_gamma_spec style CSV.
        
        Expected format: header with C1, C2, C3, C4, A
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file
        
        Returns
        -------
        EfficiencyModel
            Configured model
        """
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            values = f.readline().strip().split(',')
        
        coefficients = {}
        for h, v in zip(header, values):
            h = h.strip()
            if h in ('C1', 'C2', 'C3', 'C4', 'A'):
                coefficients[h] = float(v.strip())
        
        return cls(
            model_type='log10_poly',
            coefficients=coefficients,
            description=f'Log10 polynomial from {csv_path}',
        )


def apply_efficiency_correction(
    counts: np.ndarray,
    energies: np.ndarray,
    model: EfficiencyModel,
) -> np.ndarray:
    """
    Apply efficiency correction to spectrum counts.
    
    Parameters
    ----------
    counts : ndarray
        Raw counts
    energies : ndarray
        Calibrated energies (keV)
    model : EfficiencyModel
        Efficiency model
    
    Returns
    -------
    ndarray
        Corrected counts (counts / efficiency)
    """
    efficiency = model.calculate(energies)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        corrected = counts / efficiency
    
    return np.where(np.isfinite(corrected), corrected, np.nan)
