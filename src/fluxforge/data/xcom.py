"""
XCOM Mass Attenuation Coefficients

Provides access to NIST XCOM photon cross-section data for gamma-ray
attenuation calculations.

Epic R - Becquerel Parity

References:
    https://www.nist.gov/pml/xcom-photon-cross-sections-database
    http://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate


# =============================================================================
# XCOM Data Tables (Embedded from NIST)
# =============================================================================

# Energy grid for interpolation (MeV)
XCOM_ENERGIES = np.array([
    0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01,
    0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2,
    0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0,
    6.0, 8.0, 10.0, 15.0, 20.0
])

# Mass attenuation coefficients (cm²/g) for common elements
# Data from NIST XCOM
XCOM_DATA = {
    # Air (dry) - approximate average
    'Air': {
        'Z_eff': 7.64,
        'density': 0.001205,  # g/cm³ at STP
        'mu_rho': np.array([
            3599., 1191., 527.4, 151.8, 59.01, 27.62, 14.71, 5.503, 2.618,
            0.8204, 0.3687, 0.1606, 0.1079, 0.08712, 0.07653, 0.06545,
            0.05973, 0.05176, 0.04764, 0.04350, 0.04095, 0.03931, 0.03815,
            0.03653, 0.03546, 0.03440, 0.03361, 0.03240, 0.03086, 0.02998,
            0.02943, 0.02906, 0.02863, 0.02844, 0.02824, 0.02822
        ])
    },
    # Water (H2O)
    'Water': {
        'Z_eff': 7.42,
        'density': 1.0,
        'mu_rho': np.array([
            4078., 1376., 617.3, 181.5, 71.21, 33.53, 17.93, 6.764, 3.228,
            1.028, 0.4678, 0.2059, 0.1361, 0.1094, 0.09595, 0.08191,
            0.07461, 0.06451, 0.05933, 0.05411, 0.05095, 0.04891, 0.04747,
            0.04546, 0.04415, 0.04284, 0.04186, 0.04037, 0.03849, 0.03742,
            0.03673, 0.03627, 0.03571, 0.03543, 0.03514, 0.03510
        ])
    },
    # Lead (Pb)
    'Lead': {
        'Z_eff': 82,
        'density': 11.34,
        'mu_rho': np.array([
            5.54e3, 2.37e3, 1.17e3, 3.76e2, 1.58e2, 77.1, 42.5, 16.0, 7.26,
            2.01, 0.813, 0.329, 0.221, 0.182, 0.161, 0.139, 0.127, 0.107,
            0.0958, 0.0826, 0.0743, 0.0690, 0.0654, 0.0606, 0.0575, 0.0549,
            0.0531, 0.0509, 0.0498, 0.0509, 0.0527, 0.0549, 0.0593, 0.0632,
            0.0712, 0.0774
        ])
    },
    # Iron (Fe)
    'Iron': {
        'Z_eff': 26,
        'density': 7.874,
        'mu_rho': np.array([
            9174., 3390., 1531., 451.4, 177.0, 83.20, 44.74, 16.83, 7.957,
            2.484, 1.085, 0.4122, 0.2401, 0.1776, 0.1476, 0.1190,
            0.1044, 0.08484, 0.07527, 0.06552, 0.06001, 0.05626, 0.05357,
            0.04997, 0.04774, 0.04575, 0.04439, 0.04247, 0.04036, 0.03936,
            0.03891, 0.03872, 0.03876, 0.03904, 0.03997, 0.04079
        ])
    },
    # Aluminum (Al)
    'Aluminum': {
        'Z_eff': 13,
        'density': 2.699,
        'mu_rho': np.array([
            1185., 399.3, 172.6, 48.57, 18.50, 8.573, 4.541, 1.704, 0.8186,
            0.2658, 0.1279, 0.06685, 0.05149, 0.04589, 0.04285, 0.03928,
            0.03709, 0.03343, 0.03125, 0.02883, 0.02727, 0.02625, 0.02553,
            0.02457, 0.02395, 0.02336, 0.02292, 0.02228, 0.02145, 0.02098,
            0.02068, 0.02049, 0.02028, 0.02019, 0.02012, 0.02017
        ])
    },
    # Copper (Cu)
    'Copper': {
        'Z_eff': 29,
        'density': 8.96,
        'mu_rho': np.array([
            9033., 3334., 1504., 442.5, 173.3, 81.44, 43.75, 16.43, 7.760,
            2.412, 1.050, 0.3961, 0.2300, 0.1700, 0.1413, 0.1139,
            0.1001, 0.08143, 0.07224, 0.06291, 0.05765, 0.05405, 0.05147,
            0.04805, 0.04591, 0.04401, 0.04271, 0.04089, 0.03889, 0.03794,
            0.03752, 0.03737, 0.03745, 0.03775, 0.03873, 0.03957
        ])
    },
    # Germanium (Ge) - for HPGe detectors
    'Germanium': {
        'Z_eff': 32,
        'density': 5.323,
        'mu_rho': np.array([
            8538., 3160., 1426., 420.6, 165.0, 77.73, 41.85, 15.78, 7.473,
            2.340, 1.026, 0.3930, 0.2315, 0.1728, 0.1446, 0.1175,
            0.1038, 0.08513, 0.07587, 0.06633, 0.06093, 0.05722, 0.05453,
            0.05099, 0.04875, 0.04675, 0.04538, 0.04346, 0.04137, 0.04037,
            0.03990, 0.03971, 0.03975, 0.04003, 0.04100, 0.04184
        ])
    },
    # Silicon (Si)
    'Silicon': {
        'Z_eff': 14,
        'density': 2.33,
        'mu_rho': np.array([
            1570., 531.1, 230.0, 65.02, 24.87, 11.54, 6.126, 2.304, 1.107,
            0.3596, 0.1716, 0.08508, 0.06268, 0.05410, 0.04974, 0.04483,
            0.04191, 0.03734, 0.03467, 0.03181, 0.03001, 0.02887, 0.02806,
            0.02696, 0.02625, 0.02558, 0.02508, 0.02435, 0.02343, 0.02290,
            0.02256, 0.02234, 0.02209, 0.02197, 0.02186, 0.02189
        ])
    },
    # Cadmium (Cd) - for CZT detectors
    'Cadmium': {
        'Z_eff': 48,
        'density': 8.65,
        'mu_rho': np.array([
            6715., 2500., 1136., 341.0, 135.4, 64.50, 35.07, 13.41, 6.418,
            2.060, 0.9184, 0.3614, 0.2147, 0.1600, 0.1332, 0.1076,
            0.09499, 0.07756, 0.06898, 0.06008, 0.05514, 0.05178, 0.04934,
            0.04614, 0.04413, 0.04237, 0.04115, 0.03953, 0.03779, 0.03707,
            0.03685, 0.03688, 0.03724, 0.03774, 0.03909, 0.04023
        ])
    },
    # Tellurium (Te) - for CZT detectors
    'Tellurium': {
        'Z_eff': 52,
        'density': 6.24,
        'mu_rho': np.array([
            6274., 2337., 1064., 320.6, 127.6, 60.91, 33.19, 12.74, 6.116,
            1.975, 0.8847, 0.3510, 0.2094, 0.1566, 0.1305, 0.1058,
            0.09352, 0.07660, 0.06826, 0.05957, 0.05474, 0.05145, 0.04906,
            0.04595, 0.04398, 0.04226, 0.04107, 0.03949, 0.03783, 0.03715,
            0.03696, 0.03702, 0.03742, 0.03795, 0.03936, 0.04053
        ])
    },
    # Sodium Iodide (NaI) - for scintillators
    'NaI': {
        'Z_eff': 50.8,  # Effective Z
        'density': 3.67,
        'mu_rho': np.array([
            5926., 2206., 1003., 301.5, 119.8, 57.14, 31.11, 11.92, 5.716,
            1.840, 0.8216, 0.3243, 0.1928, 0.1438, 0.1197, 0.09671,
            0.08531, 0.06972, 0.06204, 0.05408, 0.04967, 0.04668, 0.04452,
            0.04167, 0.03988, 0.03833, 0.03726, 0.03583, 0.03426, 0.03363,
            0.03347, 0.03354, 0.03395, 0.03449, 0.03591, 0.03709
        ])
    },
    # Concrete (approximate)
    'Concrete': {
        'Z_eff': 11.0,
        'density': 2.3,
        'mu_rho': np.array([
            3150., 1050., 460., 134., 52., 24., 13., 4.9, 2.4,
            0.76, 0.35, 0.15, 0.099, 0.080, 0.071, 0.061,
            0.056, 0.049, 0.045, 0.041, 0.039, 0.037, 0.036,
            0.035, 0.034, 0.033, 0.032, 0.031, 0.030, 0.029,
            0.029, 0.028, 0.028, 0.028, 0.028, 0.028
        ])
    }
}


@dataclass
class AttenuationData:
    """
    Mass attenuation coefficient data for a material.
    
    Attributes
    ----------
    material : str
        Material name
    energies_keV : np.ndarray
        Energy grid (keV)
    mu_rho : np.ndarray
        Mass attenuation coefficients (cm²/g)
    density : float
        Material density (g/cm³)
    Z_eff : float
        Effective atomic number
    """
    material: str
    energies_keV: np.ndarray
    mu_rho: np.ndarray
    density: float
    Z_eff: float
    
    def get_mu_rho(self, energy_keV: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get mass attenuation coefficient at given energy.
        
        Uses log-log interpolation for accuracy.
        
        Parameters
        ----------
        energy_keV : float or array
            Photon energy in keV
        
        Returns
        -------
        np.ndarray
            Mass attenuation coefficient (cm²/g)
        """
        energy_keV = np.atleast_1d(energy_keV)
        
        # Log-log interpolation
        log_E = np.log10(self.energies_keV)
        log_mu = np.log10(np.maximum(self.mu_rho, 1e-10))
        
        interp = interpolate.interp1d(
            log_E, log_mu, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        
        log_result = interp(np.log10(energy_keV))
        return 10**log_result
    
    def get_mu(self, energy_keV: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get linear attenuation coefficient at given energy.
        
        μ = μ/ρ × ρ
        
        Parameters
        ----------
        energy_keV : float or array
            Photon energy in keV
        
        Returns
        -------
        np.ndarray
            Linear attenuation coefficient (cm⁻¹)
        """
        return self.get_mu_rho(energy_keV) * self.density
    
    def transmission(
        self,
        energy_keV: Union[float, np.ndarray],
        thickness_cm: float
    ) -> np.ndarray:
        """
        Calculate transmission factor through material.
        
        T = exp(-μ × t)
        
        Parameters
        ----------
        energy_keV : float or array
            Photon energy in keV
        thickness_cm : float
            Material thickness in cm
        
        Returns
        -------
        np.ndarray
            Transmission factor (0-1)
        """
        mu = self.get_mu(energy_keV)
        return np.exp(-mu * thickness_cm)
    
    def half_value_layer(self, energy_keV: float) -> float:
        """
        Calculate half-value layer (HVL) at given energy.
        
        HVL = ln(2) / μ
        
        Parameters
        ----------
        energy_keV : float
            Photon energy in keV
        
        Returns
        -------
        float
            Half-value layer in cm
        """
        mu = float(self.get_mu(energy_keV))
        if mu <= 0:
            return np.inf
        return np.log(2) / mu


def get_attenuation_data(material: str) -> AttenuationData:
    """
    Get mass attenuation data for a material.
    
    Parameters
    ----------
    material : str
        Material name (e.g., 'Lead', 'Iron', 'Water', 'Air')
    
    Returns
    -------
    AttenuationData
        Attenuation data object with interpolation methods
    
    Raises
    ------
    ValueError
        If material is not in the database
    
    Examples
    --------
    >>> data = get_attenuation_data('Lead')
    >>> data.get_mu_rho(662)  # μ/ρ at 662 keV
    >>> data.transmission(662, 2.0)  # Transmission through 2 cm Pb
    """
    if material not in XCOM_DATA:
        available = list(XCOM_DATA.keys())
        raise ValueError(f"Material '{material}' not found. Available: {available}")
    
    mat_data = XCOM_DATA[material]
    
    return AttenuationData(
        material=material,
        energies_keV=XCOM_ENERGIES * 1000,  # Convert MeV to keV
        mu_rho=mat_data['mu_rho'],
        density=mat_data['density'],
        Z_eff=mat_data['Z_eff']
    )


def list_materials() -> List[str]:
    """List available materials in the XCOM database."""
    return list(XCOM_DATA.keys())


def calculate_transmission(
    material: str,
    energy_keV: Union[float, np.ndarray],
    thickness_cm: float
) -> np.ndarray:
    """
    Quick calculation of gamma transmission through material.
    
    Parameters
    ----------
    material : str
        Material name
    energy_keV : float or array
        Photon energy in keV
    thickness_cm : float
        Material thickness in cm
    
    Returns
    -------
    np.ndarray
        Transmission factor (0-1)
    """
    data = get_attenuation_data(material)
    return data.transmission(energy_keV, thickness_cm)


def calculate_hvl(material: str, energy_keV: float) -> float:
    """
    Calculate half-value layer for material at energy.
    
    Parameters
    ----------
    material : str
        Material name
    energy_keV : float
        Photon energy in keV
    
    Returns
    -------
    float
        Half-value layer in cm
    """
    data = get_attenuation_data(material)
    return data.half_value_layer(energy_keV)


def attenuation_factor(
    material: str,
    energy_keV: float,
    thickness_cm: float,
    include_buildup: bool = False,
    geometry: str = 'narrow_beam'
) -> float:
    """
    Calculate gamma attenuation factor with optional buildup.
    
    Parameters
    ----------
    material : str
        Material name
    energy_keV : float
        Photon energy in keV
    thickness_cm : float
        Material thickness in cm
    include_buildup : bool
        Whether to include buildup factor for broad-beam geometry
    geometry : str
        'narrow_beam' (no buildup) or 'broad_beam' (with buildup)
    
    Returns
    -------
    float
        Attenuation factor (ratio of transmitted to incident intensity)
    
    Notes
    -----
    For narrow-beam (good geometry), attenuation = exp(-μt).
    For broad-beam, buildup factor B is applied: I/I₀ = B × exp(-μt).
    """
    data = get_attenuation_data(material)
    mu = float(data.get_mu(energy_keV))
    
    # Basic attenuation
    att = np.exp(-mu * thickness_cm)
    
    if include_buildup or geometry == 'broad_beam':
        # Simple Taylor buildup approximation
        # B ≈ 1 + μt for moderate shielding
        mu_t = mu * thickness_cm
        if mu_t < 10:
            # Linear approximation valid for thin shields
            buildup = 1 + 0.5 * mu_t
        else:
            # More complex buildup for thick shields
            buildup = 1 + mu_t + 0.3 * mu_t**2
        att = att * buildup
    
    return att


# =============================================================================
# Compound Material Support
# =============================================================================


def mixture_mu_rho(
    composition: Dict[str, float],
    energy_keV: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Calculate mass attenuation for a mixture of elements.
    
    Uses the additivity rule (Bragg additivity):
    μ/ρ = Σ wᵢ × (μ/ρ)ᵢ
    
    Parameters
    ----------
    composition : dict
        Element weights {material: weight_fraction}
        Weight fractions should sum to 1.0
    energy_keV : float or array
        Photon energy in keV
    
    Returns
    -------
    np.ndarray
        Mixture mass attenuation coefficient (cm²/g)
    
    Examples
    --------
    >>> # Water as H2O mixture
    >>> composition = {'Water': 1.0}  # Direct
    >>> # Or by elements (approximate)
    >>> # H: 2*1.008/(2*1.008+16) = 0.111
    >>> # O: 16/(2*1.008+16) = 0.889
    """
    energy_keV = np.atleast_1d(energy_keV)
    result = np.zeros_like(energy_keV, dtype=float)
    
    total_weight = sum(composition.values())
    
    for material, weight in composition.items():
        weight_frac = weight / total_weight
        data = get_attenuation_data(material)
        result += weight_frac * data.get_mu_rho(energy_keV)
    
    return result
