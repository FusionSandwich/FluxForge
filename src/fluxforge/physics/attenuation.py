"""
Photon attenuation helpers.

Wraps XCOM-based data in a small API for element/material attenuation
and transmission calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from fluxforge.data.xcom import AttenuationData, get_attenuation_data, mixture_mu_rho


ELEMENT_TO_MATERIAL = {
    "Fe": "Iron",
    "Pb": "Lead",
    "Al": "Aluminum",
    "Cu": "Copper",
    "Ge": "Germanium",
    "Si": "Silicon",
    "Cd": "Cadmium",
    "Te": "Tellurium",
    "NaI": "NaI",
    "Air": "Air",
    "Water": "Water",
    "Concrete": "Concrete",
}


def _resolve_material(label: str) -> str:
    if label in ELEMENT_TO_MATERIAL:
        return ELEMENT_TO_MATERIAL[label]
    return label


@dataclass
class AttenuationMaterial:
    """Material attenuation view."""

    name: str
    data: AttenuationData
    density_override: Optional[float] = None

    @property
    def density(self) -> float:
        return (
            self.density_override
            if self.density_override is not None
            else self.data.density
        )

    def mu_rho(self, energy_keV: Union[float, np.ndarray]) -> np.ndarray:
        return self.data.get_mu_rho(energy_keV)

    def mu(self, energy_keV: Union[float, np.ndarray]) -> np.ndarray:
        return self.mu_rho(energy_keV) * self.density

    def transmission(
        self, energy_keV: Union[float, np.ndarray], thickness_cm: float
    ) -> np.ndarray:
        mu = self.mu(energy_keV)
        return np.exp(-mu * thickness_cm)

    def half_value_layer(self, energy_keV: float) -> float:
        mu = float(self.mu(energy_keV))
        if mu <= 0:
            return np.inf
        return np.log(2.0) / mu


def get_material(label: str, density: Optional[float] = None) -> AttenuationMaterial:
    """Build an attenuation material from XCOM data."""
    name = _resolve_material(label)
    data = get_attenuation_data(name)
    return AttenuationMaterial(name=name, data=data, density_override=density)


def attenuation_factor(
    label: str,
    energy_keV: Union[float, np.ndarray],
    thickness_cm: float,
    density: Optional[float] = None,
) -> np.ndarray:
    material = get_material(label, density=density)
    return material.transmission(energy_keV, thickness_cm)


def mixture_attenuation_factor(
    composition: Dict[str, float],
    energy_keV: Union[float, np.ndarray],
    thickness_cm: float,
    density: float,
) -> np.ndarray:
    """Attenuation for a weighted mixture of XCOM materials."""
    mapped = {_resolve_material(key): val for key, val in composition.items()}
    mu_rho = mixture_mu_rho(mapped, energy_keV)
    mu = mu_rho * density
    return np.exp(-mu * thickness_cm)
