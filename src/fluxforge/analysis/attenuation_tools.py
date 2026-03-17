"""
Attenuation helper utilities.

Provides simple multi-layer attenuation correction factors using XCOM data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from fluxforge.physics.attenuation import attenuation_factor, get_material


@dataclass
class AttenuationLayer:
    """Definition of a material layer for attenuation correction."""

    material: str
    thickness_cm: Optional[float] = None
    areal_density_g_cm2: Optional[float] = None
    density_g_cm3: Optional[float] = None


def layer_transmission(
    energy_keV: Union[float, np.ndarray],
    layer: AttenuationLayer,
) -> np.ndarray:
    """Transmission through a single attenuation layer."""
    if layer.thickness_cm is not None:
        return attenuation_factor(
            layer.material,
            energy_keV,
            thickness_cm=layer.thickness_cm,
            density=layer.density_g_cm3,
        )
    if layer.areal_density_g_cm2 is None:
        raise ValueError("Layer requires thickness_cm or areal_density_g_cm2.")

    material = get_material(layer.material, density=layer.density_g_cm3)
    mu_rho = material.mu_rho(energy_keV)
    return np.exp(-mu_rho * layer.areal_density_g_cm2)


def stacked_transmission(
    energy_keV: Union[float, np.ndarray],
    layers: Sequence[AttenuationLayer],
    self_absorption: bool = False,
) -> np.ndarray:
    """Transmission through a stack of layers."""
    energy = np.asarray(energy_keV, dtype=float)
    factor = np.ones_like(energy, dtype=float)
    for idx, layer in enumerate(layers):
        if self_absorption and idx == 0:
            material = get_material(layer.material, density=layer.density_g_cm3)
            if layer.thickness_cm is None and layer.areal_density_g_cm2 is None:
                raise ValueError(
                    "Self-absorption layer requires thickness or areal density."
                )
            if layer.areal_density_g_cm2 is None:
                areal = material.density * layer.thickness_cm
            else:
                areal = layer.areal_density_g_cm2
            mu_rho = material.mu_rho(energy)
            tau = mu_rho * areal
            with np.errstate(divide="ignore", invalid="ignore"):
                self_factor = np.where(tau > 0, (1.0 - np.exp(-tau)) / tau, 1.0)
            factor *= self_factor
        else:
            factor *= layer_transmission(energy, layer)
    return factor
