"""
Stopping power helper utilities.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

from fluxforge.physics.stopping_power import Material, Projectile, total_stopping_power


def stopping_power_mass(
    energy_MeV: Union[float, np.ndarray],
    projectile: Projectile,
    composition: Dict[str, float],
    density_g_cm3: float,
) -> np.ndarray:
    """
    Mass stopping power in MeV/(mg/cm^2).
    """
    energy = np.asarray(energy_MeV, dtype=float)
    mat = Material(name="compound", elements=composition, density_g_cm3=density_g_cm3)
    return np.array([total_stopping_power(e, projectile, mat) for e in energy])


def stopping_power_linear(
    energy_MeV: Union[float, np.ndarray],
    projectile: Projectile,
    composition: Dict[str, float],
    density_g_cm3: float,
) -> np.ndarray:
    """
    Linear stopping power in MeV/cm.
    """
    energy = np.asarray(energy_MeV, dtype=float)
    mass_sp = stopping_power_mass(energy, projectile, composition, density_g_cm3)
    return mass_sp * density_g_cm3 * 1000.0
