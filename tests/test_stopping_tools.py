import numpy as np

from fluxforge.analysis.stopping_tools import stopping_power_linear, stopping_power_mass
from fluxforge.physics.stopping_power import Projectile


def test_stopping_power_scaling():
    energy = np.array([5.0, 10.0])
    density = 2.7
    composition = {"Al": 1.0}

    mass_sp = stopping_power_mass(energy, Projectile.PROTON, composition, density)
    linear_sp = stopping_power_linear(energy, Projectile.PROTON, composition, density)

    assert np.all(mass_sp > 0.0)
    assert np.all(linear_sp > 0.0)
    assert np.allclose(linear_sp, mass_sp * density * 1000.0, rtol=1e-6, atol=0.0)
