import numpy as np

from fluxforge.physics.attenuation import (
    attenuation_factor,
    get_material,
    mixture_attenuation_factor,
)


def test_material_mu_and_transmission():
    material = get_material("Fe")
    mu = material.mu(511.0)

    assert mu > 0
    transmission = material.transmission(511.0, 0.3)
    assert 0 < transmission < 1


def test_attenuation_factor_matches_material():
    material = get_material("Iron")
    direct = material.transmission(661.7, 1.0)
    wrapped = attenuation_factor("Iron", 661.7, 1.0)

    assert np.isclose(direct, wrapped, rtol=1e-9, atol=0.0)


def test_mixture_attenuation_factor():
    composition = {"Iron": 0.5, "Lead": 0.5}
    transmission = mixture_attenuation_factor(composition, 1000.0, 0.2, density=8.0)

    assert 0 < transmission < 1
