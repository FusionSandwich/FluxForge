import numpy as np

from fluxforge.analysis.attenuation_tools import AttenuationLayer, stacked_transmission
from fluxforge.analysis.detector_calibration import ResolutionCurve
from fluxforge.analysis.efficiency_models import (
    semi_empirical_efficiency,
    semi_empirical_efficiency_uncertainty,
)


def test_semi_empirical_efficiency_matches_curie_example():
    energies = 50.0 * np.arange(1, 10)
    coeffs = [0.02, 8.3, 2.1, 1.66, 0.4]
    expected = np.array([
        0.01855424, 0.01556396, 0.01262621, 0.01020047, 0.00813456,
        0.00662029, 0.00554586, 0.00476659, 0.00418103,
    ])

    values = semi_empirical_efficiency(energies, coeffs)
    assert np.allclose(values, expected, rtol=0.02, atol=0.0)


def test_semi_empirical_efficiency_uncertainty_matches_curie_example():
    energies = 50.0 * np.arange(1, 10)
    coeffs = [0.02, 8.3, 2.1, 1.66, 0.4]
    covariance = np.array([
        [5.038e-02, 3.266e-02, -2.151e-02, -4.869e-05, -7.748e-03],
        [3.266e-02, 2.144e-02, -1.416e-02, -3.416e-05, -4.137e-03],
        [-2.151e-02, -1.416e-02, 9.367e-03, 2.294e-05, 2.569e-03],
        [-4.869e-05, -3.416e-05, 2.294e-05, 5.411e-07, -1.165e-04],
        [-7.748e-03, -4.137e-03, 2.569e-03, -1.165e-04, 3.332e-02],
    ])
    expected = np.array([
        0.20820374, 0.17452163, 0.14139249, 0.11406261, 0.09090285,
        0.07397037, 0.06196974, 0.05327103, 0.04673696,
    ])

    values = semi_empirical_efficiency_uncertainty(energies, coeffs, covariance)
    assert np.allclose(values, expected, rtol=0.02, atol=0.0)


def test_resolution_linear_matches_curie_example():
    channels = 100.0 * np.arange(1, 10)
    expected = np.array([2.04, 2.08, 2.12, 2.16, 2.2, 2.24, 2.28, 2.32, 2.36])
    curve = ResolutionCurve(model="linear", coefficients=[2.0, 4e-4])
    values = curve.fwhm(channels)
    assert np.allclose(values, expected, rtol=0.0, atol=1e-6)


def test_attenuation_stack_matches_curie_example():
    energies = 100.0 * np.arange(1, 10)
    layers = [
        AttenuationLayer(material="Fe", thickness_cm=0.1),
        AttenuationLayer(material="Water", thickness_cm=0.5),
    ]
    expected = np.array([
        0.92484814, 0.94256307, 0.94862726, 0.95217360, 0.95454280,
        0.95623520, 0.95748897, 0.95854508, 0.95932516,
    ])
    values = stacked_transmission(energies, layers, self_absorption=True)
    assert np.allclose(values, expected, rtol=0.02, atol=0.0)
