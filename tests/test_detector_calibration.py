import numpy as np

from fluxforge.analysis.detector_calibration import (
    EfficiencyPoint,
    fit_efficiency_curve,
    fit_resolution_curve,
)


def test_efficiency_point_computation():
    point = EfficiencyPoint(
        energy_keV=661.7,
        net_counts=1000.0,
        live_time_s=100.0,
        activity_bq=1000.0,
        emission_probability=0.5,
    )

    efficiency, uncertainty = point.efficiency()
    assert np.isclose(efficiency, 0.02, rtol=0, atol=1e-6)
    assert uncertainty > 0


def test_fit_efficiency_curve_recovers_coefficients():
    coeffs = [-4.0, -0.8, 0.05]
    energies = np.array([100.0, 200.0, 400.0, 800.0], dtype=float)
    ln_e = np.log(energies)
    eff = np.exp(coeffs[0] + coeffs[1] * ln_e + coeffs[2] * ln_e**2)

    points = []
    for energy, efficiency in zip(energies, eff):
        points.append(
            EfficiencyPoint(
                energy_keV=float(energy),
                net_counts=float(efficiency * 1e6),
                live_time_s=1.0,
                activity_bq=1e6,
                emission_probability=1.0,
                count_uncertainty=1.0,
            )
        )

    fit = fit_efficiency_curve(points, degree=2)
    assert np.allclose(fit.coefficients, coeffs, rtol=0.05, atol=0.05)


def test_fit_resolution_curve_sqrt_poly():
    coeffs = [1.0, 0.01, 1e-5]
    energies = np.array([100.0, 500.0, 1000.0, 1500.0], dtype=float)
    fwhm = np.sqrt(coeffs[0] + coeffs[1] * energies + coeffs[2] * energies**2)

    fit = fit_resolution_curve(energies, fwhm, model="sqrt_poly")
    assert np.allclose(fit.coefficients, coeffs, rtol=0.1, atol=0.1)
