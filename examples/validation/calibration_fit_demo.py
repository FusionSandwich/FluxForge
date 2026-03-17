#!/usr/bin/env python3
"""
Detector calibration fit demo (efficiency + resolution).
"""

import numpy as np

from fluxforge.analysis.detector_calibration import (
    EfficiencyPoint,
    fit_efficiency_curve,
    fit_resolution_curve,
)


def main() -> None:
    # Synthetic efficiency points
    coeffs = [-4.0, -0.8, 0.05]
    energies = np.array([100.0, 200.0, 400.0, 800.0], dtype=float)
    ln_e = np.log(energies)
    efficiencies = np.exp(coeffs[0] + coeffs[1] * ln_e + coeffs[2] * ln_e**2)

    points = [
        EfficiencyPoint(
            energy_keV=float(energy),
            net_counts=float(eff * 1e6),
            live_time_s=1.0,
            activity_bq=1e6,
            emission_probability=1.0,
            count_uncertainty=1.0,
        )
        for energy, eff in zip(energies, efficiencies)
    ]

    eff_fit = fit_efficiency_curve(points, degree=2, detector_id="demo")
    print("Efficiency coefficients:", eff_fit.coefficients)

    # Synthetic resolution curve
    res_coeffs = [1.0, 0.01, 1e-5]
    fwhm = np.sqrt(
        res_coeffs[0] + res_coeffs[1] * energies + res_coeffs[2] * energies**2
    )
    res_fit = fit_resolution_curve(energies, fwhm, model="sqrt_poly")
    print("Resolution coefficients:", res_fit.coefficients)


if __name__ == "__main__":
    main()
