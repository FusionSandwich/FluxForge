from __future__ import annotations

import numpy as np

from fluxforge.uncertainty.mc import propagate_measurement_uncertainty


def test_propagate_measurement_uncertainty_band_ordering():
    def linear_solver(sample):
        arr = np.asarray(sample, dtype=float)
        return (2.0 * arr).tolist()

    mean = [10.0, 20.0, 30.0]
    cov = [
        [1.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 9.0],
    ]

    band = propagate_measurement_uncertainty(
        solver=linear_solver,
        mean_measurement=mean,
        measurement_cov=cov,
        n_samples=300,
        confidence=0.68,
    )

    median = np.asarray(band.median, dtype=float)
    lower = np.asarray(band.lower, dtype=float)
    upper = np.asarray(band.upper, dtype=float)

    assert median.shape == lower.shape == upper.shape == (3,)
    assert np.all(lower <= median)
    assert np.all(median <= upper)
    # Sanity check around expected linear response.
    assert np.allclose(median, np.array(mean) * 2.0, rtol=0.2)
