"""Monte Carlo uncertainty propagation for flux solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from fluxforge.core.linalg import Matrix, Vector, multivariate_normal_samples, percentile


@dataclass
class MonteCarloBand:
    median: Vector
    lower: Vector
    upper: Vector


def propagate_measurement_uncertainty(
    solver: Callable[[Vector], Vector],
    mean_measurement: Vector,
    measurement_cov: Matrix,
    n_samples: int = 100,
    confidence: float = 0.68,
) -> MonteCarloBand:
    samples = multivariate_normal_samples(mean_measurement, measurement_cov, n_samples)
    spectra = [solver(sample) for sample in samples]
    lower_q = 50 * (1 - confidence)
    upper_q = 50 * (1 + confidence)
    return MonteCarloBand(
        median=percentile(spectra, 50),
        lower=percentile(spectra, lower_q),
        upper=percentile(spectra, upper_q),
    )
