"""Synthetic, particle-agnostic parametric unfolding. No physical qualification."""

import json

import numpy as np

from fluxforge.solvers.parametric import (
    GaussianResponseModel,
    Parameter,
    ParametricProblem,
    integrate_spectrum,
    parametric_unfold,
)


def run():
    # Energy is in MeV; differential flux is per MeV. Responses multiply
    # integrated group flux. These are synthetic numbers, not evaluated data.
    edges = np.array([0, 0.2, 1, 3, 8])

    def spectrum(theta):
        return integrate_spectrum(
            lambda e, t: t[0] * np.sqrt(e) * np.exp(-e / t[1]), edges, theta
        )

    response = np.array([[1, 0.7, 0.2, 0], [0, 0.2, 1, 0.3], [0, 0, 0.2, 1]])
    truth = response @ spectrum([4, 1.2])
    shared = response.flatten() * 0.03
    likelihood = GaussianResponseModel(
        response, truth, np.diag([0.1, 0.12, 0.15]) ** 2, np.outer(shared, shared)
    )
    problem = ParametricProblem(
        likelihood,
        spectrum,
        [
            Parameter("amplitude", 0.1, 10, 0.3),
            Parameter("temperature_MeV", 0.2, 3, 0.06),
        ],
        "synthetic_maxwellian_v1",
    )
    result = parametric_unfold(
        problem, [[2, 0.7], [6, 2]], draws=1000, warmup=500, seed=20261017
    )
    return {
        "flux": result.flux.tolist(),
        "split_rhat": result.rhat.tolist(),
        "acceptance_rates": result.acceptance_rates.tolist(),
        "metadata": result.metadata,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
