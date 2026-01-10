"""Minimal Poisson RMLE gamma unfolding demo.

Demonstrates detector-response unfolding with an explicit background component:
  y ~ Poisson(R mu + b)

Also demonstrates the minimum-viable calibration/response uncertainty propagation
hook using response-operator Monte Carlo via `PoissonRMLEConfig.response_sampler`.

Run:
  python examples/gamma_poisson_rmle_demo.py
"""

from __future__ import annotations

import numpy as np

from fluxforge.solvers.rmle import (
    ResponseMatrix,
    SpectrumData,
    PoissonRMLEConfig,
    PoissonPenalty,
    poisson_rmle_unfolding,
)


def main() -> None:
    rng = np.random.default_rng(0)

    n_channels = 120
    n_bins = 12

    # Simple block response, normalized columns.
    R = np.zeros((n_channels, n_bins))
    block = n_channels // n_bins
    for j in range(n_bins):
        lo = j * block
        hi = (j + 1) * block
        R[lo:hi, j] = 1.0

    response = ResponseMatrix(matrix=R).normalize_columns()

    # True emitted spectrum (mu) and a constant background.
    true_mu = np.zeros(n_bins)
    true_mu[[2, 6, 9]] = [40.0, 80.0, 25.0]
    b_true = 2.0

    expected = response.matrix @ true_mu + b_true
    counts = rng.poisson(np.maximum(expected, 0.0)).astype(float)

    spectrum = SpectrumData(counts=counts)

    # Response sampler: perturb response slightly to represent calibration/response uncertainty.
    base = response.matrix.copy()

    def response_sampler(r: np.random.Generator) -> ResponseMatrix:
        pert = base + r.normal(0.0, 0.02, size=base.shape)
        pert = np.clip(pert, 0.0, None)
        return ResponseMatrix(matrix=pert).normalize_columns()

    cfg = PoissonRMLEConfig(
        penalty=PoissonPenalty.SOBLEV_1,
        alpha=0.2,
        background_mode="constant",
        max_iterations=400,
        tolerance=1e-8,
        guardrail_max_reduced_chi2=1e6,
        mc_samples=50,
        random_seed=123,
        response_sampler=response_sampler,
    )

    res = poisson_rmle_unfolding(spectrum=spectrum, response=response, config=cfg)

    print("=== Poisson RMLE demo ===")
    print(f"Converged: {res.converged}")
    print(f"Reduced Pearson chi2: {res.diagnostics.get('reduced_pearson_chi2'):.3f}")
    print(f"Background mode: {res.diagnostics.get('background_mode')}")
    print(f"MC response sampling: {res.diagnostics.get('mc_response_sampling')}")
    print("\nTrue mu:")
    print(true_mu)
    print("\nEstimated mu:")
    print(np.round(res.solution, 3))
    print("\n1-sigma uncertainty (MC):")
    print(np.round(res.uncertainty, 3))


if __name__ == "__main__":
    main()
