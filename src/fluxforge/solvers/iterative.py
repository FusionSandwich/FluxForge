"""Iterative unfolding algorithms (GRAVEL and MLEM)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from fluxforge.core.linalg import Matrix, Vector, elementwise_maximum, matmul


@dataclass
class IterativeSolution:
    """Container for iterative solver results."""

    flux: Vector
    history: List[Vector]
    iterations: int
    converged: bool


def _default_flux(n_groups: int, scale: float = 1.0) -> Vector:
    return [scale for _ in range(n_groups)]


def gravel(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    measurement_uncertainty: Optional[Vector] = None,
    max_iters: int = 500,
    tolerance: float = 1e-6,
    floor: float = 1e-20,
) -> IterativeSolution:
    """GRAVEL algorithm (log-space SAND-II variant) for spectrum unfolding.

    Args:
        response: Response matrix R_{i,g}.
        measurements: Measured reaction rates y_i.
        initial_flux: Starting flux guess. Defaults to uniform.
        measurement_uncertainty: Optional 1-sigma uncertainties on y_i for weighting.
        max_iters: Maximum iterations to perform.
        tolerance: Relative max change threshold for convergence.
        floor: Minimum flux/prediction value to avoid divide-by-zero.
    """

    n_groups = len(response[0])
    phi = initial_flux[:] if initial_flux is not None else _default_flux(n_groups)
    phi = elementwise_maximum(phi, floor)
    weights = (
        [1.0 / (u * u) if u > 0 else 0.0 for u in measurement_uncertainty]
        if measurement_uncertainty
        else [1.0 for _ in measurements]
    )

    history: List[Vector] = [phi[:]]
    converged = False

    for it in range(1, max_iters + 1):
        predicted = matmul(response, phi)  # type: ignore[arg-type]
        predicted = [max(p, floor) for p in predicted]
        ratios = [m / p for m, p in zip(measurements, predicted)]
        log_ratios = [math.log(r) for r in ratios]

        updated: Vector = []
        max_rel_change = 0.0
        for g in range(n_groups):
            num = sum(weights[i] * response[i][g] * log_ratios[i] for i in range(len(measurements)))
            den = sum(weights[i] * response[i][g] for i in range(len(measurements)))
            if den <= 0:
                updated.append(phi[g])
                continue
            factor = math.exp(num / den)
            new_phi = max(phi[g] * factor, floor)
            max_rel_change = max(max_rel_change, abs(new_phi - phi[g]) / max(phi[g], floor))
            updated.append(new_phi)

        phi = updated
        history.append(phi[:])
        if max_rel_change < tolerance:
            converged = True
            return IterativeSolution(flux=phi, history=history, iterations=it, converged=converged)

    return IterativeSolution(flux=phi, history=history, iterations=max_iters, converged=converged)


def mlem(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    max_iters: int = 500,
    tolerance: float = 1e-6,
    floor: float = 1e-20,
) -> IterativeSolution:
    """Maximum-likelihood expectation maximization (MLEM) unfolding.

    Args:
        response: Response matrix R_{i,g}.
        measurements: Measured reaction rates y_i.
        initial_flux: Starting flux guess. Defaults to uniform.
        max_iters: Maximum iterations to perform.
        tolerance: Relative max change threshold for convergence.
        floor: Minimum flux/prediction value to avoid divide-by-zero.
    """

    n_groups = len(response[0])
    phi = initial_flux[:] if initial_flux is not None else _default_flux(n_groups)
    phi = elementwise_maximum(phi, floor)

    history: List[Vector] = [phi[:]]
    converged = False

    for it in range(1, max_iters + 1):
        predicted = matmul(response, phi)  # type: ignore[arg-type]
        predicted = [max(p, floor) for p in predicted]

        updated: Vector = []
        max_rel_change = 0.0
        for g in range(n_groups):
            numerator = sum(response[i][g] * measurements[i] / predicted[i] for i in range(len(measurements)))
            denominator = sum(response[i][g] for i in range(len(measurements)))
            if denominator <= 0:
                updated.append(phi[g])
                continue
            new_phi = max(phi[g] * numerator / denominator, floor)
            max_rel_change = max(max_rel_change, abs(new_phi - phi[g]) / max(phi[g], floor))
            updated.append(new_phi)

        phi = updated
        history.append(phi[:])
        if max_rel_change < tolerance:
            converged = True
            return IterativeSolution(flux=phi, history=history, iterations=it, converged=converged)

    return IterativeSolution(flux=phi, history=history, iterations=max_iters, converged=converged)
