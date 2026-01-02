"""Iterative unfolding algorithms (GRAVEL and MLEM).

These implement the standard iterative spectrum unfolding methods:
- GRAVEL: Log-space multiplicative update (SAND-II variant)
- MLEM: Maximum-Likelihood Expectation Maximization

Enhanced with:
- Relaxation/damping for improved convergence
- Chi-squared convergence monitoring
- Support for MCNP spectrum as initial guess
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from fluxforge.core.linalg import Matrix, Vector, elementwise_maximum, matmul


@dataclass
class IterativeSolution:
    """Container for iterative solver results."""

    flux: Vector
    history: List[Vector]
    iterations: int
    converged: bool
    chi_squared: float = 0.0
    chi_squared_history: List[float] = field(default_factory=list)
    final_residuals: Optional[Vector] = None


def _default_flux(n_groups: int, scale: float = 1.0) -> Vector:
    return [scale for _ in range(n_groups)]


def _compute_chi_squared(response: Matrix, flux: Vector, measurements: Vector,
                         uncertainties: Optional[Vector] = None, floor: float = 1e-20) -> Tuple[float, Vector]:
    """Compute chi-squared and residuals for current flux estimate."""
    predicted = matmul(response, flux)
    predicted = [max(p, floor) for p in predicted]
    
    residuals = []
    chi2 = 0.0
    for i, (m, p) in enumerate(zip(measurements, predicted)):
        unc = uncertainties[i] if uncertainties else max(m * 0.1, floor)
        if unc > 0:
            r = (m - p) / unc
            residuals.append(r)
            chi2 += r * r
        else:
            residuals.append(0.0)
    
    return chi2, residuals


def gravel(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    measurement_uncertainty: Optional[Vector] = None,
    max_iters: int = 1000,
    tolerance: float = 1e-4,
    chi2_tolerance: float = 0.01,
    floor: float = 1e-20,
    relaxation: float = 0.7,
    verbose: bool = False,
) -> IterativeSolution:
    """GRAVEL algorithm (log-space SAND-II variant) for spectrum unfolding.

    Args:
        response: Response matrix R_{i,g}.
        measurements: Measured reaction rates y_i.
        initial_flux: Starting flux guess. Defaults to uniform.
        measurement_uncertainty: Optional 1-sigma uncertainties on y_i for weighting.
        max_iters: Maximum iterations to perform.
        tolerance: Relative max change threshold for convergence.
        chi2_tolerance: Chi-squared per DOF threshold for convergence.
        floor: Minimum flux/prediction value to avoid divide-by-zero.
        relaxation: Under-relaxation factor (0-1). Lower = more stable, slower.
        verbose: Print convergence progress.
    """

    n_groups = len(response[0])
    n_meas = len(measurements)
    
    # Initialize flux
    if initial_flux is not None:
        phi = initial_flux[:]
    else:
        # Better default: scale to match average measurement
        avg_meas = sum(measurements) / len(measurements)
        phi = _default_flux(n_groups, scale=avg_meas / n_groups)
    
    phi = elementwise_maximum(phi, floor)
    
    # Weights from uncertainties
    weights = (
        [1.0 / (u * u) if u > 0 else 0.0 for u in measurement_uncertainty]
        if measurement_uncertainty
        else [1.0 for _ in measurements]
    )

    history: List[Vector] = [phi[:]]
    chi2_history: List[float] = []
    converged = False

    for it in range(1, max_iters + 1):
        predicted = matmul(response, phi)
        predicted = [max(p, floor) for p in predicted]
        
        # Compute chi-squared
        chi2, residuals = _compute_chi_squared(response, phi, measurements, measurement_uncertainty, floor)
        chi2_per_dof = chi2 / max(n_meas - 1, 1)
        chi2_history.append(chi2_per_dof)
        
        if verbose and it % 100 == 0:
            print(f"  GRAVEL iter {it}: chi2/dof = {chi2_per_dof:.4f}")
        
        ratios = [m / p for m, p in zip(measurements, predicted)]
        log_ratios = [math.log(max(r, floor)) for r in ratios]

        updated: Vector = []
        max_rel_change = 0.0
        for g in range(n_groups):
            num = sum(weights[i] * response[i][g] * log_ratios[i] for i in range(n_meas))
            den = sum(weights[i] * response[i][g] for i in range(n_meas))
            if den <= 0:
                updated.append(phi[g])
                continue
            
            # Calculate update with relaxation
            factor = math.exp(num / den)
            target_phi = phi[g] * factor
            new_phi = phi[g] + relaxation * (target_phi - phi[g])
            new_phi = max(new_phi, floor)
            
            max_rel_change = max(max_rel_change, abs(new_phi - phi[g]) / max(phi[g], floor))
            updated.append(new_phi)

        phi = updated
        history.append(phi[:])
        
        # Check convergence criteria
        if max_rel_change < tolerance:
            converged = True
            if verbose:
                print(f"  GRAVEL converged at iter {it}: rel_change = {max_rel_change:.2e}, chi2/dof = {chi2_per_dof:.4f}")
            break
        
        if chi2_per_dof < chi2_tolerance:
            converged = True
            if verbose:
                print(f"  GRAVEL converged at iter {it}: chi2/dof = {chi2_per_dof:.4f}")
            break

    final_chi2, final_residuals = _compute_chi_squared(response, phi, measurements, measurement_uncertainty, floor)
    
    return IterativeSolution(
        flux=phi, 
        history=history, 
        iterations=it if 'it' in dir() else max_iters, 
        converged=converged,
        chi_squared=final_chi2 / max(n_meas - 1, 1),
        chi_squared_history=chi2_history,
        final_residuals=final_residuals,
    )


def mlem(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    measurement_uncertainty: Optional[Vector] = None,
    max_iters: int = 1000,
    tolerance: float = 1e-4,
    chi2_tolerance: float = 0.01,
    floor: float = 1e-20,
    relaxation: float = 0.8,
    verbose: bool = False,
) -> IterativeSolution:
    """Maximum-likelihood expectation maximization (MLEM) unfolding.

    Args:
        response: Response matrix R_{i,g}.
        measurements: Measured reaction rates y_i.
        initial_flux: Starting flux guess. Defaults to uniform.
        measurement_uncertainty: Optional 1-sigma uncertainties for chi2 calculation.
        max_iters: Maximum iterations to perform.
        tolerance: Relative max change threshold for convergence.
        chi2_tolerance: Chi-squared per DOF threshold for convergence.
        floor: Minimum flux/prediction value to avoid divide-by-zero.
        relaxation: Under-relaxation factor (0-1). Lower = more stable, slower.
        verbose: Print convergence progress.
    """

    n_groups = len(response[0])
    n_meas = len(measurements)
    
    # Initialize flux
    if initial_flux is not None:
        phi = initial_flux[:]
    else:
        avg_meas = sum(measurements) / len(measurements)
        phi = _default_flux(n_groups, scale=avg_meas / n_groups)
    
    phi = elementwise_maximum(phi, floor)

    history: List[Vector] = [phi[:]]
    chi2_history: List[float] = []
    converged = False

    for it in range(1, max_iters + 1):
        predicted = matmul(response, phi)
        predicted = [max(p, floor) for p in predicted]
        
        # Compute chi-squared
        chi2, residuals = _compute_chi_squared(response, phi, measurements, measurement_uncertainty, floor)
        chi2_per_dof = chi2 / max(n_meas - 1, 1)
        chi2_history.append(chi2_per_dof)
        
        if verbose and it % 100 == 0:
            print(f"  MLEM iter {it}: chi2/dof = {chi2_per_dof:.4f}")

        updated: Vector = []
        max_rel_change = 0.0
        for g in range(n_groups):
            numerator = sum(response[i][g] * measurements[i] / predicted[i] for i in range(n_meas))
            denominator = sum(response[i][g] for i in range(n_meas))
            if denominator <= 0:
                updated.append(phi[g])
                continue
            
            # Calculate update with relaxation
            target_phi = phi[g] * numerator / denominator
            new_phi = phi[g] + relaxation * (target_phi - phi[g])
            new_phi = max(new_phi, floor)
            
            max_rel_change = max(max_rel_change, abs(new_phi - phi[g]) / max(phi[g], floor))
            updated.append(new_phi)

        phi = updated
        history.append(phi[:])
        
        # Check convergence criteria
        if max_rel_change < tolerance:
            converged = True
            if verbose:
                print(f"  MLEM converged at iter {it}: rel_change = {max_rel_change:.2e}, chi2/dof = {chi2_per_dof:.4f}")
            break
        
        if chi2_per_dof < chi2_tolerance:
            converged = True
            if verbose:
                print(f"  MLEM converged at iter {it}: chi2/dof = {chi2_per_dof:.4f}")
            break

    final_chi2, final_residuals = _compute_chi_squared(response, phi, measurements, measurement_uncertainty, floor)
    
    return IterativeSolution(
        flux=phi, 
        history=history, 
        iterations=it if 'it' in dir() else max_iters, 
        converged=converged,
        chi_squared=final_chi2 / max(n_meas - 1, 1),
        chi_squared_history=chi2_history,
        final_residuals=final_residuals,
    )
