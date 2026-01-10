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
        
        # Compute log ratios of measurements to predictions
        log_ratios = [math.log(max(m / p, floor)) for m, p in zip(measurements, predicted)]

        updated: Vector = []
        max_rel_change = 0.0
        for g in range(n_groups):
            # SAND-II/GRAVEL weighting: W[i,g] = data[i] * R[i,g] * phi[g] / predicted[i]
            # This properly weights by both the response and current flux estimate
            num = 0.0
            den = 0.0
            for i in range(n_meas):
                if measurements[i] > 0 and predicted[i] > floor:
                    W_ig = measurements[i] * response[i][g] * phi[g] / predicted[i]
                    num += W_ig * log_ratios[i]
                    den += W_ig
            
            if den <= floor:
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
    convergence_mode: str = "relative",  # "relative" or "ddJ" (Neutron-Unfolding style)
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
        convergence_mode: "relative" (default) uses max relative flux change,
            "ddJ" uses Neutron-Unfolding style second derivative criterion.
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
    
    # For ddJ convergence mode (Neutron-Unfolding style)
    J_prev = 0.0
    dJ_prev = 1.0

    for it in range(1, max_iters + 1):
        predicted = matmul(response, phi)
        predicted = [max(p, floor) for p in predicted]
        
        # Compute chi-squared
        chi2, residuals = _compute_chi_squared(response, phi, measurements, measurement_uncertainty, floor)
        chi2_per_dof = chi2 / max(n_meas - 1, 1)
        chi2_history.append(chi2_per_dof)
        
        # Compute J for ddJ mode (Neutron-Unfolding objective)
        if convergence_mode == "ddJ":
            sum_pred = sum(predicted)
            J = sum((p - m)**2 for p, m in zip(predicted, measurements)) / max(sum_pred, floor)
            dJ = J_prev - J
            ddJ = abs(dJ - dJ_prev)
            J_prev = J
            dJ_prev = dJ
        
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
        if convergence_mode == "ddJ" and ddJ < tolerance:
            converged = True
            if verbose:
                print(f"  MLEM converged at iter {it}: ddJ = {ddJ:.2e}, chi2/dof = {chi2_per_dof:.4f}")
            break
        elif convergence_mode == "relative" and max_rel_change < tolerance:
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


def gradient_descent(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    measurement_uncertainty: Optional[Vector] = None,
    max_iters: int = 10000,
    tolerance: float = 1e-6,
    chi2_tolerance: float = 0.01,
    floor: float = 1e-12,
    learning_rate: float = 1.0,
    smoothness_weight: float = 0.01,
    auto_scale: bool = True,
    verbose: bool = False,
) -> IterativeSolution:
    """Gradient descent solver with smoothness regularization (SpecKit-style).

    This solver minimizes: L = MSE(Ax - b) + λ * smoothness_penalty
    where smoothness_penalty = mean(diff(log(x))^2)

    Inspired by the SpecKit neural network approach but without the GUI.

    Args:
        response: Response matrix R_{i,g} (n_meas x n_groups).
        measurements: Measured reaction rates y_i.
        initial_flux: Starting flux guess. Defaults to uniform.
        measurement_uncertainty: Optional 1-sigma uncertainties.
        max_iters: Maximum iterations to perform.
        tolerance: Loss change threshold for convergence.
        chi2_tolerance: Chi-squared per DOF threshold for convergence.
        floor: Minimum flux value to ensure positivity.
        learning_rate: Step size for gradient updates.
        smoothness_weight: Weight λ for smoothness regularization term.
        auto_scale: Auto-scale flux to match measurement magnitude.
        verbose: Print convergence progress.
    """
    n_groups = len(response[0])
    n_meas = len(measurements)
    
    # Initialize flux
    if initial_flux is not None:
        phi = [max(x, floor) for x in initial_flux]
    else:
        avg_meas = sum(measurements) / len(measurements)
        phi = [avg_meas / n_groups for _ in range(n_groups)]
    
    history: List[Vector] = [phi[:]]
    chi2_history: List[float] = []
    loss_history: List[float] = []
    converged = False
    
    # Track minimum loss for early stopping
    min_loss = float('inf')
    no_improvement_count = 0
    patience = 1000  # Reset if no improvement for this many iterations
    
    for it in range(1, max_iters + 1):
        # Compute prediction: Ax
        predicted = matmul(response, phi)
        predicted = [max(p, floor) for p in predicted]
        
        # Compute error: (Ax - b)
        errors = [p - m for p, m in zip(predicted, measurements)]
        
        # MSE loss
        mse_loss = sum(e * e for e in errors) / n_meas
        
        # Smoothness penalty on log(phi)
        log_phi = [math.log(max(p, floor)) for p in phi]
        log_diffs = [log_phi[g+1] - log_phi[g] for g in range(n_groups - 1)]
        smoothness_penalty = sum(d * d for d in log_diffs) / max(len(log_diffs), 1)
        
        # Total loss
        loss = mse_loss + smoothness_weight * smoothness_penalty
        loss_history.append(loss)
        
        # Compute chi-squared
        chi2, residuals = _compute_chi_squared(response, phi, measurements, measurement_uncertainty, floor)
        chi2_per_dof = chi2 / max(n_meas - 1, 1)
        chi2_history.append(chi2_per_dof)
        
        if verbose and it % 500 == 0:
            print(f"  GD iter {it}: loss = {loss:.4e}, chi2/dof = {chi2_per_dof:.4f}")
        
        # Auto-scale on first iteration
        if it == 1 and auto_scale:
            sum_pred = sum(predicted)
            sum_meas = sum(measurements)
            if sum_pred > 0 and sum_meas > 0:
                scale_factor = sum_meas / sum_pred
                phi = [p * scale_factor for p in phi]
                if verbose:
                    print(f"  Auto-scaled flux by factor {scale_factor:.4e}")
                continue  # Re-evaluate with scaled flux
        
        # Gradient: dL/dx = (2/n) * A^T @ (Ax - b)
        gradient = []
        for g in range(n_groups):
            grad_g = sum(2.0 * errors[i] * response[i][g] / n_meas for i in range(n_meas))
            
            # Add smoothness gradient contribution
            if smoothness_weight > 0 and n_groups > 1:
                if g > 0:
                    dlog_left = log_phi[g] - log_phi[g-1]
                    grad_g += 2.0 * smoothness_weight * dlog_left / (phi[g] * (n_groups - 1))
                if g < n_groups - 1:
                    dlog_right = log_phi[g] - log_phi[g+1]
                    grad_g += 2.0 * smoothness_weight * dlog_right / (phi[g] * (n_groups - 1))
            
            gradient.append(grad_g)
        
        # Update flux: x = x - lr * gradient
        max_rel_change = 0.0
        updated = []
        for g in range(n_groups):
            new_phi = phi[g] - learning_rate * gradient[g]
            new_phi = max(new_phi, floor)  # Enforce positivity
            max_rel_change = max(max_rel_change, abs(new_phi - phi[g]) / max(phi[g], floor))
            updated.append(new_phi)
        
        phi = updated
        history.append(phi[:])
        
        # Check for convergence
        if chi2_per_dof < chi2_tolerance:
            converged = True
            if verbose:
                print(f"  GD converged at iter {it}: chi2/dof = {chi2_per_dof:.4f}")
            break
        
        # Check loss plateau
        if len(loss_history) >= 500:
            recent_losses = loss_history[-500:]
            loss_range = max(recent_losses) - min(recent_losses)
            if loss_range / max(max(recent_losses), floor) < 1e-5:
                converged = True
                if verbose:
                    print(f"  GD converged at iter {it}: loss plateau detected")
                break
        
        # Track improvement and reset if stuck
        if loss < min_loss:
            min_loss = loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            no_improvement_count = 0
            min_loss = float('inf')
            # Reset with some noise for exploration
            if verbose:
                print(f"  GD iter {it}: resetting due to no improvement")
    
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
