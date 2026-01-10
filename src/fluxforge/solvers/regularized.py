"""
Regularized Spectrum Solver

Provides Tikhonov and log-smoothness regularized spectrum unfolding
with uncertainty quantification and automatic regularization parameter selection.

Based on SpecKit's neutron_spectrum_solver approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize
from scipy.stats import chi2


@dataclass
class RegularizedSolution:
    """Solution from regularized spectrum unfolding."""
    
    spectrum: np.ndarray
    uncertainties: np.ndarray
    residuals: np.ndarray
    chi_squared: float
    regularization_param: float
    n_iterations: int
    converged: bool
    loss_history: List[float] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    
    @property
    def reduced_chi_squared(self) -> float:
        """Reduced chi-squared (chi²/dof)."""
        dof = max(len(self.residuals) - 1, 1)
        return self.chi_squared / dof


def log_smoothness_penalty(spectrum: np.ndarray, eps: float = 1e-12) -> float:
    """
    Calculate log-smoothness penalty: mean of squared differences in log-space.
    
    penalty = mean((log(φ_i+1) - log(φ_i))²)
    
    This penalizes rapid variations in the spectrum on a logarithmic scale,
    which is appropriate for neutron spectra spanning many orders of magnitude.
    """
    log_spec = np.log(spectrum + eps)
    diffs = np.diff(log_spec)
    return float(np.mean(diffs ** 2))


def first_derivative_penalty(spectrum: np.ndarray) -> float:
    """First derivative (gradient) penalty."""
    diffs = np.diff(spectrum)
    return float(np.sum(diffs ** 2))


def second_derivative_penalty(spectrum: np.ndarray) -> float:
    """Second derivative (curvature) penalty - Tikhonov regularization."""
    if len(spectrum) < 3:
        return 0.0
    d2 = spectrum[:-2] - 2 * spectrum[1:-1] + spectrum[2:]
    return float(np.sum(d2 ** 2))


def build_tikhonov_matrix(n: int, order: int = 2) -> np.ndarray:
    """
    Build Tikhonov regularization matrix L.
    
    Parameters
    ----------
    n : int
        Size of spectrum
    order : int
        Order of derivative (0, 1, or 2)
    
    Returns
    -------
    np.ndarray
        Regularization matrix L such that ||Lx||² is the penalty
    """
    if order == 0:
        return np.eye(n)
    elif order == 1:
        L = np.zeros((n-1, n))
        for i in range(n-1):
            L[i, i] = -1
            L[i, i+1] = 1
        return L
    elif order == 2:
        L = np.zeros((n-2, n))
        for i in range(n-2):
            L[i, i] = 1
            L[i, i+1] = -2
            L[i, i+2] = 1
        return L
    else:
        raise ValueError(f"Order must be 0, 1, or 2, got {order}")


def tikhonov_solve(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    order: int = 2,
    x0: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Solve Tikhonov-regularized least squares problem.
    
    Minimizes ||Ax - b||² + α||Lx||²
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n)
    b : np.ndarray
        Measurement vector (m,)
    alpha : float
        Regularization parameter
    order : int
        Regularization order (0, 1, or 2)
    x0 : np.ndarray, optional
        Not used, for API compatibility
    
    Returns
    -------
    np.ndarray
        Regularized solution
    """
    n = A.shape[1]
    L = build_tikhonov_matrix(n, order)
    
    # Augmented system: [A; sqrt(alpha)*L] x = [b; 0]
    A_aug = np.vstack([A, np.sqrt(alpha) * L])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    
    # Solve using least squares
    x, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    
    # Enforce non-negativity
    x = np.maximum(x, 0)
    
    return x


def gradient_descent_regularized(
    A: np.ndarray,
    b: np.ndarray,
    b_error: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    reg_weight: float = 0.01,
    reg_type: str = 'log_smooth',
    learning_rate: float = 1.0,
    max_epochs: int = 100000,
    loss_threshold: Optional[float] = None,
    patience: int = 1000,
    min_improvement: float = 1e-5,
    callback: Optional[Callable] = None
) -> RegularizedSolution:
    """
    Gradient descent with regularization for spectrum unfolding.
    
    Minimizes: ||Ax - b||² + λ * R(x)
    
    where R(x) is a regularization term (log-smoothness, first derivative, etc.)
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix (m x n)
    b : np.ndarray
        Measurement vector (m,)
    b_error : np.ndarray, optional
        Measurement uncertainties
    x0 : np.ndarray, optional
        Initial spectrum guess
    reg_weight : float
        Regularization weight λ
    reg_type : str
        Type of regularization: 'log_smooth', 'first_deriv', 'second_deriv'
    learning_rate : float
        Initial learning rate
    max_epochs : int
        Maximum iterations
    loss_threshold : float, optional
        Stop when loss falls below this. If None, computed from chi² distribution.
    patience : int
        Epochs without improvement before early stopping
    min_improvement : float
        Minimum relative improvement to reset patience counter
    callback : Callable, optional
        Called each epoch with (epoch, x, loss)
    
    Returns
    -------
    RegularizedSolution
        Solution with spectrum, uncertainties, and convergence info
    """
    m, n = A.shape
    
    # Initialize spectrum
    if x0 is None:
        x = np.random.uniform(1e-6, 1.0, size=n)
    else:
        x = np.maximum(x0.flatten(), 1e-6)
    
    # Compute loss threshold from chi² distribution if not provided
    if loss_threshold is None:
        confidence = 0.95
        loss_threshold = chi2.ppf(confidence, m)
    
    # Select regularization function
    if reg_type == 'log_smooth':
        reg_func = log_smoothness_penalty
    elif reg_type == 'first_deriv':
        reg_func = first_derivative_penalty
    elif reg_type == 'second_deriv':
        reg_func = second_derivative_penalty
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
    
    loss_history = []
    min_loss = float('inf')
    no_improvement = 0
    converged = False
    
    # Scaling factor from first iteration
    scaling_factor = 1.0
    
    for epoch in range(max_epochs):
        # Forward pass
        Ax = np.dot(A, x)
        error = Ax - b
        
        # Data fidelity loss (chi²-like)
        if b_error is not None:
            weighted_error = error / np.maximum(b_error, 1e-12)
            data_loss = float(np.mean(weighted_error ** 2))
        else:
            data_loss = float(np.mean(error ** 2))
        
        # Regularization
        reg_penalty = reg_func(x)
        loss = data_loss + reg_weight * reg_penalty
        
        loss_history.append(loss)
        
        # Auto-scale on first iteration
        if epoch == 0 and np.any(Ax > 0) and np.any(b > 0):
            scaling_values = np.abs(Ax / np.maximum(b, 1e-12))
            scaling_factor = float(np.mean(scaling_values[scaling_values > 0]))
            if scaling_factor > 0:
                x = x / scaling_factor
                continue
        
        # Check convergence
        if loss < loss_threshold:
            converged = True
            break
        
        # Gradient of data fidelity
        if b_error is not None:
            grad = 2 * np.dot(A.T, weighted_error / np.maximum(b_error, 1e-12)) / m
        else:
            grad = 2 * np.dot(A.T, error) / m
        
        # Gradient of log-smoothness regularization
        if reg_type == 'log_smooth':
            eps = 1e-12
            log_x = np.log(x + eps)
            grad_reg = np.zeros(n)
            for i in range(1, n-1):
                grad_reg[i] = 2 * reg_weight * (
                    (log_x[i] - log_x[i-1]) - (log_x[i+1] - log_x[i])
                ) / (x[i] + eps)
        else:
            # Numerical gradient for other regularization types
            grad_reg = np.zeros(n)
        
        # Update
        x = x - learning_rate * (grad + grad_reg)
        x = np.maximum(x, 1e-12)  # Non-negativity
        
        # Early stopping check
        if loss < min_loss:
            if (min_loss - loss) / min_loss > min_improvement:
                no_improvement = 0
            else:
                no_improvement += 1
            min_loss = loss
        else:
            no_improvement += 1
        
        if no_improvement >= patience:
            # Reduce learning rate and reset
            learning_rate *= 0.5
            no_improvement = 0
            if learning_rate < 1e-6:
                break
        
        # Check for stagnation
        if len(loss_history) >= 500:
            recent = loss_history[-500:]
            if (max(recent) - min(recent)) / max(recent) < 1e-5:
                break
        
        if callback is not None:
            callback(epoch, x, loss)
    
    # Compute final residuals and uncertainties
    Ax_final = np.dot(A, x)
    residuals = Ax_final - b
    chi_squared = float(np.sum((residuals / np.maximum(b_error if b_error is not None else np.sqrt(b), 1e-12)) ** 2))
    
    # Estimate uncertainties from residuals
    # Simple approach: scale by residual standard deviation
    residual_std = np.std(residuals)
    uncertainties = x * (residual_std / np.maximum(np.mean(b), 1e-12))
    
    return RegularizedSolution(
        spectrum=x,
        uncertainties=uncertainties,
        residuals=residuals,
        chi_squared=chi_squared,
        regularization_param=reg_weight,
        n_iterations=epoch + 1,
        converged=converged,
        loss_history=loss_history,
        details={
            'scaling_factor': scaling_factor,
            'final_loss': loss_history[-1] if loss_history else 0.0,
            'loss_threshold': loss_threshold
        }
    )


def l_curve_corner(
    A: np.ndarray,
    b: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    order: int = 2,
    n_points: int = 20
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Find optimal regularization parameter using L-curve criterion.
    
    The L-curve plots ||Ax-b|| vs ||Lx|| for different alpha values.
    The optimal alpha is at the corner of maximum curvature.
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix
    b : np.ndarray
        Measurements
    alphas : np.ndarray, optional
        Regularization parameters to test
    order : int
        Regularization order
    n_points : int
        Number of alpha values if not provided
    
    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        Optimal alpha, residual norms, solution norms
    """
    if alphas is None:
        alphas = np.logspace(-6, 2, n_points)
    
    L = build_tikhonov_matrix(A.shape[1], order)
    
    residual_norms = []
    solution_norms = []
    
    for alpha in alphas:
        x = tikhonov_solve(A, b, alpha, order)
        residual_norms.append(np.linalg.norm(A @ x - b))
        solution_norms.append(np.linalg.norm(L @ x))
    
    residual_norms = np.array(residual_norms)
    solution_norms = np.array(solution_norms)
    
    # Find corner by maximum curvature
    # Using log-log space for better curvature estimation
    log_rho = np.log(residual_norms + 1e-12)
    log_eta = np.log(solution_norms + 1e-12)
    
    # Curvature approximation
    curvatures = np.zeros(len(alphas))
    for i in range(1, len(alphas) - 1):
        # Central difference for first and second derivatives
        drho = (log_rho[i+1] - log_rho[i-1]) / 2
        deta = (log_eta[i+1] - log_eta[i-1]) / 2
        d2rho = log_rho[i+1] - 2*log_rho[i] + log_rho[i-1]
        d2eta = log_eta[i+1] - 2*log_eta[i] + log_eta[i-1]
        
        # Curvature formula
        num = drho * d2eta - deta * d2rho
        denom = (drho**2 + deta**2) ** 1.5
        curvatures[i] = num / denom if denom > 1e-12 else 0.0
    
    best_idx = np.argmax(curvatures)
    optimal_alpha = alphas[best_idx]
    
    return optimal_alpha, residual_norms, solution_norms


def gcv_select_alpha(
    A: np.ndarray,
    b: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    order: int = 2,
    n_points: int = 20
) -> Tuple[float, np.ndarray]:
    """
    Select regularization parameter using Generalized Cross-Validation.
    
    GCV minimizes: ||Ax - b||² / (trace(I - A(A'A + αL'L)^{-1}A'))²
    
    Parameters
    ----------
    A : np.ndarray
        Response matrix
    b : np.ndarray
        Measurements
    alphas : np.ndarray, optional
        Regularization parameters to test
    order : int
        Regularization order
    n_points : int
        Number of alpha values
    
    Returns
    -------
    Tuple[float, np.ndarray]
        Optimal alpha, GCV values
    """
    if alphas is None:
        alphas = np.logspace(-6, 2, n_points)
    
    m, n = A.shape
    L = build_tikhonov_matrix(n, order)
    
    gcv_values = []
    
    for alpha in alphas:
        # Compute regularized solution
        x = tikhonov_solve(A, b, alpha, order)
        residual_norm = np.linalg.norm(A @ x - b) ** 2
        
        # Compute effective degrees of freedom
        # Using simplified approximation: trace(A(A'A + αL'L)^{-1}A')
        ATA = A.T @ A
        LTL = L.T @ L
        try:
            H = A @ np.linalg.solve(ATA + alpha * LTL, A.T)
            trace_H = np.trace(H)
            denom = (m - trace_H) ** 2
            gcv = residual_norm / denom if denom > 1e-12 else float('inf')
        except np.linalg.LinAlgError:
            gcv = float('inf')
        
        gcv_values.append(gcv)
    
    gcv_values = np.array(gcv_values)
    best_idx = np.argmin(gcv_values)
    
    return alphas[best_idx], gcv_values


def regularized_unfold(
    response_matrix: np.ndarray,
    measurements: np.ndarray,
    measurement_errors: Optional[np.ndarray] = None,
    prior_spectrum: Optional[np.ndarray] = None,
    method: str = 'gradient',
    reg_type: str = 'log_smooth',
    reg_param: Optional[float] = None,
    auto_regularization: str = 'none',
    **kwargs
) -> RegularizedSolution:
    """
    High-level interface for regularized spectrum unfolding.
    
    Parameters
    ----------
    response_matrix : np.ndarray
        Response/transfer matrix (m x n)
    measurements : np.ndarray
        Measured values (m,)
    measurement_errors : np.ndarray, optional
        Measurement uncertainties
    prior_spectrum : np.ndarray, optional
        Prior/initial spectrum estimate
    method : str
        Solver method: 'gradient', 'tikhonov'
    reg_type : str
        Regularization type: 'log_smooth', 'first_deriv', 'second_deriv'
    reg_param : float, optional
        Regularization parameter (if None, auto-selected or default)
    auto_regularization : str
        Method for automatic parameter selection: 'none', 'lcurve', 'gcv'
    **kwargs
        Additional arguments passed to solver
    
    Returns
    -------
    RegularizedSolution
        Unfolded spectrum with uncertainties
    """
    A = np.atleast_2d(response_matrix)
    b = np.atleast_1d(measurements).flatten()
    
    # Auto-select regularization parameter
    if reg_param is None:
        if auto_regularization == 'lcurve':
            reg_param, _, _ = l_curve_corner(A, b)
        elif auto_regularization == 'gcv':
            reg_param, _ = gcv_select_alpha(A, b)
        else:
            reg_param = 0.01  # Default
    
    if method == 'gradient':
        return gradient_descent_regularized(
            A, b,
            b_error=measurement_errors,
            x0=prior_spectrum,
            reg_weight=reg_param,
            reg_type=reg_type,
            **kwargs
        )
    elif method == 'tikhonov':
        order = {'log_smooth': 2, 'first_deriv': 1, 'second_deriv': 2}.get(reg_type, 2)
        x = tikhonov_solve(A, b, reg_param, order, prior_spectrum)
        
        residuals = A @ x - b
        chi_sq = float(np.sum(residuals ** 2))
        
        return RegularizedSolution(
            spectrum=x,
            uncertainties=np.sqrt(np.abs(x)),  # Simple approximation
            residuals=residuals,
            chi_squared=chi_sq,
            regularization_param=reg_param,
            n_iterations=1,
            converged=True
        )
    else:
        raise ValueError(f"Unknown method: {method}")
