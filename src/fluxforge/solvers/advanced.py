"""
Advanced Inference Methods for Spectrum Adjustment

Epic S - GMApy Parity

Implements advanced inference methods from IAEA GMA package:
- Levenberg-Marquardt optimization for nonlinear GLS
- Romberg adaptive integration for spectrum-averaged cross sections
- PPP (Peelle's Pertinent Puzzle) correction
- Unknown uncertainty estimation

References:
    - GMApy (IAEA-NDS): https://github.com/IAEA-NDS/gmapy
    - IAEA Technical Report on Neutron Cross Section Standards Evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize, integrate


# =============================================================================
# Levenberg-Marquardt Optimization
# =============================================================================


@dataclass
class LMConfig:
    """
    Configuration for Levenberg-Marquardt optimization.
    
    Attributes
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance (relative change in cost)
    lambda_init : float
        Initial damping parameter
    lambda_factor : float
        Factor for adjusting damping parameter
    min_lambda : float
        Minimum damping parameter
    max_lambda : float
        Maximum damping parameter
    """
    max_iter: int = 100
    tol: float = 1e-8
    lambda_init: float = 0.01
    lambda_factor: float = 10.0
    min_lambda: float = 1e-10
    max_lambda: float = 1e10


@dataclass
class LMResult:
    """
    Result of Levenberg-Marquardt optimization.
    
    Attributes
    ----------
    x : np.ndarray
        Optimized parameters
    covariance : np.ndarray
        Parameter covariance matrix
    residuals : np.ndarray
        Final residuals
    chi2 : float
        Final chi-squared value
    n_iter : int
        Number of iterations
    converged : bool
        Whether optimization converged
    message : str
        Status message
    """
    x: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    chi2: float
    n_iter: int
    converged: bool
    message: str


def levenberg_marquardt(
    model_func: Callable[[np.ndarray], np.ndarray],
    jacobian_func: Callable[[np.ndarray], np.ndarray],
    y_data: np.ndarray,
    y_cov: np.ndarray,
    x0: np.ndarray,
    config: Optional[LMConfig] = None
) -> LMResult:
    """
    Levenberg-Marquardt optimization for weighted least squares.
    
    Minimizes: χ² = (y - f(x))ᵀ V⁻¹ (y - f(x))
    
    Parameters
    ----------
    model_func : callable
        Model function f(x) returning predicted values
    jacobian_func : callable
        Jacobian of model function J[i,j] = ∂f_i/∂x_j
    y_data : np.ndarray
        Observed data
    y_cov : np.ndarray
        Data covariance matrix
    x0 : np.ndarray
        Initial parameter guess
    config : LMConfig, optional
        Optimization configuration
    
    Returns
    -------
    LMResult
        Optimization result with parameters and covariance
    
    Notes
    -----
    Based on gmapy.inference.lm_update implementation.
    """
    if config is None:
        config = LMConfig()
    
    x = np.asarray(x0, dtype=float).copy()
    n_params = len(x)
    n_data = len(y_data)
    
    # Compute inverse of covariance
    try:
        V_inv = np.linalg.inv(y_cov)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for singular matrices
        V_inv = np.linalg.pinv(y_cov)
    
    # Initial predictions and residuals
    y_pred = model_func(x)
    residuals = y_data - y_pred
    chi2 = float(residuals @ V_inv @ residuals)
    
    lam = config.lambda_init
    converged = False
    
    for iteration in range(config.max_iter):
        # Compute Jacobian
        J = jacobian_func(x)
        J = np.atleast_2d(J)
        
        # Normal equations with damping
        # (JᵀV⁻¹J + λI) Δx = JᵀV⁻¹r
        JTV = J.T @ V_inv
        H = JTV @ J
        grad = JTV @ residuals
        
        # Add damping
        H_damped = H + lam * np.diag(np.diag(H) + 1e-10)
        
        try:
            delta_x = np.linalg.solve(H_damped, grad)
        except np.linalg.LinAlgError:
            delta_x = np.linalg.lstsq(H_damped, grad, rcond=None)[0]
        
        # Trial step
        x_trial = x + delta_x
        y_pred_trial = model_func(x_trial)
        residuals_trial = y_data - y_pred_trial
        chi2_trial = float(residuals_trial @ V_inv @ residuals_trial)
        
        # Accept or reject step
        if chi2_trial < chi2:
            # Accept step
            x = x_trial
            y_pred = y_pred_trial
            residuals = residuals_trial
            chi2_old = chi2
            chi2 = chi2_trial
            
            # Decrease damping
            lam = max(lam / config.lambda_factor, config.min_lambda)
            
            # Check convergence
            if abs(chi2_old - chi2) / max(chi2, 1e-10) < config.tol:
                converged = True
                break
        else:
            # Reject step, increase damping
            lam = min(lam * config.lambda_factor, config.max_lambda)
            
            if lam >= config.max_lambda:
                break
    
    # Compute final covariance
    J = jacobian_func(x)
    J = np.atleast_2d(J)
    H = J.T @ V_inv @ J
    
    try:
        covariance = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(H)
    
    message = "Converged" if converged else f"Max iterations ({config.max_iter}) reached"
    
    return LMResult(
        x=x,
        covariance=covariance,
        residuals=residuals,
        chi2=chi2,
        n_iter=iteration + 1,
        converged=converged,
        message=message
    )


def gls_update_numpy(
    response: np.ndarray,
    measurements: np.ndarray,
    measurement_cov: np.ndarray,
    prior_flux: np.ndarray,
    prior_cov: np.ndarray,
    use_sparse: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    GLS spectral adjustment with numpy arrays.
    
    Implements the closed-form GLS solution:
        φ̂ = φ₀ + V_φ Rᵀ (R V_φ Rᵀ + V_y)⁻¹ (y - R φ₀)
        V_φ̂ = V_φ - V_φ Rᵀ (R V_φ Rᵀ + V_y)⁻¹ R V_φ
    
    Parameters
    ----------
    response : np.ndarray
        Response matrix R (n_reactions × n_groups)
    measurements : np.ndarray
        Measurement vector y
    measurement_cov : np.ndarray
        Measurement covariance V_y
    prior_flux : np.ndarray
        Prior flux estimate φ₀
    prior_cov : np.ndarray
        Prior covariance V_φ
    use_sparse : bool
        If True, attempt sparse matrix operations
    
    Returns
    -------
    tuple
        (posterior_flux, posterior_cov, chi_squared)
    
    Notes
    -----
    Based on gmapy.inference.gls_update implementation.
    """
    R = np.atleast_2d(response)
    y = np.atleast_1d(measurements)
    V_y = np.atleast_2d(measurement_cov)
    phi0 = np.atleast_1d(prior_flux)
    V_phi = np.atleast_2d(prior_cov)
    
    # Predicted measurements
    y_pred = R @ phi0
    
    # Innovation (residuals)
    innovation = y - y_pred
    
    # Innovation covariance
    V_inn = R @ V_phi @ R.T + V_y
    
    # Solve for gain matrix: K = V_phi @ R.T @ inv(V_inn)
    if use_sparse:
        try:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import spsolve
            
            V_inn_sparse = csc_matrix(V_inn)
            RTV_phi = (R @ V_phi).T
            
            # Solve V_inn @ K.T = R @ V_phi
            K_T = np.zeros_like(RTV_phi)
            for j in range(K_T.shape[1]):
                K_T[:, j] = spsolve(V_inn_sparse, RTV_phi[:, j])
            K = K_T.T
        except ImportError:
            # Fall back to dense
            V_inn_inv = np.linalg.inv(V_inn)
            K = V_phi @ R.T @ V_inn_inv
    else:
        try:
            # Use Cholesky for numerical stability
            L = np.linalg.cholesky(V_inn)
            V_inn_inv = np.linalg.inv(L).T @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            V_inn_inv = np.linalg.pinv(V_inn)
        K = V_phi @ R.T @ V_inn_inv
    
    # Posterior flux
    phi_post = phi0 + K @ innovation
    
    # Posterior covariance
    V_phi_post = V_phi - K @ R @ V_phi
    
    # Chi-squared
    chi2 = float(innovation @ np.linalg.solve(V_inn, innovation))
    
    return phi_post, V_phi_post, chi2


# =============================================================================
# Romberg Adaptive Integration
# =============================================================================


def romberg_integrate(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    rtol: float = 1e-8,
    max_divisions: int = 16
) -> Tuple[float, float]:
    """
    Romberg adaptive integration.
    
    Uses Richardson extrapolation on the trapezoidal rule for
    high-accuracy integration.
    
    Parameters
    ----------
    func : callable
        Function to integrate f(x)
    a : float
        Lower integration bound
    b : float
        Upper integration bound
    tol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    max_divisions : int
        Maximum number of subdivisions
    
    Returns
    -------
    tuple
        (integral_value, error_estimate)
    
    Notes
    -----
    Uses scipy.integrate.quad for robust adaptive integration.
    """
    # Use quad for robust integration
    result, error = integrate.quad(func, a, b, epsabs=tol, epsrel=rtol)
    
    return float(result), float(error)


def spectrum_averaged_cross_section(
    sigma: Callable[[float], float],
    flux: Callable[[float], float],
    E_low: float,
    E_high: float,
    method: str = 'romberg',
    n_points: int = 100
) -> Tuple[float, float]:
    """
    Calculate spectrum-averaged cross section.
    
    <σ> = ∫ σ(E) φ(E) dE / ∫ φ(E) dE
    
    Parameters
    ----------
    sigma : callable
        Cross section function σ(E) [barns]
    flux : callable
        Flux function φ(E) [n/cm²/s/MeV]
    E_low : float
        Lower energy bound [MeV]
    E_high : float
        Upper energy bound [MeV]
    method : str
        Integration method: 'romberg', 'quad', 'trapezoid'
    n_points : int
        Number of points for trapezoid method
    
    Returns
    -------
    tuple
        (sigma_avg, uncertainty_estimate)
    
    Notes
    -----
    Based on gmapy spectrum averaging with Romberg integration.
    """
    if method == 'romberg':
        # Numerator: ∫ σ(E) φ(E) dE
        def integrand(E):
            return sigma(E) * flux(E)
        
        numerator, num_err = romberg_integrate(integrand, E_low, E_high)
        denominator, den_err = romberg_integrate(flux, E_low, E_high)
        
        if denominator <= 0:
            return 0.0, np.inf
        
        sigma_avg = numerator / denominator
        
        # Error propagation
        rel_err = np.sqrt((num_err/max(numerator, 1e-30))**2 + 
                         (den_err/max(denominator, 1e-30))**2)
        uncertainty = sigma_avg * rel_err
        
    elif method == 'quad':
        def integrand(E):
            return sigma(E) * flux(E)
        
        numerator, num_err = integrate.quad(integrand, E_low, E_high)
        denominator, den_err = integrate.quad(flux, E_low, E_high)
        
        if denominator <= 0:
            return 0.0, np.inf
        
        sigma_avg = numerator / denominator
        rel_err = np.sqrt((num_err/max(numerator, 1e-30))**2 + 
                         (den_err/max(denominator, 1e-30))**2)
        uncertainty = sigma_avg * rel_err
        
    else:  # trapezoid
        E_grid = np.linspace(E_low, E_high, n_points)
        sigma_vals = np.array([sigma(E) for E in E_grid])
        flux_vals = np.array([flux(E) for E in E_grid])
        
        numerator = np.trapz(sigma_vals * flux_vals, E_grid)
        denominator = np.trapz(flux_vals, E_grid)
        
        if denominator <= 0:
            return 0.0, np.inf
        
        sigma_avg = numerator / denominator
        uncertainty = 0.0  # No error estimate for trapezoid
    
    return float(sigma_avg), float(uncertainty)


def group_averaged_cross_section(
    sigma: Callable[[float], float],
    flux_per_group: np.ndarray,
    energy_bounds: np.ndarray,
    method: str = 'romberg'
) -> np.ndarray:
    """
    Calculate group-averaged cross sections.
    
    σ_g = ∫_{E_g}^{E_{g+1}} σ(E) φ(E) dE / ∫_{E_g}^{E_{g+1}} φ(E) dE
    
    Parameters
    ----------
    sigma : callable
        Cross section function σ(E) [barns]
    flux_per_group : np.ndarray
        Flux in each energy group [n/cm²/s]
    energy_bounds : np.ndarray
        Energy group boundaries [MeV], length n_groups+1
    method : str
        Integration method
    
    Returns
    -------
    np.ndarray
        Group-averaged cross sections [barns]
    """
    n_groups = len(flux_per_group)
    sigma_g = np.zeros(n_groups)
    
    for g in range(n_groups):
        E_low = energy_bounds[g]
        E_high = energy_bounds[g + 1]
        dE = E_high - E_low
        
        # Approximate flux within group
        phi_g = flux_per_group[g] / dE  # Convert to spectral density
        
        if method == 'romberg':
            # For flat flux assumption within group
            numerator, _ = romberg_integrate(sigma, E_low, E_high)
            sigma_g[g] = numerator / dE
        else:
            sigma_g[g], _ = spectrum_averaged_cross_section(
                sigma, lambda E: phi_g, E_low, E_high, method=method
            )
    
    return sigma_g


# =============================================================================
# PPP (Peelle's Pertinent Puzzle) Correction
# =============================================================================


class PPPCorrectionMethod(Enum):
    """PPP correction methods."""
    NONE = "none"
    CHIBA_SMITH = "chiba_smith"  # Chiba-Smith logarithmic method
    RATIO = "ratio"  # Convert to ratio data


def apply_ppp_correction(
    measurements: np.ndarray,
    measurement_cov: np.ndarray,
    method: PPPCorrectionMethod = PPPCorrectionMethod.CHIBA_SMITH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PPP (Peelle's Pertinent Puzzle) correction.
    
    PPP arises when data with correlated normalization uncertainties
    are fitted, leading to biased estimates. The correction transforms
    the problem to avoid this bias.
    
    Parameters
    ----------
    measurements : np.ndarray
        Measured values
    measurement_cov : np.ndarray
        Measurement covariance matrix
    method : PPPCorrectionMethod
        Correction method to use
    
    Returns
    -------
    tuple
        (corrected_measurements, corrected_covariance)
    
    Notes
    -----
    Based on gmapy.legacy.ppp_correction.
    
    The Chiba-Smith method transforms to log-space where the bias
    vanishes: y' = log(y), V' = V / (y ⊗ y)
    """
    y = np.atleast_1d(measurements).astype(float)
    V = np.atleast_2d(measurement_cov).astype(float)
    
    if method == PPPCorrectionMethod.NONE:
        return y, V
    
    elif method == PPPCorrectionMethod.CHIBA_SMITH:
        # Log transformation
        # y' = log(y)
        # V'[i,j] = V[i,j] / (y[i] * y[j])
        
        y_corrected = np.log(np.maximum(y, 1e-30))
        
        V_corrected = np.zeros_like(V)
        for i in range(len(y)):
            for j in range(len(y)):
                if y[i] > 0 and y[j] > 0:
                    V_corrected[i, j] = V[i, j] / (y[i] * y[j])
        
        return y_corrected, V_corrected
    
    elif method == PPPCorrectionMethod.RATIO:
        # Convert to ratio of measurements to some reference
        # This is a simplified approach
        y_mean = np.mean(y)
        y_corrected = y / y_mean
        
        # Scale covariance
        V_corrected = V / (y_mean ** 2)
        
        return y_corrected, V_corrected
    
    else:
        return y, V


# =============================================================================
# Unknown Uncertainty Estimation
# =============================================================================


def estimate_unknown_uncertainty(
    residuals: np.ndarray,
    covariance: np.ndarray,
    method: str = 'ml'
) -> float:
    """
    Estimate unknown systematic uncertainty from residuals.
    
    When the chi-squared is larger than expected, this suggests
    unaccounted systematic uncertainty. This function estimates
    the magnitude of that unknown uncertainty.
    
    Parameters
    ----------
    residuals : np.ndarray
        Fit residuals (y - y_model)
    covariance : np.ndarray
        Data covariance matrix
    method : str
        Estimation method: 'ml' (maximum likelihood) or 'birge'
    
    Returns
    -------
    float
        Estimated unknown uncertainty to add to diagonal
    
    Notes
    -----
    Based on gmapy.inference.estimate_unknown_uncertainty.
    
    The ML estimate solves:
        χ²(V + σ²I) = n
    where n is the number of data points.
    """
    n = len(residuals)
    r = np.atleast_1d(residuals)
    V = np.atleast_2d(covariance)
    
    # Current chi-squared
    try:
        chi2 = float(r @ np.linalg.solve(V, r))
    except np.linalg.LinAlgError:
        chi2 = float(r @ np.linalg.lstsq(V, r, rcond=None)[0])
    
    if chi2 <= n:
        # No additional uncertainty needed
        return 0.0
    
    if method == 'ml':
        # Binary search for σ² such that χ²(V + σ²I) = n
        sigma2_low = 0.0
        sigma2_high = np.max(np.abs(r)) ** 2
        
        for _ in range(50):  # Maximum iterations
            sigma2 = (sigma2_low + sigma2_high) / 2
            V_aug = V + sigma2 * np.eye(n)
            
            try:
                chi2_aug = float(r @ np.linalg.solve(V_aug, r))
            except np.linalg.LinAlgError:
                chi2_aug = float(r @ np.linalg.lstsq(V_aug, r, rcond=None)[0])
            
            if chi2_aug > n:
                sigma2_low = sigma2
            else:
                sigma2_high = sigma2
            
            if abs(chi2_aug - n) < 0.01:
                break
        
        return np.sqrt(sigma2)
    
    elif method == 'birge':
        # Birge ratio method: scale uncertainties by sqrt(χ²/n)
        birge_ratio = np.sqrt(chi2 / n)
        # Equivalent additional uncertainty
        avg_var = np.mean(np.diag(V))
        return np.sqrt(avg_var) * (birge_ratio - 1)
    
    else:
        return 0.0
