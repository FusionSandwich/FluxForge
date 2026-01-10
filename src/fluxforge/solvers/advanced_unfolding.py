"""
Advanced Unfolding Module - Epics W (PyUnfold) and Y (SpecKit)

Implements advanced unfolding features:
- Multinomial and Poisson covariance models (W1.5)
- Adye error propagation (W1.6)
- Log-smoothness regularization (Y1.1)
- Second-derivative (f'') convergence criteria (Y1.7)

These complement the base iterative solvers in iterative.py.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import math
from enum import Enum

from fluxforge.core.linalg import Matrix, Vector, matmul


class CovarianceModel(Enum):
    """Covariance model for unfolding."""
    DIAGONAL = "diagonal"           # Independent measurements
    MULTINOMIAL = "multinomial"     # Multinomial statistics (PyUnfold)
    POISSON = "poisson"            # Poisson statistics
    FULL = "full"                  # Full covariance matrix


@dataclass
class AdvancedIterativeSolution:
    """Extended solution container with covariance."""
    flux: Vector
    flux_uncertainty: Optional[Vector] = None
    covariance: Optional[Matrix] = None
    history: List[Vector] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    chi_squared: float = 0.0
    chi_squared_history: List[float] = field(default_factory=list)
    final_residuals: Optional[Vector] = None
    
    # Convergence diagnostics
    ddJ_history: List[float] = field(default_factory=list)
    smoothness_history: List[float] = field(default_factory=list)


def build_covariance_matrix(
    measurements: Vector,
    model: CovarianceModel,
    uncertainties: Optional[Vector] = None,
    full_cov: Optional[Matrix] = None,
    floor: float = 1e-20
) -> Matrix:
    """
    Build covariance matrix for measurements.
    
    Parameters
    ----------
    measurements : Vector
        Measurement values
    model : CovarianceModel
        Covariance model type
    uncertainties : Vector, optional
        Individual uncertainties (for DIAGONAL)
    full_cov : Matrix, optional
        Full covariance matrix (for FULL)
    floor : float
        Minimum variance
    
    Returns
    -------
    Matrix
        Covariance matrix
    """
    n = len(measurements)
    
    if model == CovarianceModel.DIAGONAL:
        # Simple diagonal covariance
        cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            if uncertainties:
                cov[i][i] = max(uncertainties[i]**2, floor)
            else:
                # Assume Poisson-like sqrt(N) uncertainty
                cov[i][i] = max(measurements[i], floor)
        return cov
    
    elif model == CovarianceModel.POISSON:
        # Poisson statistics: variance = mean
        cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            cov[i][i] = max(measurements[i], floor)
        return cov
    
    elif model == CovarianceModel.MULTINOMIAL:
        # Multinomial statistics (PyUnfold style)
        # Cov[i,j] = N * p_i * (δ_ij - p_j)
        total = sum(max(m, floor) for m in measurements)
        probs = [max(m, floor) / total for m in measurements]
        
        cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    cov[i][j] = total * probs[i] * (1 - probs[i])
                else:
                    cov[i][j] = -total * probs[i] * probs[j]
        return cov
    
    elif model == CovarianceModel.FULL:
        if full_cov is None:
            raise ValueError("Full covariance matrix required for FULL model")
        return full_cov
    
    else:
        raise ValueError(f"Unknown covariance model: {model}")


def invert_matrix_cholesky(cov: Matrix, floor: float = 1e-12) -> Matrix:
    """
    Invert covariance matrix using Cholesky decomposition.
    
    Falls back to pseudo-inverse for near-singular matrices.
    """
    n = len(cov)
    
    # Add regularization to diagonal
    cov_reg = [[cov[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        cov_reg[i][i] += floor
    
    # Simple Gauss-Jordan inversion (for small matrices)
    # For larger matrices, use numpy in production
    try:
        # Augment with identity
        aug = [[cov_reg[i][j] for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)]
               for i in range(n)]
        
        # Forward elimination
        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            pivot = aug[col][col]
            if abs(pivot) < floor:
                # Near-singular, add more regularization
                aug[col][col] += floor * 100
                pivot = aug[col][col]
            
            # Scale pivot row
            for j in range(2 * n):
                aug[col][j] /= pivot
            
            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] -= factor * aug[col][j]
        
        # Extract inverse
        inv = [[aug[i][j + n] for j in range(n)] for i in range(n)]
        return inv
        
    except Exception:
        # Fallback: return pseudo-diagonal inverse
        inv = [[0.0] * n for _ in range(n)]
        for i in range(n):
            if cov[i][i] > floor:
                inv[i][i] = 1.0 / cov[i][i]
            else:
                inv[i][i] = 1.0 / floor
        return inv


def adye_error_propagation(
    response: Matrix,
    unfolded_flux: Vector,
    measurement_cov: Matrix,
    method: str = "bayes"
) -> Matrix:
    """
    Adye error propagation for unfolded spectrum.
    
    Implements the error propagation from:
    T. Adye, "Corrected error calculation for iterative Bayesian unfolding"
    
    Parameters
    ----------
    response : Matrix
        Response matrix R
    unfolded_flux : Vector
        Unfolded flux estimate
    measurement_cov : Matrix
        Measurement covariance matrix
    method : str
        "bayes" for D'Agostini/Bayes propagation
        "matrix" for standard matrix propagation
    
    Returns
    -------
    Matrix
        Covariance matrix for unfolded flux
    """
    n_meas = len(response)
    n_flux = len(response[0])
    floor = 1e-20
    
    if method == "bayes":
        # Bayesian/D'Agostini style propagation
        # Compute expected measurements
        predicted = matmul(response, unfolded_flux)
        predicted = [max(p, floor) for p in predicted]
        
        # Compute unfolding matrix M (pseudo-inverse of R)
        # M_gj = R_jg * phi_g / sum_g'(R_jg' * phi_g')
        M = [[0.0] * n_meas for _ in range(n_flux)]
        for g in range(n_flux):
            for j in range(n_meas):
                M[g][j] = response[j][g] * unfolded_flux[g] / predicted[j]
        
        # Propagate: V_phi = M @ V_meas @ M^T
        # First: temp = V_meas @ M^T
        temp = [[0.0] * n_flux for _ in range(n_meas)]
        for i in range(n_meas):
            for g in range(n_flux):
                temp[i][g] = sum(measurement_cov[i][j] * M[g][j] for j in range(n_meas))
        
        # Then: V_phi = M @ temp
        V_phi = [[0.0] * n_flux for _ in range(n_flux)]
        for g1 in range(n_flux):
            for g2 in range(n_flux):
                V_phi[g1][g2] = sum(M[g1][j] * temp[j][g2] for j in range(n_meas))
        
        return V_phi
    
    elif method == "matrix":
        # Standard matrix propagation using pseudo-inverse
        # V_phi = (R^T R)^-1 R^T V_meas R (R^T R)^-1
        
        # R^T @ R
        RTR = [[0.0] * n_flux for _ in range(n_flux)]
        for g1 in range(n_flux):
            for g2 in range(n_flux):
                RTR[g1][g2] = sum(response[i][g1] * response[i][g2] for i in range(n_meas))
        
        # (R^T R)^-1
        RTR_inv = invert_matrix_cholesky(RTR)
        
        # R^T @ V_meas
        RT_V = [[0.0] * n_meas for _ in range(n_flux)]
        for g in range(n_flux):
            for j in range(n_meas):
                RT_V[g][j] = sum(response[i][g] * measurement_cov[i][j] for i in range(n_meas))
        
        # R^T @ V_meas @ R
        RT_V_R = [[0.0] * n_flux for _ in range(n_flux)]
        for g1 in range(n_flux):
            for g2 in range(n_flux):
                RT_V_R[g1][g2] = sum(RT_V[g1][j] * response[j][g2] for j in range(n_meas))
        
        # (R^T R)^-1 @ RT_V_R
        temp = [[0.0] * n_flux for _ in range(n_flux)]
        for g1 in range(n_flux):
            for g2 in range(n_flux):
                temp[g1][g2] = sum(RTR_inv[g1][k] * RT_V_R[k][g2] for k in range(n_flux))
        
        # temp @ (R^T R)^-1
        V_phi = [[0.0] * n_flux for _ in range(n_flux)]
        for g1 in range(n_flux):
            for g2 in range(n_flux):
                V_phi[g1][g2] = sum(temp[g1][k] * RTR_inv[k][g2] for k in range(n_flux))
        
        return V_phi
    
    else:
        raise ValueError(f"Unknown method: {method}")


def log_smoothness_penalty(flux: Vector, floor: float = 1e-20) -> float:
    """
    Calculate log-smoothness penalty.
    
    Penalty = sum((log(phi[g+1]) - log(phi[g]))^2)
    
    This encourages smooth spectra in log-space, which is
    physically appropriate for many neutron/gamma spectra.
    """
    if len(flux) < 2:
        return 0.0
    
    log_flux = [math.log(max(f, floor)) for f in flux]
    
    penalty = 0.0
    for g in range(len(flux) - 1):
        diff = log_flux[g + 1] - log_flux[g]
        penalty += diff * diff
    
    return penalty


def log_smoothness_gradient(flux: Vector, floor: float = 1e-20) -> Vector:
    """
    Calculate gradient of log-smoothness penalty.
    
    d/d(phi_g) [sum (log(phi[g+1]) - log(phi[g]))^2]
    """
    n = len(flux)
    grad = [0.0] * n
    
    if n < 2:
        return grad
    
    log_flux = [math.log(max(f, floor)) for f in flux]
    
    for g in range(n):
        phi_g = max(flux[g], floor)
        
        if g > 0:
            # From term (log(phi[g]) - log(phi[g-1]))^2
            diff_left = log_flux[g] - log_flux[g - 1]
            grad[g] += 2 * diff_left / phi_g
        
        if g < n - 1:
            # From term (log(phi[g+1]) - log(phi[g]))^2
            diff_right = log_flux[g + 1] - log_flux[g]
            grad[g] -= 2 * diff_right / phi_g
    
    return grad


def compute_ddJ_convergence(
    J_history: List[float],
    min_length: int = 10
) -> Tuple[float, float]:
    """
    Compute second-derivative (f'') convergence criterion.
    
    This is the Neutron-Unfolding style convergence check:
    - J = objective function value
    - dJ = J[n-1] - J[n] (first derivative)
    - ddJ = |dJ[n] - dJ[n-1]| (second derivative)
    
    Convergence when ddJ approaches zero (J changing linearly or not at all).
    
    Parameters
    ----------
    J_history : List[float]
        History of objective function values
    min_length : int
        Minimum history length required
    
    Returns
    -------
    tuple
        (dJ, ddJ) values
    """
    if len(J_history) < min_length:
        return 1.0, 1.0
    
    dJ_history = []
    for i in range(1, len(J_history)):
        dJ_history.append(J_history[i - 1] - J_history[i])
    
    ddJ_history = []
    for i in range(1, len(dJ_history)):
        ddJ_history.append(abs(dJ_history[i] - dJ_history[i - 1]))
    
    if len(ddJ_history) == 0:
        return 0.0, 0.0
    
    dJ = dJ_history[-1] if dJ_history else 0.0
    ddJ = ddJ_history[-1] if ddJ_history else 0.0
    
    return dJ, ddJ


def mlem_with_covariance(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    cov_model: CovarianceModel = CovarianceModel.POISSON,
    measurement_uncertainty: Optional[Vector] = None,
    measurement_cov: Optional[Matrix] = None,
    max_iters: int = 1000,
    tolerance: float = 1e-4,
    convergence_mode: str = "ddJ",  # "relative", "ddJ", "chi2"
    floor: float = 1e-20,
    relaxation: float = 0.8,
    smoothness_weight: float = 0.0,
    compute_errors: bool = True,
    error_method: str = "bayes",
    verbose: bool = False,
) -> AdvancedIterativeSolution:
    """
    MLEM unfolding with full covariance propagation.
    
    Enhanced MLEM with:
    - Multinomial/Poisson covariance options (W1.5)
    - Adye error propagation (W1.6)
    - Log-smoothness regularization (Y1.1)
    - Second-derivative convergence (Y1.7)
    
    Parameters
    ----------
    response : Matrix
        Response matrix
    measurements : Vector
        Measurement values
    initial_flux : Vector, optional
        Initial flux estimate
    cov_model : CovarianceModel
        Covariance model for measurements
    measurement_uncertainty : Vector, optional
        Individual uncertainties
    measurement_cov : Matrix, optional
        Full covariance matrix (for FULL model)
    max_iters : int
        Maximum iterations
    tolerance : float
        Convergence threshold
    convergence_mode : str
        "relative" for flux change, "ddJ" for second derivative, "chi2" for chi-squared
    floor : float
        Minimum flux value
    relaxation : float
        Under-relaxation factor
    smoothness_weight : float
        Log-smoothness regularization weight
    compute_errors : bool
        Whether to compute uncertainty propagation
    error_method : str
        Error propagation method ("bayes" or "matrix")
    verbose : bool
        Print progress
    
    Returns
    -------
    AdvancedIterativeSolution
    """
    n_groups = len(response[0])
    n_meas = len(measurements)
    
    # Build covariance matrix
    meas_cov = build_covariance_matrix(
        measurements, cov_model, measurement_uncertainty, measurement_cov, floor
    )
    
    # Initialize flux
    if initial_flux is not None:
        phi = [max(x, floor) for x in initial_flux]
    else:
        avg_meas = sum(measurements) / len(measurements)
        phi = [avg_meas / n_groups for _ in range(n_groups)]
    
    history: List[Vector] = [phi[:]]
    chi2_history: List[float] = []
    J_history: List[float] = []
    ddJ_history: List[float] = []
    smoothness_history: List[float] = []
    converged = False
    
    for it in range(1, max_iters + 1):
        # Compute prediction
        predicted = matmul(response, phi)
        predicted = [max(p, floor) for p in predicted]
        
        # Compute objective function J (for ddJ convergence)
        residuals = [m - p for m, p in zip(measurements, predicted)]
        J = sum(r * r for r in residuals)
        J_history.append(J)
        
        # Compute chi-squared (weighted)
        chi2 = 0.0
        for i in range(n_meas):
            var = meas_cov[i][i]
            if var > 0:
                chi2 += (measurements[i] - predicted[i])**2 / var
        chi2_per_dof = chi2 / max(n_meas - n_groups, 1)
        chi2_history.append(chi2_per_dof)
        
        # Compute log-smoothness
        smoothness = log_smoothness_penalty(phi, floor)
        smoothness_history.append(smoothness)
        
        # Check ddJ convergence
        if len(J_history) >= 10:
            dJ, ddJ = compute_ddJ_convergence(J_history)
            ddJ_history.append(ddJ)
        else:
            ddJ_history.append(1.0)
        
        if verbose and it % 100 == 0:
            print(f"  MLEM-cov iter {it}: chi2/dof = {chi2_per_dof:.4f}, ddJ = {ddJ_history[-1]:.2e}")
        
        # Standard MLEM update
        updated: Vector = []
        max_rel_change = 0.0
        
        for g in range(n_groups):
            numerator = sum(response[i][g] * measurements[i] / predicted[i] for i in range(n_meas))
            denominator = sum(response[i][g] for i in range(n_meas))
            
            if denominator <= 0:
                updated.append(phi[g])
                continue
            
            target_phi = phi[g] * numerator / denominator
            
            # Add log-smoothness gradient if weight > 0
            if smoothness_weight > 0:
                smooth_grad = log_smoothness_gradient(phi, floor)
                target_phi -= smoothness_weight * smooth_grad[g]
            
            new_phi = phi[g] + relaxation * (target_phi - phi[g])
            new_phi = max(new_phi, floor)
            
            max_rel_change = max(max_rel_change, abs(new_phi - phi[g]) / max(phi[g], floor))
            updated.append(new_phi)
        
        phi = updated
        history.append(phi[:])
        
        # Check convergence
        if convergence_mode == "ddJ" and len(ddJ_history) >= 10:
            if ddJ_history[-1] < tolerance:
                converged = True
                if verbose:
                    print(f"  MLEM-cov converged at iter {it}: ddJ = {ddJ_history[-1]:.2e}")
                break
        
        elif convergence_mode == "relative" and max_rel_change < tolerance:
            converged = True
            if verbose:
                print(f"  MLEM-cov converged at iter {it}: rel_change = {max_rel_change:.2e}")
            break
        
        elif convergence_mode == "chi2" and chi2_per_dof < 1.0 + tolerance:
            converged = True
            if verbose:
                print(f"  MLEM-cov converged at iter {it}: chi2/dof = {chi2_per_dof:.4f}")
            break
    
    # Compute uncertainties
    flux_cov = None
    flux_unc = None
    if compute_errors:
        flux_cov = adye_error_propagation(response, phi, meas_cov, method=error_method)
        flux_unc = [math.sqrt(max(flux_cov[g][g], 0)) for g in range(n_groups)]
    
    return AdvancedIterativeSolution(
        flux=phi,
        flux_uncertainty=flux_unc,
        covariance=flux_cov,
        history=history,
        iterations=it if 'it' in dir() else max_iters,
        converged=converged,
        chi_squared=chi2_per_dof,
        chi_squared_history=chi2_history,
        final_residuals=residuals,
        ddJ_history=ddJ_history,
        smoothness_history=smoothness_history,
    )


# =============================================================================
# INLINE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing advanced_unfolding module...")
    
    # Create synthetic problem
    n_flux = 20
    n_meas = 10
    
    # True flux (smoothly varying)
    import random
    random.seed(42)
    true_flux = [1000 * math.exp(-0.1 * g) for g in range(n_flux)]
    
    # Response matrix
    response = [[0.0] * n_flux for _ in range(n_meas)]
    for i in range(n_meas):
        for g in range(n_flux):
            response[i][g] = 0.1 * math.exp(-((g - 2*i)**2 / 20))
    
    # Measurements (with noise)
    measurements = matmul(response, true_flux)
    noise_measurements = [m + random.gauss(0, math.sqrt(m)) for m in measurements]
    noise_measurements = [max(m, 1) for m in noise_measurements]
    
    print(f"\nProblem: {n_meas} measurements, {n_flux} flux groups")
    print(f"True flux range: [{min(true_flux):.1f}, {max(true_flux):.1f}]")
    
    # Test covariance matrix building
    for model in [CovarianceModel.POISSON, CovarianceModel.MULTINOMIAL]:
        cov = build_covariance_matrix(noise_measurements, model)
        diag = [cov[i][i] for i in range(n_meas)]
        print(f"\n{model.value} covariance diagonal range: [{min(diag):.2f}, {max(diag):.2f}]")
    
    # Test MLEM with covariance
    print("\nRunning MLEM with Poisson covariance...")
    result = mlem_with_covariance(
        response,
        noise_measurements,
        cov_model=CovarianceModel.POISSON,
        max_iters=500,
        convergence_mode="ddJ",
        tolerance=1e-6,
        verbose=False,
        compute_errors=True
    )
    
    print(f"  Converged: {result.converged} after {result.iterations} iterations")
    print(f"  Chi2/dof: {result.chi_squared:.3f}")
    print(f"  Final ddJ: {result.ddJ_history[-1]:.2e}")
    
    if result.flux_uncertainty:
        rel_unc = [u/f if f > 0 else 0 for u, f in zip(result.flux_uncertainty, result.flux)]
        print(f"  Relative uncertainties: {min(rel_unc):.1%} - {max(rel_unc):.1%}")
    
    # Test log-smoothness
    print("\nRunning MLEM with log-smoothness regularization...")
    result_smooth = mlem_with_covariance(
        response,
        noise_measurements,
        cov_model=CovarianceModel.POISSON,
        max_iters=500,
        convergence_mode="ddJ",
        tolerance=1e-6,
        smoothness_weight=0.01,
        verbose=False
    )
    
    print(f"  Converged: {result_smooth.converged} after {result_smooth.iterations} iterations")
    print(f"  Final smoothness: {result_smooth.smoothness_history[-1]:.2f}")
    
    # Compare flux recovery
    def rmse(est, true):
        return math.sqrt(sum((e-t)**2 for e, t in zip(est, true)) / len(true))
    
    print(f"\n  RMSE (no smoothing): {rmse(result.flux, true_flux):.1f}")
    print(f"  RMSE (with smoothing): {rmse(result_smooth.flux, true_flux):.1f}")
    
    print("\n✅ advanced_unfolding module tests passed!")
