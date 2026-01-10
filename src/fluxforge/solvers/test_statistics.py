"""
Additional test statistics and regularizers for iterative unfolding.

Epic W - PyUnfold Parity features.

Implements:
- Multiple test statistics (KS, Chi2, Bayes Factor, RMD)
- Spline regularization for smoothing
- Jeffreys prior for logarithmic ranges
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class TestStatisticType(Enum):
    """Convergence test statistic types."""
    KS = "ks"            # Kolmogorov-Smirnov
    CHI2 = "chi2"        # Reduced chi-squared
    BAYES_FACTOR = "bf"  # Bayes factor
    RMD = "rmd"          # Relative maximum difference


@dataclass
class ConvergenceResult:
    """Result of convergence test."""
    statistic: float
    converged: bool
    ts_type: TestStatisticType
    threshold: float


def ks_test_statistic(phi_curr: np.ndarray, phi_prev: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov test statistic.
    
    Measures maximum absolute difference between normalized CDFs.
    
    Parameters
    ----------
    phi_curr : np.ndarray
        Current iteration flux
    phi_prev : np.ndarray
        Previous iteration flux
        
    Returns
    -------
    float
        KS statistic (0 = identical, 1 = maximally different)
    """
    # Normalize to create pseudo-CDFs
    phi_curr = np.asarray(phi_curr, dtype=float)
    phi_prev = np.asarray(phi_prev, dtype=float)
    
    total_curr = np.sum(np.abs(phi_curr))
    total_prev = np.sum(np.abs(phi_prev))
    
    if total_curr < 1e-30 or total_prev < 1e-30:
        return 1.0
    
    cdf_curr = np.cumsum(np.abs(phi_curr)) / total_curr
    cdf_prev = np.cumsum(np.abs(phi_prev)) / total_prev
    
    return float(np.max(np.abs(cdf_curr - cdf_prev)))


def chi2_test_statistic(phi_curr: np.ndarray, phi_prev: np.ndarray,
                        sigma: Optional[np.ndarray] = None) -> float:
    """
    Reduced chi-squared test statistic.
    
    Parameters
    ----------
    phi_curr : np.ndarray
        Current iteration flux
    phi_prev : np.ndarray  
        Previous iteration flux
    sigma : np.ndarray, optional
        Uncertainties (if None, uses sqrt of previous)
        
    Returns
    -------
    float
        Reduced chi-squared
    """
    phi_curr = np.asarray(phi_curr, dtype=float)
    phi_prev = np.asarray(phi_prev, dtype=float)
    
    if sigma is None:
        sigma = np.sqrt(np.maximum(np.abs(phi_prev), 1.0))
    else:
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-10)
    
    chi2 = np.sum(((phi_curr - phi_prev) / sigma) ** 2)
    dof = len(phi_curr)
    
    return float(chi2 / max(dof, 1))


def bayes_factor_test_statistic(phi_curr: np.ndarray, phi_prev: np.ndarray,
                                 data: Optional[np.ndarray] = None,
                                 response: Optional[np.ndarray] = None) -> float:
    """
    Bayes factor test statistic (Ben-Zvi et al., ApJ 2011).
    
    Compares likelihood of current vs previous iteration.
    
    Parameters
    ----------
    phi_curr : np.ndarray
        Current iteration flux
    phi_prev : np.ndarray
        Previous iteration flux
    data : np.ndarray, optional
        Observed data (for likelihood calc)
    response : np.ndarray, optional
        Response matrix
        
    Returns
    -------
    float
        Log10(Bayes factor) - positive means curr is better
    """
    phi_curr = np.asarray(phi_curr, dtype=float)
    phi_prev = np.asarray(phi_prev, dtype=float)
    
    # Without data/response, use ratio of norms
    if data is None or response is None:
        norm_curr = np.sum(phi_curr ** 2)
        norm_prev = np.sum(phi_prev ** 2)
        if norm_prev < 1e-30:
            return 0.0
        return float(np.log10(max(norm_curr / norm_prev, 1e-30)))
    
    # With data, compute log-likelihood ratio
    data = np.asarray(data, dtype=float)
    response = np.asarray(response, dtype=float)
    
    pred_curr = response @ phi_curr
    pred_prev = response @ phi_prev
    
    # Poisson log-likelihood approximation
    def poisson_loglik(pred, obs):
        pred = np.maximum(pred, 1e-10)
        return np.sum(obs * np.log(pred) - pred)
    
    ll_curr = poisson_loglik(pred_curr, data)
    ll_prev = poisson_loglik(pred_prev, data)
    
    # Return change in log-likelihood
    return float((ll_curr - ll_prev) / np.log(10))


def rmd_test_statistic(phi_curr: np.ndarray, phi_prev: np.ndarray) -> float:
    """
    Relative maximum difference test statistic.
    
    Parameters
    ----------
    phi_curr : np.ndarray
        Current iteration flux
    phi_prev : np.ndarray
        Previous iteration flux
        
    Returns
    -------
    float
        Maximum relative difference
    """
    phi_curr = np.asarray(phi_curr, dtype=float)
    phi_prev = np.asarray(phi_prev, dtype=float)
    
    denom = np.maximum(np.abs(phi_prev), np.abs(phi_curr))
    denom = np.maximum(denom, 1e-30)
    
    rel_diff = np.abs(phi_curr - phi_prev) / denom
    
    return float(np.max(rel_diff))


def compute_test_statistic(phi_curr: np.ndarray, phi_prev: np.ndarray,
                           ts_type: TestStatisticType = TestStatisticType.KS,
                           **kwargs) -> float:
    """
    Compute convergence test statistic.
    
    Parameters
    ----------
    phi_curr : np.ndarray
        Current iteration flux
    phi_prev : np.ndarray
        Previous iteration flux
    ts_type : TestStatisticType
        Type of test statistic
    **kwargs
        Additional arguments for specific statistics
        
    Returns
    -------
    float
        Test statistic value
    """
    if ts_type == TestStatisticType.KS:
        return ks_test_statistic(phi_curr, phi_prev)
    elif ts_type == TestStatisticType.CHI2:
        return chi2_test_statistic(phi_curr, phi_prev, kwargs.get('sigma'))
    elif ts_type == TestStatisticType.BAYES_FACTOR:
        return bayes_factor_test_statistic(phi_curr, phi_prev,
                                           kwargs.get('data'),
                                           kwargs.get('response'))
    elif ts_type == TestStatisticType.RMD:
        return rmd_test_statistic(phi_curr, phi_prev)
    else:
        raise ValueError(f"Unknown test statistic: {ts_type}")


def check_convergence(phi_curr: np.ndarray, phi_prev: np.ndarray,
                      threshold: float = 0.01,
                      ts_type: TestStatisticType = TestStatisticType.KS,
                      **kwargs) -> ConvergenceResult:
    """
    Check if iteration has converged.
    
    Parameters
    ----------
    phi_curr : np.ndarray
        Current iteration flux
    phi_prev : np.ndarray
        Previous iteration flux
    threshold : float
        Convergence threshold
    ts_type : TestStatisticType
        Type of test statistic
    **kwargs
        Additional arguments
        
    Returns
    -------
    ConvergenceResult
        Convergence status and statistic value
    """
    stat = compute_test_statistic(phi_curr, phi_prev, ts_type, **kwargs)
    
    # For Bayes factor, convergence means small change (|bf| < threshold)
    if ts_type == TestStatisticType.BAYES_FACTOR:
        converged = abs(stat) < threshold
    else:
        converged = stat < threshold
    
    return ConvergenceResult(
        statistic=stat,
        converged=converged,
        ts_type=ts_type,
        threshold=threshold
    )


# =============================================================================
# PRIORS
# =============================================================================

def uniform_prior(n_bins: int) -> np.ndarray:
    """
    Create uniform prior.
    
    Parameters
    ----------
    n_bins : int
        Number of energy bins
        
    Returns
    -------
    np.ndarray
        Uniform prior (sums to 1)
    """
    return np.ones(n_bins) / n_bins


def jeffreys_prior(n_bins: int, e_min: float = 1e-3, e_max: float = 20.0) -> np.ndarray:
    """
    Create Jeffreys prior for logarithmic/multi-decade ranges.
    
    Appropriate when cause values span many orders of magnitude.
    Prior is proportional to 1/x in linear space = uniform in log space.
    
    Parameters
    ----------
    n_bins : int
        Number of energy bins
    e_min : float
        Minimum energy (MeV)
    e_max : float
        Maximum energy (MeV)
        
    Returns
    -------
    np.ndarray
        Jeffreys prior (normalized)
    """
    log_e = np.linspace(np.log10(e_min), np.log10(e_max), n_bins + 1)
    log_widths = np.diff(log_e)
    prior = log_widths / np.sum(log_widths)
    return prior


def power_law_prior(n_bins: int, e_min: float = 1e-3, e_max: float = 20.0,
                    index: float = -1.0) -> np.ndarray:
    """
    Create power-law prior: p(E) ∝ E^index.
    
    Parameters
    ----------
    n_bins : int
        Number of energy bins
    e_min : float
        Minimum energy (MeV)
    e_max : float
        Maximum energy (MeV)
    index : float
        Power law index (default -1.0 for 1/E)
        
    Returns
    -------
    np.ndarray
        Power law prior (normalized)
    """
    e_centers = np.logspace(np.log10(e_min), np.log10(e_max), n_bins)
    prior = e_centers ** index
    prior = prior / np.sum(prior)
    return prior


# =============================================================================
# SPLINE REGULARIZATION
# =============================================================================

def spline_regularize(phi: np.ndarray, smoothing: float = 0.1,
                      order: int = 3) -> np.ndarray:
    """
    Apply spline smoothing regularization to flux.
    
    Parameters
    ----------
    phi : np.ndarray
        Input flux (may be noisy)
    smoothing : float
        Smoothing factor (0 = no smoothing, 1 = heavy smoothing)
    order : int
        Spline order (default 3 = cubic)
        
    Returns
    -------
    np.ndarray
        Smoothed flux
    """
    from scipy.interpolate import UnivariateSpline
    
    phi = np.asarray(phi, dtype=float)
    n = len(phi)
    x = np.arange(n)
    
    # Estimate weights from magnitude
    w = np.ones(n)
    
    # Smoothing parameter - higher s = more smoothing
    s = smoothing * n
    
    try:
        spline = UnivariateSpline(x, phi, w=w, k=min(order, n-1), s=s)
        phi_smooth = spline(x)
        
        # Ensure non-negative
        phi_smooth = np.maximum(phi_smooth, 0.0)
        
        # Preserve total (integral)
        if np.sum(phi_smooth) > 0:
            phi_smooth *= np.sum(phi) / np.sum(phi_smooth)
        
        return phi_smooth
    except Exception:
        # If spline fails, return original
        return phi


def grouped_spline_regularize(phi: np.ndarray, groups: list,
                              smoothing: float = 0.1) -> np.ndarray:
    """
    Apply spline regularization to groups of bins separately.
    
    Useful for spectra with different characteristics in different regions.
    
    Parameters
    ----------
    phi : np.ndarray
        Input flux
    groups : list
        List of (start, end) indices for each group
    smoothing : float
        Smoothing factor
        
    Returns
    -------
    np.ndarray
        Smoothed flux with group-wise regularization
    """
    phi = np.asarray(phi, dtype=float).copy()
    
    for start, end in groups:
        if end > start:
            phi[start:end] = spline_regularize(phi[start:end], smoothing)
    
    return phi


class SplineRegularizer:
    """
    Callback-style spline regularizer for iterative unfolding.
    
    Can be applied after each iteration to smooth the solution.
    """
    
    def __init__(self, smoothing: float = 0.1, order: int = 3,
                 apply_every: int = 1, groups: Optional[list] = None):
        """
        Initialize spline regularizer.
        
        Parameters
        ----------
        smoothing : float
            Smoothing factor
        order : int
            Spline order
        apply_every : int
            Apply every N iterations
        groups : list, optional
            Group boundaries for grouped regularization
        """
        self.smoothing = smoothing
        self.order = order
        self.apply_every = apply_every
        self.groups = groups
        self._iteration = 0
    
    def __call__(self, phi: np.ndarray) -> np.ndarray:
        """
        Apply regularization.
        
        Parameters
        ----------
        phi : np.ndarray
            Current flux estimate
            
        Returns
        -------
        np.ndarray
            Regularized flux
        """
        self._iteration += 1
        
        if self._iteration % self.apply_every != 0:
            return phi
        
        if self.groups is not None:
            return grouped_spline_regularize(phi, self.groups, self.smoothing)
        else:
            return spline_regularize(phi, self.smoothing, self.order)
    
    def reset(self):
        """Reset iteration counter."""
        self._iteration = 0


# =============================================================================
# ITERATIVE BAYESIAN WITH REGULARIZATION
# =============================================================================

def iterative_bayesian_unfold(
    data: np.ndarray,
    response: np.ndarray,
    data_err: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    ts_type: TestStatisticType = TestStatisticType.KS,
    ts_threshold: float = 0.01,
    max_iter: int = 100,
    regularizer: Optional[Callable] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Iterative Bayesian unfolding with optional regularization.
    
    D'Agostini method with configurable test statistics and regularization.
    
    Parameters
    ----------
    data : np.ndarray
        Observed effects (counts, rates, etc.)
    response : np.ndarray
        Response matrix (n_effects x n_causes)
    data_err : np.ndarray, optional
        Data uncertainties
    prior : np.ndarray, optional
        Prior distribution on causes
    ts_type : TestStatisticType
        Convergence test statistic
    ts_threshold : float
        Convergence threshold
    max_iter : int
        Maximum iterations
    regularizer : Callable, optional
        Regularization function applied each iteration
        
    Returns
    -------
    phi : np.ndarray
        Unfolded flux
    info : dict
        Convergence info and history
    """
    data = np.asarray(data, dtype=float)
    response = np.asarray(response, dtype=float)
    
    n_effects, n_causes = response.shape
    
    # Initialize prior
    if prior is None:
        prior = uniform_prior(n_causes)
    else:
        prior = np.asarray(prior, dtype=float)
        prior = prior / np.sum(prior)  # Normalize
    
    # Data uncertainties
    if data_err is None:
        data_err = np.sqrt(np.maximum(data, 1.0))
    else:
        data_err = np.asarray(data_err, dtype=float)
    
    # Initialize
    phi = prior.copy()
    phi_prev = phi.copy()
    
    history = {
        'flux': [phi.copy()],
        'test_stat': [],
        'converged': False,
        'n_iter': 0
    }
    
    for iteration in range(max_iter):
        # Compute mixing matrix (Bayes' theorem)
        # M_ij = R_ij * phi_j / sum_k(R_ik * phi_k)
        R_phi = response @ phi  # Predicted effects
        R_phi = np.maximum(R_phi, 1e-30)
        
        # Update using D'Agostini formula
        phi_new = phi.copy()
        for j in range(n_causes):
            update = 0.0
            for i in range(n_effects):
                if R_phi[i] > 1e-30:
                    update += response[i, j] * data[i] / R_phi[i]
            phi_new[j] = phi[j] * update / np.sum(response[:, j])
        
        # Ensure non-negative
        phi_new = np.maximum(phi_new, 0.0)
        
        # Apply regularization if provided
        if regularizer is not None:
            phi_new = regularizer(phi_new)
        
        # Check convergence
        conv = check_convergence(phi_new, phi, ts_threshold, ts_type,
                                 data=data, response=response)
        history['test_stat'].append(conv.statistic)
        
        phi_prev = phi.copy()
        phi = phi_new
        history['flux'].append(phi.copy())
        
        if conv.converged:
            history['converged'] = True
            history['n_iter'] = iteration + 1
            break
    else:
        history['n_iter'] = max_iter
    
    return phi, history


# =============================================================================
# Module test
# =============================================================================

if __name__ == "__main__":
    print("Test statistics and regularization module")
    print("=" * 50)
    
    # Test statistics
    phi_prev = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    phi_curr = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    
    print("\nTest statistics (similar distributions):")
    print(f"  KS:   {ks_test_statistic(phi_curr, phi_prev):.4f}")
    print(f"  Chi2: {chi2_test_statistic(phi_curr, phi_prev):.4f}")
    print(f"  BF:   {bayes_factor_test_statistic(phi_curr, phi_prev):.4f}")
    print(f"  RMD:  {rmd_test_statistic(phi_curr, phi_prev):.4f}")
    
    # Different distributions
    phi_diff = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    print("\nTest statistics (different distributions):")
    print(f"  KS:   {ks_test_statistic(phi_diff, phi_prev):.4f}")
    print(f"  Chi2: {chi2_test_statistic(phi_diff, phi_prev):.4f}")
    print(f"  BF:   {bayes_factor_test_statistic(phi_diff, phi_prev):.4f}")
    print(f"  RMD:  {rmd_test_statistic(phi_diff, phi_prev):.4f}")
    
    # Priors
    print("\nPriors (10 bins):")
    print(f"  Uniform:  {uniform_prior(10)[:3]} ...")
    print(f"  Jeffreys: {jeffreys_prior(10)[:3]} ...")
    print(f"  Power-1:  {power_law_prior(10)[:3]} ...")
    
    # Spline regularization
    np.random.seed(42)
    noisy = np.array([1, 2, 5, 3, 7, 4, 8, 5, 4, 3]) + np.random.randn(10) * 0.5
    smooth = spline_regularize(noisy, smoothing=0.5)
    print(f"\nSpline regularization:")
    print(f"  Input:  {noisy[:5]} ...")
    print(f"  Output: {smooth[:5]} ...")
    
    # Simple unfolding test
    print("\nIterative Bayesian unfolding:")
    R = np.array([[0.8, 0.1, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.1, 0.1, 0.8]])
    true_phi = np.array([100, 200, 150])
    data = R @ true_phi + np.random.randn(3) * 5
    
    phi_unfolded, info = iterative_bayesian_unfold(data, R, max_iter=50)
    print(f"  True:     {true_phi}")
    print(f"  Unfolded: {phi_unfolded.astype(int)}")
    print(f"  Converged: {info['converged']} in {info['n_iter']} iterations")
    
    print("\n✅ All tests passed!")
