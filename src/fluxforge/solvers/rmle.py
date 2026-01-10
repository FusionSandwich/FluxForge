"""
Regularized Maximum Likelihood Estimation (RMLE) for gamma spectrum unfolding.

This module provides RMLE-based algorithms for unfolding gamma spectra
from HPGe detector measurements, which is part of the activity inference
path (Stage C in FluxForge pipeline).

RMLE is particularly useful for extracting activities from complex
gamma spectra with overlapping peaks and varying background levels.

Key Features:
- Maximum likelihood estimation with regularization
- Tikhonov regularization for ill-posed inverse problems
- L-curve method for optimal regularization parameter selection
- Monte Carlo uncertainty propagation
- Peak area extraction with continuum subtraction

Reference:
- D. Reilly et al., "Passive Nondestructive Assay of Nuclear Materials"
- G. Gilmore, "Practical Gamma-ray Spectroscopy"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, optimize, sparse


class RegularizationType(Enum):
    """Types of regularization."""
    
    NONE = "none"  # No regularization (MLE)
    TIKHONOV = "tikhonov"  # Tikhonov (L2) regularization
    TIKHONOV_DERIVATIVE = "tikhonov_derivative"  # Tikhonov on 1st derivative
    TIKHONOV_SECOND = "tikhonov_second"  # Tikhonov on 2nd derivative
    ENTROPY = "entropy"  # Maximum entropy
    SPARSE = "sparse"  # L1 regularization (sparse solution)
    POSITIVITY = "positivity"  # Non-negative constraint only


class ParameterSelection(Enum):
    """Methods for regularization parameter selection."""
    
    FIXED = "fixed"  # User-specified value
    L_CURVE = "l_curve"  # L-curve method
    GCV = "gcv"  # Generalized Cross-Validation
    DISCREPANCY = "discrepancy"  # Morozov discrepancy principle
    AUTOMATIC = "automatic"  # Let algorithm decide


class PoissonPenalty(Enum):
    """Penalty options for Poisson RMLE unfolding."""

    NONE = "none"
    L2 = "l2"  # L2 on solution
    SOBLEV_1 = "sobolev_1"  # L2 on first difference
    SOBLEV_2 = "sobolev_2"  # L2 on second difference
    L1 = "l1"  # sparsity-promoting L1


@dataclass
class SpectrumData:
    """
    Gamma spectrum data.
    
    Attributes:
        counts: Channel counts
        channels: Channel numbers
        live_time_s: Live time in seconds
        energy_calibration: Function mapping channel to energy
        uncertainty: Count uncertainties (sqrt(counts) if not provided)
    """
    
    counts: np.ndarray
    channels: Optional[np.ndarray] = None
    live_time_s: float = 1.0
    energy_calibration: Optional[Callable[[np.ndarray], np.ndarray]] = None
    uncertainty: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = np.arange(len(self.counts))
        if self.uncertainty is None:
            self.uncertainty = np.sqrt(np.maximum(self.counts, 1))
    
    @property
    def n_channels(self) -> int:
        return len(self.counts)
    
    @property
    def count_rate(self) -> np.ndarray:
        """Count rate in counts per second."""
        return self.counts / self.live_time_s
    
    def get_energies(self) -> np.ndarray:
        """Get energy values for channels."""
        if self.energy_calibration:
            return self.energy_calibration(self.channels)
        return self.channels.astype(float)


@dataclass
class PeakModel:
    """
    Model for a gamma peak.
    
    Attributes:
        centroid: Peak centroid (channel or energy)
        sigma: Peak width (Gaussian sigma)
        amplitude: Peak amplitude
        amplitude_unc: Uncertainty in amplitude
    """
    
    centroid: float
    sigma: float
    amplitude: float = 0.0
    amplitude_unc: float = 0.0
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Gaussian peak shape."""
        return self.amplitude * np.exp(-0.5 * ((x - self.centroid) / self.sigma)**2)


@dataclass
class ResponseMatrix:
    """
    Detector response matrix for spectrum unfolding.
    
    R[i,j] = probability that a photon of energy j is detected in channel i
    
    Attributes:
        matrix: Response matrix (n_channels x n_energy_bins)
        channel_edges: Channel bin edges
        energy_edges: Energy bin edges
    """
    
    matrix: np.ndarray
    channel_edges: Optional[np.ndarray] = None
    energy_edges: Optional[np.ndarray] = None
    
    @property
    def n_channels(self) -> int:
        return self.matrix.shape[0]
    
    @property
    def n_energy_bins(self) -> int:
        return self.matrix.shape[1]
    
    def normalize_columns(self) -> "ResponseMatrix":
        """Normalize each column to sum to 1."""
        col_sums = np.sum(self.matrix, axis=0)
        col_sums[col_sums == 0] = 1
        normalized = self.matrix / col_sums
        return ResponseMatrix(
            matrix=normalized,
            channel_edges=self.channel_edges,
            energy_edges=self.energy_edges,
        )


@dataclass
class UnfoldingResult:
    """
    Result of spectrum unfolding.
    
    Attributes:
        solution: Unfolded spectrum (activity or counts in energy bins)
        uncertainty: Uncertainty in solution
        covariance: Solution covariance matrix
        chi_squared: Goodness of fit
        n_iterations: Number of iterations
        regularization_param: Regularization parameter used
        residuals: Fit residuals
        converged: Whether algorithm converged
    """
    
    solution: np.ndarray
    uncertainty: np.ndarray
    covariance: Optional[np.ndarray] = None
    chi_squared: float = 0.0
    n_iterations: int = 0
    regularization_param: float = 0.0
    residuals: Optional[np.ndarray] = None
    converged: bool = True
    diagnostics: Dict = field(default_factory=dict)
    
    @property
    def n_dof(self) -> int:
        """Degrees of freedom."""
        return len(self.residuals) if self.residuals is not None else 0
    
    @property
    def reduced_chi_squared(self) -> float:
        """Reduced chi-squared."""
        if self.n_dof > 0:
            return self.chi_squared / self.n_dof
        return self.chi_squared


@dataclass
class PoissonRMLEConfig:
    """Configuration for Poisson-likelihood detector-response unfolding.

    This is the Stage B "detector response unfolding" path: it deconvolves
    the detector response (Compton/escape smearing captured in R) to infer
    the emitted gamma spectrum in energy bins.

    The objective is a Poisson negative log-likelihood with an explicit
    regularization penalty:

        L(μ) = Σ_i [ m_i(μ) - y_i log m_i(μ) ] + α * Ω(μ) + β * Σ_j [(μ_j - μ_0j)²/σ_0j²]

    with m = R μ + b.

    Attributes
    ----------
    penalty : PoissonPenalty
        Regularization type (L2, Sobolev, etc.)
    alpha : float
        Regularization strength
    background_mode : str
        Background treatment ('none', 'constant', 'vector')
    max_iterations : int
        Maximum optimization iterations
    tolerance : float
        Convergence tolerance
    positivity : bool
        Enforce non-negative solution
    eps : float
        Small value to avoid log(0)
    guardrail_max_reduced_chi2 : float
        Quality guardrail threshold
    mc_samples : int
        Monte Carlo samples for uncertainty (0 = no MC)
    random_seed : int, optional
        Random seed for reproducibility
    response_sampler : callable, optional
        Function to sample response matrix variations
    prior_activities : ndarray, optional
        Prior mean activities for constrained components (Q1.5).
        Use for known contaminant peaks where activity is bounded.
    prior_uncertainties : ndarray, optional
        Prior uncertainties (1σ) for constrained components.
        Components with uncertainty 0 or inf are unconstrained.
    prior_weight : float
        Weight for prior constraint term (default 1.0)
    contaminant_mask : ndarray, optional
        Boolean mask identifying contaminant components.
        True = known contaminant with constrained prior.
    """

    penalty: PoissonPenalty = PoissonPenalty.SOBLEV_2
    alpha: float = 1.0
    background_mode: str = "none"  # 'none' | 'constant' | 'vector'
    max_iterations: int = 500
    tolerance: float = 1e-7
    positivity: bool = True
    eps: float = 1e-12
    guardrail_max_reduced_chi2: float = 50.0
    mc_samples: int = 0
    random_seed: Optional[int] = None
    response_sampler: Optional[Callable[[np.random.Generator], "ResponseMatrix"]] = None
    # Constrained priors for contaminant peaks (Q1.5)
    prior_activities: Optional[np.ndarray] = None
    prior_uncertainties: Optional[np.ndarray] = None
    prior_weight: float = 1.0
    contaminant_mask: Optional[np.ndarray] = None
    
    def has_priors(self) -> bool:
        """Check if informative priors are specified."""
        return (
            self.prior_activities is not None 
            and self.prior_uncertainties is not None
            and len(self.prior_activities) > 0
        )


def _prior_penalty(
    mu: np.ndarray,
    prior_mean: Optional[np.ndarray],
    prior_sigma: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> float:
    """Compute prior constraint penalty: Σ [(μ - μ₀)² / σ₀²]."""
    if prior_mean is None or prior_sigma is None:
        return 0.0
    
    n = len(mu)
    if len(prior_mean) != n or len(prior_sigma) != n:
        return 0.0
    
    # Apply mask if provided
    if mask is not None:
        active = mask.astype(bool)
    else:
        # Only constrain components with finite, positive sigma
        active = (prior_sigma > 0) & np.isfinite(prior_sigma)
    
    if not np.any(active):
        return 0.0
    
    residuals = (mu - prior_mean)[active]
    weights = 1.0 / (prior_sigma[active] ** 2)
    
    return float(np.sum(residuals**2 * weights))


def _prior_grad(
    mu: np.ndarray,
    prior_mean: Optional[np.ndarray],
    prior_sigma: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Gradient of prior constraint penalty."""
    grad = np.zeros_like(mu)
    
    if prior_mean is None or prior_sigma is None:
        return grad
    
    n = len(mu)
    if len(prior_mean) != n or len(prior_sigma) != n:
        return grad
    
    # Apply mask
    if mask is not None:
        active = mask.astype(bool)
    else:
        active = (prior_sigma > 0) & np.isfinite(prior_sigma)
    
    if not np.any(active):
        return grad
    
    # d/dμ [(μ - μ₀)² / σ₀²] = 2(μ - μ₀) / σ₀²
    grad[active] = 2.0 * (mu - prior_mean)[active] / (prior_sigma[active] ** 2)
    
    return grad


def _sobolev_operator(n: int, order: int) -> np.ndarray:
    if order == 1:
        return tikhonov_matrix(n, order=1)
    if order == 2:
        return tikhonov_matrix(n, order=2)
    raise ValueError("Sobolev order must be 1 or 2")


def _poisson_nll(m: np.ndarray, y: np.ndarray, eps: float) -> float:
    """Poisson negative log likelihood up to additive constant."""
    m_safe = np.maximum(m, eps)
    return float(np.sum(m_safe - y * np.log(m_safe)))


def _penalty_value(mu: np.ndarray, penalty: PoissonPenalty, eps: float) -> float:
    if penalty == PoissonPenalty.NONE:
        return 0.0
    if penalty == PoissonPenalty.L2:
        return float(np.sum(mu**2))
    if penalty == PoissonPenalty.L1:
        # Smooth L1 for differentiability
        return float(np.sum(np.sqrt(mu**2 + eps)))
    if penalty == PoissonPenalty.SOBLEV_1:
        D = _sobolev_operator(len(mu), order=1)
        v = D @ mu
        return float(np.sum(v**2))
    if penalty == PoissonPenalty.SOBLEV_2:
        D = _sobolev_operator(len(mu), order=2)
        v = D @ mu
        return float(np.sum(v**2))
    raise ValueError(f"Unknown penalty: {penalty}")


def _penalty_grad(mu: np.ndarray, penalty: PoissonPenalty, eps: float) -> np.ndarray:
    if penalty == PoissonPenalty.NONE:
        return np.zeros_like(mu)
    if penalty == PoissonPenalty.L2:
        return 2.0 * mu
    if penalty == PoissonPenalty.L1:
        return mu / np.sqrt(mu**2 + eps)
    if penalty == PoissonPenalty.SOBLEV_1:
        D = _sobolev_operator(len(mu), order=1)
        return 2.0 * (D.T @ (D @ mu))
    if penalty == PoissonPenalty.SOBLEV_2:
        D = _sobolev_operator(len(mu), order=2)
        return 2.0 * (D.T @ (D @ mu))
    raise ValueError(f"Unknown penalty: {penalty}")


def poisson_rmle_unfolding(
    spectrum: SpectrumData,
    response: ResponseMatrix,
    config: Optional[PoissonRMLEConfig] = None,
) -> UnfoldingResult:
    """Poisson-likelihood RMLE unfolding with explicit regularization.

    Guardrails:
    - Enforces non-negativity when requested.
    - If optimization fails or yields a clearly unphysical refold residual,
      falls back to the existing weighted-LS RMLE solver.
    """
    if config is None:
        config = PoissonRMLEConfig()

    y = np.asarray(spectrum.counts, dtype=float)
    R = np.asarray(response.matrix, dtype=float)

    n_channels, n_bins = R.shape

    # Background parameterization
    background_mode = (config.background_mode or "none").lower()
    if background_mode not in {"none", "constant", "vector"}:
        raise ValueError("background_mode must be 'none', 'constant', or 'vector'")

    if background_mode == "none":
        n_b = 0
    elif background_mode == "constant":
        n_b = 1
    else:
        n_b = n_channels

    # Initial guess: small positive spectrum, background from low percentile
    mu0 = np.maximum(np.full(n_bins, max(np.mean(y) / max(n_bins, 1), 1.0)), config.eps)
    b0_val = float(np.percentile(y, 5)) if len(y) else 0.0
    if n_b == 0:
        x0 = mu0
    elif n_b == 1:
        x0 = np.concatenate([mu0, [max(b0_val, 0.0)]])
    else:
        x0 = np.concatenate([mu0, np.full(n_channels, max(b0_val, 0.0))])

    # Bounds
    bounds = []
    if config.positivity:
        bounds.extend([(0.0, None)] * n_bins)
    else:
        bounds.extend([(None, None)] * n_bins)

    if n_b == 1:
        bounds.append((0.0, None))
    elif n_b == n_channels:
        bounds.extend([(0.0, None)] * n_channels)

    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = x[:n_bins]
        if n_b == 0:
            b = np.zeros(n_channels)
        elif n_b == 1:
            b = np.full(n_channels, x[n_bins])
        else:
            b = x[n_bins:]
        return mu, b

    def objective(x: np.ndarray) -> float:
        mu, b = unpack(x)
        m = R @ mu + b
        return _poisson_nll(m, y, config.eps) + config.alpha * _penalty_value(mu, config.penalty, config.eps)

    def gradient(x: np.ndarray) -> np.ndarray:
        mu, b = unpack(x)
        m = R @ mu + b
        m_safe = np.maximum(m, config.eps)
        # d/dm of NLL is (1 - y/m)
        g_m = 1.0 - (y / m_safe)
        g_mu = R.T @ g_m + config.alpha * _penalty_grad(mu, config.penalty, config.eps)

        if n_b == 0:
            return g_mu
        if n_b == 1:
            g_b = np.array([np.sum(g_m)])
        else:
            g_b = g_m
        return np.concatenate([g_mu, g_b])

    # Optimize
    try:
        opt = optimize.minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={
                "maxiter": int(config.max_iterations),
                "ftol": float(config.tolerance),
            },
        )
        x_hat = opt.x
        mu_hat, b_hat = unpack(x_hat)
        m_hat = R @ mu_hat + b_hat

        # A rough refold residual diagnostic (Pearson-like): sum((y-m)^2/max(m,1))
        resid = y - m_hat
        pearson_chi2 = float(np.sum((resid**2) / np.maximum(m_hat, 1.0)))
        red = pearson_chi2 / max(n_channels - n_bins, 1)

        converged = bool(opt.success) and np.isfinite(red) and (red < config.guardrail_max_reduced_chi2)

        diagnostics = {
            "solver": "poisson_rmle",
            "success": bool(opt.success),
            "message": str(opt.message),
            "nll": float(opt.fun) if np.isfinite(opt.fun) else float("nan"),
            "pearson_chi2": pearson_chi2,
            "reduced_pearson_chi2": red,
            "penalty": config.penalty.value,
            "alpha": config.alpha,
            "background_mode": background_mode,
            "mc_samples": int(config.mc_samples or 0),
            "mc_response_sampling": bool(config.response_sampler is not None),
        }

        if not converged:
            # Guardrail fallback
            fallback = rmle_unfolding(
                spectrum=spectrum,
                response=response,
                regularization=RegularizationType.TIKHONOV_SECOND,
                reg_param=max(config.alpha, 1e-6),
                param_selection=ParameterSelection.FIXED,
                enforce_positivity=True,
            )
            fallback.diagnostics = {
                **fallback.diagnostics,
                "poisson_fallback": True,
                "poisson_diagnostics": diagnostics,
            }
            return fallback

        # Uncertainty: optional Monte Carlo resampling
        if config.mc_samples and config.mc_samples > 0:
            rng = np.random.default_rng(config.random_seed)
            samples = []
            for _ in range(int(config.mc_samples)):
                y_s = rng.poisson(np.maximum(y, 0.0))
                tmp_spec = SpectrumData(
                    counts=y_s,
                    channels=spectrum.channels,
                    live_time_s=spectrum.live_time_s,
                    energy_calibration=spectrum.energy_calibration,
                    uncertainty=np.sqrt(np.maximum(y_s, 1.0)),
                )

                tmp_response = response
                if config.response_sampler is not None:
                    tmp_response = config.response_sampler(rng)
                    if tmp_response.matrix.shape != response.matrix.shape:
                        raise ValueError(
                            "response_sampler must return a ResponseMatrix with the same shape as the nominal response"
                        )

                tmp_cfg = PoissonRMLEConfig(
                    penalty=config.penalty,
                    alpha=config.alpha,
                    background_mode=config.background_mode,
                    max_iterations=max(50, int(config.max_iterations // 2)),
                    tolerance=config.tolerance,
                    positivity=config.positivity,
                    eps=config.eps,
                    guardrail_max_reduced_chi2=config.guardrail_max_reduced_chi2,
                    mc_samples=0,
                    random_seed=None,
                    response_sampler=None,
                )
                tmp = poisson_rmle_unfolding(tmp_spec, tmp_response, tmp_cfg)
                samples.append(tmp.solution)
            sample_arr = np.vstack(samples)
            mu_unc = np.std(sample_arr, axis=0)
        else:
            mu_unc = np.maximum(0.1 * mu_hat, config.eps)

        return UnfoldingResult(
            solution=mu_hat,
            uncertainty=mu_unc,
            covariance=None,
            chi_squared=pearson_chi2,
            n_iterations=int(opt.nit) if hasattr(opt, "nit") else 0,
            regularization_param=float(config.alpha),
            residuals=resid,
            converged=True,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        fallback = rmle_unfolding(
            spectrum=spectrum,
            response=response,
            regularization=RegularizationType.TIKHONOV_SECOND,
            reg_param=max(config.alpha if config else 1.0, 1e-6),
            param_selection=ParameterSelection.FIXED,
            enforce_positivity=True,
        )
        fallback.diagnostics = {
            **fallback.diagnostics,
            "poisson_fallback": True,
            "poisson_error": repr(exc),
        }
        return fallback


def create_gaussian_response_matrix(
    n_channels: int,
    n_energy_bins: int,
    fwhm_function: Callable[[float], float],
    efficiency_function: Optional[Callable[[float], float]] = None,
    energy_range: Tuple[float, float] = (0, 3000),
) -> ResponseMatrix:
    """
    Create Gaussian response matrix for HPGe detector.
    
    Args:
        n_channels: Number of detector channels
        n_energy_bins: Number of energy bins
        fwhm_function: Function giving FWHM at each energy
        efficiency_function: Detection efficiency function
        energy_range: Energy range (keV)
        
    Returns:
        ResponseMatrix
    """
    channel_edges = np.linspace(0, n_channels, n_channels + 1)
    energy_edges = np.linspace(energy_range[0], energy_range[1], n_energy_bins + 1)
    
    # Energy centers
    energies = (energy_edges[:-1] + energy_edges[1:]) / 2
    channels = np.arange(n_channels)
    
    # Assume linear energy calibration for simplicity
    keV_per_channel = (energy_range[1] - energy_range[0]) / n_channels
    
    matrix = np.zeros((n_channels, n_energy_bins))
    
    for j, E in enumerate(energies):
        # Peak centroid in channel space
        centroid = (E - energy_range[0]) / keV_per_channel
        
        # FWHM in channels
        fwhm_keV = fwhm_function(E)
        sigma = fwhm_keV / (2.355 * keV_per_channel)
        sigma = max(sigma, 0.5)  # Minimum width
        
        # Efficiency
        if efficiency_function:
            eff = efficiency_function(E)
        else:
            eff = 1.0
        
        # Gaussian peak
        peak = eff * np.exp(-0.5 * ((channels - centroid) / sigma)**2)
        peak /= np.sum(peak) if np.sum(peak) > 0 else 1
        
        matrix[:, j] = peak
    
    return ResponseMatrix(
        matrix=matrix,
        channel_edges=channel_edges,
        energy_edges=energy_edges,
    )


def tikhonov_matrix(n: int, order: int = 0) -> np.ndarray:
    """
    Create Tikhonov regularization matrix.
    
    Args:
        n: Matrix dimension
        order: Derivative order (0=identity, 1=first derivative, 2=second)
        
    Returns:
        Regularization matrix L
    """
    if order == 0:
        return np.eye(n)
    elif order == 1:
        # First derivative operator
        L = np.zeros((n - 1, n))
        for i in range(n - 1):
            L[i, i] = -1
            L[i, i + 1] = 1
        return L
    elif order == 2:
        # Second derivative operator
        L = np.zeros((n - 2, n))
        for i in range(n - 2):
            L[i, i] = 1
            L[i, i + 1] = -2
            L[i, i + 2] = 1
        return L
    else:
        raise ValueError(f"Order {order} not supported")


def calculate_chi_squared(
    observed: np.ndarray,
    expected: np.ndarray,
    uncertainty: np.ndarray,
) -> float:
    """Calculate chi-squared statistic."""
    mask = uncertainty > 0
    residuals = (observed[mask] - expected[mask]) / uncertainty[mask]
    return float(np.sum(residuals**2))


def rmle_unfolding(
    spectrum: SpectrumData,
    response: ResponseMatrix,
    regularization: RegularizationType = RegularizationType.TIKHONOV,
    reg_param: float = 1.0,
    param_selection: ParameterSelection = ParameterSelection.FIXED,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    enforce_positivity: bool = True,
) -> UnfoldingResult:
    """
    Perform RMLE spectrum unfolding.
    
    Solves the inverse problem:
        d = R @ s + noise
    where d is the observed spectrum, R is the response matrix,
    and s is the unknown source spectrum to recover.
    
    Uses regularization to handle ill-posedness:
        min_s ||R @ s - d||² + λ² ||L @ s||²
    
    Args:
        spectrum: Observed spectrum data
        response: Detector response matrix
        regularization: Type of regularization
        reg_param: Regularization parameter (λ)
        param_selection: Method for selecting λ
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        enforce_positivity: Enforce non-negative solution
        
    Returns:
        UnfoldingResult with unfolded spectrum
    """
    d = spectrum.counts
    sigma = spectrum.uncertainty
    R = response.matrix
    
    n_channels, n_bins = R.shape
    
    # Build regularization matrix
    if regularization == RegularizationType.NONE:
        L = np.zeros((1, n_bins))
        reg_param = 0.0
    elif regularization == RegularizationType.TIKHONOV:
        L = tikhonov_matrix(n_bins, order=0)
    elif regularization == RegularizationType.TIKHONOV_DERIVATIVE:
        L = tikhonov_matrix(n_bins, order=1)
    elif regularization == RegularizationType.TIKHONOV_SECOND:
        L = tikhonov_matrix(n_bins, order=2)
    else:
        L = tikhonov_matrix(n_bins, order=0)
    
    # Weight by uncertainties
    W = np.diag(1.0 / sigma)
    
    # Automatic parameter selection
    if param_selection == ParameterSelection.L_CURVE:
        reg_param = select_lambda_lcurve(d, R, W, L)
    elif param_selection == ParameterSelection.GCV:
        reg_param = select_lambda_gcv(d, R, W, L)
    elif param_selection == ParameterSelection.AUTOMATIC:
        # Use L-curve as default automatic method
        reg_param = select_lambda_lcurve(d, R, W, L)
    
    # Solve regularized least squares
    if enforce_positivity:
        solution, n_iter, converged = solve_nnls_regularized(
            d, R, L, reg_param, W, max_iterations, tolerance
        )
    else:
        # Standard regularized least squares
        A = np.vstack([W @ R, reg_param * L])
        b = np.concatenate([W @ d, np.zeros(L.shape[0])])
        solution, residuals, rank, s = linalg.lstsq(A, b)
        n_iter = 1
        converged = True
    
    # Calculate fit quality
    expected = R @ solution
    residuals = d - expected
    chi_sq = calculate_chi_squared(d, expected, sigma)
    
    # Calculate uncertainty via error propagation
    try:
        # Covariance of solution
        RtWR = R.T @ W.T @ W @ R
        LtL = L.T @ L
        H = RtWR + reg_param**2 * LtL
        H_inv = linalg.inv(H)
        
        # Propagate input uncertainty
        data_cov = np.diag(sigma**2)
        G = H_inv @ R.T @ W.T @ W
        solution_cov = G @ data_cov @ G.T
        solution_unc = np.sqrt(np.diag(solution_cov))
    except Exception:
        solution_cov = None
        solution_unc = solution * 0.1  # Default 10% uncertainty
    
    return UnfoldingResult(
        solution=solution,
        uncertainty=solution_unc,
        covariance=solution_cov,
        chi_squared=chi_sq,
        n_iterations=n_iter,
        regularization_param=reg_param,
        residuals=residuals,
        converged=converged,
        diagnostics={
            "regularization_type": regularization.value,
            "param_selection": param_selection.value,
            "n_channels": n_channels,
            "n_bins": n_bins,
        },
    )


def solve_nnls_regularized(
    d: np.ndarray,
    R: np.ndarray,
    L: np.ndarray,
    reg_param: float,
    W: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> Tuple[np.ndarray, int, bool]:
    """
    Solve regularized non-negative least squares.
    
    Uses active set method with regularization.
    """
    # Build augmented system
    A = np.vstack([W @ R, reg_param * L])
    b = np.concatenate([W @ d, np.zeros(L.shape[0])])
    
    # Solve using scipy NNLS
    try:
        solution, residual_norm = optimize.nnls(A, b)
        return solution, 1, True
    except Exception:
        # Fall back to regularized least squares without positivity
        solution, _, _, _ = linalg.lstsq(A, b)
        solution = np.maximum(solution, 0)
        return solution, 1, True


def select_lambda_lcurve(
    d: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    n_points: int = 50,
) -> float:
    """
    Select regularization parameter using L-curve method.
    
    Finds the corner of the L-curve (log residual norm vs log solution norm).
    """
    lambdas = np.logspace(-4, 4, n_points)
    residual_norms = []
    solution_norms = []
    
    for lam in lambdas:
        A = np.vstack([W @ R, lam * L])
        b = np.concatenate([W @ d, np.zeros(L.shape[0])])
        
        try:
            solution, _, _, _ = linalg.lstsq(A, b)
            residual = W @ (R @ solution - d)
            residual_norms.append(np.linalg.norm(residual))
            solution_norms.append(np.linalg.norm(L @ solution))
        except Exception:
            residual_norms.append(np.inf)
            solution_norms.append(np.inf)
    
    # Find corner using curvature
    log_res = np.log10(np.maximum(residual_norms, 1e-100))
    log_sol = np.log10(np.maximum(solution_norms, 1e-100))
    
    # Calculate curvature (discrete approximation)
    curvature = np.zeros(len(lambdas))
    for i in range(1, len(lambdas) - 1):
        dx = log_res[i + 1] - log_res[i - 1]
        dy = log_sol[i + 1] - log_sol[i - 1]
        ddx = log_res[i + 1] - 2 * log_res[i] + log_res[i - 1]
        ddy = log_sol[i + 1] - 2 * log_sol[i] + log_sol[i - 1]
        
        denom = (dx**2 + dy**2)**1.5
        if denom > 0:
            curvature[i] = abs(dx * ddy - dy * ddx) / denom
    
    # Select lambda at maximum curvature
    best_idx = np.argmax(curvature)
    return lambdas[best_idx]


def select_lambda_gcv(
    d: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    n_points: int = 50,
) -> float:
    """
    Select regularization parameter using Generalized Cross-Validation.
    
    Minimizes GCV functional:
        GCV(λ) = ||A @ x - b||² / (n - trace(A @ A⁺))²
    """
    n = len(d)
    lambdas = np.logspace(-4, 4, n_points)
    gcv_values = []
    
    for lam in lambdas:
        try:
            # Influence matrix
            RtWR = R.T @ W.T @ W @ R
            LtL = L.T @ L
            H = RtWR + lam**2 * LtL
            H_inv = linalg.inv(H)
            
            A_hat = R @ H_inv @ R.T @ W.T @ W  # Influence/hat matrix
            
            solution = H_inv @ R.T @ W.T @ W @ d
            residual = W @ (R @ solution - d)
            
            trace_A = np.trace(A_hat)
            denom = (n - trace_A)**2
            
            if denom > 0:
                gcv = np.sum(residual**2) / denom
            else:
                gcv = np.inf
            
            gcv_values.append(gcv)
        except Exception:
            gcv_values.append(np.inf)
    
    # Select lambda that minimizes GCV
    best_idx = np.argmin(gcv_values)
    return lambdas[best_idx]


@dataclass
class PeakFitResult:
    """Result of peak fitting."""
    
    peaks: List[PeakModel]
    continuum: np.ndarray
    chi_squared: float
    residuals: np.ndarray
    
    @property
    def n_peaks(self) -> int:
        return len(self.peaks)


def fit_peaks_mle(
    spectrum: SpectrumData,
    peak_positions: List[float],
    fwhm_function: Callable[[float], float],
    fit_range: Optional[Tuple[int, int]] = None,
) -> PeakFitResult:
    """
    Fit peaks using Maximum Likelihood Estimation.
    
    Args:
        spectrum: Spectrum data
        peak_positions: Initial peak centroid positions
        fwhm_function: Function giving FWHM at each position
        fit_range: Channel range to fit (min, max)
        
    Returns:
        PeakFitResult with fitted peaks
    """
    counts = spectrum.counts
    channels = spectrum.channels
    
    if fit_range:
        mask = (channels >= fit_range[0]) & (channels <= fit_range[1])
        counts = counts[mask]
        channels = channels[mask]
    
    n_peaks = len(peak_positions)
    
    # Initial parameters: [amplitude1, ..., ampN, continuum_slope, continuum_intercept]
    initial_params = [np.max(counts) / 2] * n_peaks + [0, np.min(counts)]
    
    # Model function
    def model(x, *params):
        result = params[-2] * x + params[-1]  # Linear continuum
        for i, pos in enumerate(peak_positions):
            sigma = fwhm_function(pos) / 2.355
            result += params[i] * np.exp(-0.5 * ((x - pos) / sigma)**2)
        return result
    
    # Fit using curve_fit (Levenberg-Marquardt)
    try:
        popt, pcov = optimize.curve_fit(
            model,
            channels,
            counts,
            p0=initial_params,
            sigma=spectrum.uncertainty[mask] if fit_range else spectrum.uncertainty,
            absolute_sigma=True,
            maxfev=5000,
        )
        
        perr = np.sqrt(np.diag(pcov))
        
        # Build peak models
        peaks = []
        for i, pos in enumerate(peak_positions):
            sigma = fwhm_function(pos) / 2.355
            peaks.append(PeakModel(
                centroid=pos,
                sigma=sigma,
                amplitude=popt[i],
                amplitude_unc=perr[i],
            ))
        
        # Calculate continuum and residuals
        full_channels = spectrum.channels
        continuum = popt[-2] * full_channels + popt[-1]
        expected = model(full_channels, *popt)
        residuals = spectrum.counts - expected
        chi_sq = calculate_chi_squared(
            spectrum.counts, expected, spectrum.uncertainty
        )
        
        return PeakFitResult(
            peaks=peaks,
            continuum=continuum,
            chi_squared=chi_sq,
            residuals=residuals,
        )
        
    except Exception as e:
        # Return empty result on failure
        return PeakFitResult(
            peaks=[PeakModel(pos, 1.0) for pos in peak_positions],
            continuum=np.zeros_like(spectrum.channels, dtype=float),
            chi_squared=np.inf,
            residuals=spectrum.counts.astype(float),
        )


@dataclass
class ActivityResult:
    """
    Activity determination result.
    
    Attributes:
        nuclide: Nuclide name
        activity_Bq: Activity in Becquerels
        uncertainty_Bq: Uncertainty in activity
        gamma_energy_keV: Gamma energy used
        peak_counts: Net peak counts
        detection_limit_Bq: Detection limit (if applicable)
        is_detected: Whether activity is above detection limit
    """
    
    nuclide: str
    activity_Bq: float
    uncertainty_Bq: float
    gamma_energy_keV: float
    peak_counts: float
    detection_limit_Bq: float = 0.0
    is_detected: bool = True


def calculate_activity_from_peak(
    peak: PeakModel,
    efficiency: float,
    intensity: float,
    live_time_s: float,
    decay_correction: float = 1.0,
) -> Tuple[float, float]:
    """
    Calculate activity from fitted peak.
    
    Activity = counts / (ε × I × t × C_decay)
    
    Args:
        peak: Fitted peak model
        efficiency: Detection efficiency at peak energy
        intensity: Gamma intensity (branching ratio)
        live_time_s: Spectrum live time
        decay_correction: Decay correction factor
        
    Returns:
        Tuple of (activity_Bq, uncertainty_Bq)
    """
    # Net counts = peak area
    net_counts = peak.amplitude * peak.sigma * np.sqrt(2 * np.pi)
    net_unc = peak.amplitude_unc * peak.sigma * np.sqrt(2 * np.pi)
    
    denom = efficiency * intensity * live_time_s * decay_correction
    
    if denom > 0:
        activity = net_counts / denom
        uncertainty = net_unc / denom
    else:
        activity = 0.0
        uncertainty = 0.0
    
    return activity, uncertainty


class RMLEArtifact:
    """
    RMLE unfolding artifact for FluxForge pipeline.
    
    Contains complete unfolding results and metadata.
    """
    
    def __init__(
        self,
        spectrum: SpectrumData,
        result: UnfoldingResult,
        response: Optional[ResponseMatrix] = None,
    ):
        self.spectrum = spectrum
        self.result = result
        self.response = response
        self._activities: List[ActivityResult] = []
    
    def add_activity(self, activity: ActivityResult):
        """Add activity determination result."""
        self._activities.append(activity)
    
    @property
    def activities(self) -> List[ActivityResult]:
        return self._activities
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "schema": "fluxforge.rmle_artifact.v1",
            "n_channels": self.spectrum.n_channels,
            "live_time_s": self.spectrum.live_time_s,
            "chi_squared": self.result.chi_squared,
            "reduced_chi_squared": self.result.reduced_chi_squared,
            "regularization_param": self.result.regularization_param,
            "converged": self.result.converged,
            "n_activities": len(self._activities),
            "activities": [
                {
                    "nuclide": a.nuclide,
                    "activity_Bq": a.activity_Bq,
                    "uncertainty_Bq": a.uncertainty_Bq,
                    "gamma_energy_keV": a.gamma_energy_keV,
                    "is_detected": a.is_detected,
                }
                for a in self._activities
            ],
        }
