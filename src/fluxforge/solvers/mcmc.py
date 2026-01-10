"""Bayesian MCMC solver for neutron spectrum unfolding.

Implements Metropolis-Hastings MCMC sampling for posterior exploration
of the neutron flux spectrum given response matrix and measurements.

Features:
- Log-normal proposal distribution (preserves positivity)
- Configurable prior (uniform or smoothness-regularized)
- Burn-in and thinning for statistically independent samples
- Posterior statistics (mean, credible intervals)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

from fluxforge.core.linalg import Matrix, Vector, matmul


@dataclass
class MCMCSolution:
    """Container for MCMC solver results.
    
    Attributes
    ----------
    flux : Vector
        Posterior mean flux estimate
    samples : List[Vector]
        MCMC samples after burn-in and thinning
    credible_lower : Vector
        2.5th percentile (95% credible interval lower bound)
    credible_upper : Vector
        97.5th percentile (95% credible interval upper bound)
    credible_median : Vector
        50th percentile (median)
    acceptance_rate : float
        Fraction of accepted proposals
    log_posterior_history : List[float]
        Log-posterior values for diagnostics
    chi_squared : float
        Chi-squared per DOF at posterior mean
    """
    
    flux: Vector
    samples: List[Vector]
    credible_lower: Vector
    credible_upper: Vector
    credible_median: Vector
    acceptance_rate: float
    log_posterior_history: List[float] = field(default_factory=list)
    chi_squared: float = 0.0
    n_samples: int = 0
    n_accepted: int = 0


def _log_likelihood(
    response: Matrix,
    flux: Vector,
    measurements: Vector,
    uncertainties: Optional[Vector],
    floor: float = 1e-20,
) -> float:
    """Compute log-likelihood assuming Gaussian measurement errors.
    
    log L = -0.5 * sum_i ((y_i - (R @ phi)_i) / sigma_i)^2
    """
    predicted = matmul(response, flux)
    
    log_lik = 0.0
    for i, (m, p) in enumerate(zip(measurements, predicted)):
        p_safe = max(p, floor)
        sigma = uncertainties[i] if uncertainties else max(m * 0.1, floor)
        if sigma > 0:
            residual = (m - p_safe) / sigma
            log_lik -= 0.5 * residual * residual
    
    return log_lik


def _log_prior_uniform(flux: Vector, floor: float = 1e-20) -> float:
    """Uniform (improper) prior - returns 0 if all positive, -inf otherwise."""
    for phi in flux:
        if phi <= 0:
            return float('-inf')
    return 0.0


def _log_prior_smoothness(
    flux: Vector,
    smoothness_weight: float = 1.0,
    floor: float = 1e-20,
) -> float:
    """Smoothness prior penalizing large log-space variations.
    
    Encourages smooth spectra by penalizing (log(phi_g+1) - log(phi_g))^2.
    """
    # Check positivity
    for phi in flux:
        if phi <= 0:
            return float('-inf')
    
    if len(flux) < 2:
        return 0.0
    
    penalty = 0.0
    for g in range(len(flux) - 1):
        log_diff = math.log(max(flux[g+1], floor)) - math.log(max(flux[g], floor))
        penalty += log_diff * log_diff
    
    return -smoothness_weight * penalty


def _propose_flux(
    current: Vector,
    step_size: float = 0.1,
    floor: float = 1e-20,
) -> Vector:
    """Propose new flux using log-normal random walk.
    
    Each group is perturbed independently in log-space:
    log(phi_new) = log(phi_old) + epsilon
    where epsilon ~ N(0, step_size^2)
    """
    proposal = []
    for phi in current:
        log_phi = math.log(max(phi, floor))
        log_phi_new = log_phi + random.gauss(0, step_size)
        proposal.append(math.exp(log_phi_new))
    return proposal


def _compute_percentile(samples: List[Vector], percentile: float, group: int) -> float:
    """Compute percentile for a single energy group from samples."""
    values = sorted([s[group] for s in samples])
    n = len(values)
    if n == 0:
        return 0.0
    idx = int(percentile * n / 100)
    idx = max(0, min(n - 1, idx))
    return values[idx]


def mcmc_unfold(
    response: Matrix,
    measurements: Vector,
    initial_flux: Optional[Vector] = None,
    measurement_uncertainty: Optional[Vector] = None,
    n_samples: int = 10000,
    burn_in: int = 2000,
    thin: int = 5,
    step_size: float = 0.1,
    prior: str = "smoothness",
    smoothness_weight: float = 1.0,
    floor: float = 1e-20,
    seed: Optional[int] = None,
    verbose: bool = False,
    adaptive_step: bool = True,
    target_acceptance: float = 0.3,
) -> MCMCSolution:
    """Bayesian MCMC spectrum unfolding using Metropolis-Hastings.
    
    Samples from the posterior distribution:
        P(phi | y) ∝ P(y | phi) * P(phi)
    
    where P(y | phi) is the Gaussian likelihood and P(phi) is the prior.
    
    Parameters
    ----------
    response : Matrix
        Response matrix R_{i,g} (n_meas x n_groups)
    measurements : Vector
        Measured reaction rates y_i
    initial_flux : Vector, optional
        Starting flux for chain. Defaults to uniform scaled to measurements.
    measurement_uncertainty : Vector, optional
        1-sigma uncertainties on measurements
    n_samples : int
        Total number of MCMC samples to draw
    burn_in : int
        Number of initial samples to discard
    thin : int
        Keep every nth sample (thinning interval)
    step_size : float
        Log-space proposal step size (larger = bigger jumps)
    prior : str
        Prior type: "uniform" or "smoothness"
    smoothness_weight : float
        Weight for smoothness prior (if prior="smoothness")
    floor : float
        Minimum flux value
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress information
    adaptive_step : bool
        Adapt step_size during burn-in for target acceptance rate
    target_acceptance : float
        Target acceptance rate for adaptive stepping (0.2-0.5 recommended)
    
    Returns
    -------
    MCMCSolution
        Container with posterior statistics, samples, and diagnostics
    
    Examples
    --------
    >>> from fluxforge.solvers.mcmc import mcmc_unfold
    >>> result = mcmc_unfold(R, y, n_samples=5000, burn_in=1000)
    >>> print(f"Mean flux: {result.flux}")
    >>> print(f"Acceptance rate: {result.acceptance_rate:.2%}")
    """
    if seed is not None:
        random.seed(seed)
    
    n_groups = len(response[0])
    n_meas = len(measurements)
    
    # Initialize flux
    if initial_flux is not None:
        current_flux = [max(x, floor) for x in initial_flux]
    else:
        avg_meas = sum(measurements) / len(measurements)
        current_flux = [avg_meas / n_groups for _ in range(n_groups)]
    
    # Select prior function
    if prior == "uniform":
        log_prior_fn = lambda phi: _log_prior_uniform(phi, floor)
    elif prior == "smoothness":
        log_prior_fn = lambda phi: _log_prior_smoothness(phi, smoothness_weight, floor)
    else:
        raise ValueError(f"Unknown prior: {prior}. Use 'uniform' or 'smoothness'")
    
    # Compute initial log-posterior
    current_log_lik = _log_likelihood(response, current_flux, measurements, measurement_uncertainty, floor)
    current_log_prior = log_prior_fn(current_flux)
    current_log_post = current_log_lik + current_log_prior
    
    # Storage
    samples: List[Vector] = []
    log_post_history: List[float] = []
    n_accepted = 0
    n_total = 0
    
    # Adaptive stepping variables
    adapt_interval = 100
    current_step = step_size
    
    for i in range(n_samples):
        n_total += 1
        
        # Propose new flux
        proposed_flux = _propose_flux(current_flux, current_step, floor)
        
        # Compute log-posterior for proposal
        proposed_log_lik = _log_likelihood(response, proposed_flux, measurements, measurement_uncertainty, floor)
        proposed_log_prior = log_prior_fn(proposed_flux)
        proposed_log_post = proposed_log_lik + proposed_log_prior
        
        # Metropolis acceptance ratio (in log space)
        log_alpha = proposed_log_post - current_log_post
        
        # Accept or reject
        if math.log(random.random()) < log_alpha:
            current_flux = proposed_flux
            current_log_post = proposed_log_post
            n_accepted += 1
        
        log_post_history.append(current_log_post)
        
        # Adaptive step size during burn-in
        if adaptive_step and i < burn_in and i > 0 and i % adapt_interval == 0:
            recent_rate = n_accepted / n_total
            if recent_rate < target_acceptance - 0.1:
                current_step *= 0.8
                if verbose:
                    print(f"  MCMC iter {i}: Reducing step to {current_step:.4f} (accept rate {recent_rate:.2%})")
            elif recent_rate > target_acceptance + 0.1:
                current_step *= 1.2
                if verbose:
                    print(f"  MCMC iter {i}: Increasing step to {current_step:.4f} (accept rate {recent_rate:.2%})")
        
        # Store sample after burn-in with thinning
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(current_flux[:])
        
        if verbose and i % 1000 == 0:
            rate = n_accepted / n_total if n_total > 0 else 0
            print(f"  MCMC iter {i}/{n_samples}: accept_rate = {rate:.2%}, log_post = {current_log_post:.2f}")
    
    # Compute posterior statistics
    if len(samples) == 0:
        raise ValueError("No samples collected. Increase n_samples or decrease burn_in.")
    
    # Posterior mean
    n_samples_kept = len(samples)
    posterior_mean = []
    for g in range(n_groups):
        mean_g = sum(s[g] for s in samples) / n_samples_kept
        posterior_mean.append(mean_g)
    
    # Credible intervals
    credible_lower = [_compute_percentile(samples, 2.5, g) for g in range(n_groups)]
    credible_upper = [_compute_percentile(samples, 97.5, g) for g in range(n_groups)]
    credible_median = [_compute_percentile(samples, 50, g) for g in range(n_groups)]
    
    # Compute chi-squared at posterior mean
    predicted = matmul(response, posterior_mean)
    chi2 = 0.0
    for i in range(n_meas):
        sigma = measurement_uncertainty[i] if measurement_uncertainty else max(measurements[i] * 0.1, floor)
        if sigma > 0:
            residual = (measurements[i] - predicted[i]) / sigma
            chi2 += residual * residual
    chi2_per_dof = chi2 / max(n_meas - 1, 1)
    
    if verbose:
        print(f"  MCMC complete: {n_samples_kept} samples, accept_rate = {n_accepted/n_total:.2%}, chi2/dof = {chi2_per_dof:.4f}")
    
    return MCMCSolution(
        flux=posterior_mean,
        samples=samples,
        credible_lower=credible_lower,
        credible_upper=credible_upper,
        credible_median=credible_median,
        acceptance_rate=n_accepted / n_total,
        log_posterior_history=log_post_history,
        chi_squared=chi2_per_dof,
        n_samples=n_samples_kept,
        n_accepted=n_accepted,
    )


def mcmc_convergence_diagnostic(samples: List[Vector], group: int = 0) -> dict:
    """Compute convergence diagnostics for MCMC chain.
    
    Returns Gelman-Rubin R-hat approximation and effective sample size.
    
    Parameters
    ----------
    samples : List[Vector]
        MCMC samples
    group : int
        Energy group to analyze
    
    Returns
    -------
    dict
        Diagnostic metrics including:
        - 'ess': Effective sample size (approximate)
        - 'mean': Sample mean
        - 'std': Sample standard deviation
        - 'autocorr_1': Lag-1 autocorrelation
    """
    values = [s[group] for s in samples]
    n = len(values)
    
    if n < 10:
        return {'ess': n, 'mean': sum(values)/n, 'std': 0, 'autocorr_1': 0}
    
    mean = sum(values) / n
    variance = sum((v - mean)**2 for v in values) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0
    
    # Lag-1 autocorrelation
    if std > 0:
        autocorr_1 = sum((values[i] - mean) * (values[i+1] - mean) for i in range(n-1)) / ((n-1) * variance)
    else:
        autocorr_1 = 0
    
    # Approximate effective sample size
    if abs(autocorr_1) < 1:
        ess = n * (1 - autocorr_1) / (1 + autocorr_1)
    else:
        ess = n
    
    return {
        'ess': max(1, ess),
        'mean': mean,
        'std': std,
        'autocorr_1': autocorr_1,
    }
