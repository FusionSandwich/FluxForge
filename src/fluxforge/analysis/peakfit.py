"""
Peak Fitting Module for HPGe Gamma Spectroscopy

Provides Gaussian peak fitting and analysis for gamma-ray spectra.
Includes background estimation, peak finding, and uncertainty propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize, signal


@dataclass
class GaussianPeak:
    """
    Representation of a Gaussian peak.
    
    The peak function is:
        f(x) = amplitude * exp(-(x - centroid)² / (2 * sigma²))
    
    Attributes
    ----------
    centroid : float
        Peak centroid (energy or channel)
    amplitude : float
        Peak height (counts)
    sigma : float
        Gaussian standard deviation
    centroid_unc : float
        Centroid uncertainty
    amplitude_unc : float
        Amplitude uncertainty
    sigma_unc : float
        Sigma uncertainty
    
    Properties
    ----------
    fwhm : float
        Full width at half maximum
    area : float
        Integrated peak area (counts)
    area_uncertainty : float
        Area uncertainty
    """
    
    centroid: float
    amplitude: float
    sigma: float
    centroid_unc: float = 0.0
    amplitude_unc: float = 0.0
    sigma_unc: float = 0.0
    
    @property
    def fwhm(self) -> float:
        """Full width at half maximum."""
        return 2.355 * self.sigma
    
    @property
    def fwhm_unc(self) -> float:
        """FWHM uncertainty."""
        return 2.355 * self.sigma_unc
    
    @property
    def area(self) -> float:
        """Integrated Gaussian area = amplitude * sigma * sqrt(2π)."""
        return self.amplitude * self.sigma * np.sqrt(2 * np.pi)
    
    @property
    def area_uncertainty(self) -> float:
        """Area uncertainty from error propagation."""
        # d(area)/d(amp) = sigma * sqrt(2π)
        # d(area)/d(sigma) = amp * sqrt(2π)
        sqrt_2pi = np.sqrt(2 * np.pi)
        da_damp = self.sigma * sqrt_2pi
        da_dsig = self.amplitude * sqrt_2pi
        
        var = (da_damp * self.amplitude_unc)**2 + (da_dsig * self.sigma_unc)**2
        return np.sqrt(var)
    
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate Gaussian at x."""
        return self.amplitude * np.exp(
            -(x - self.centroid)**2 / (2 * self.sigma**2)
        )
    
    def __repr__(self) -> str:
        return (
            f"GaussianPeak(centroid={self.centroid:.2f}±{self.centroid_unc:.3f}, "
            f"area={self.area:.1f}±{self.area_uncertainty:.1f}, "
            f"FWHM={self.fwhm:.3f})"
        )


@dataclass
class HypermetPeak:
    """
    Representation of a Hypermet peak (Theuerkauf model from hdtv).
    
    The Hypermet function is a Gaussian with optional:
    - Left exponential tail (low-energy tailing from incomplete charge collection)
    - Right exponential tail (rare, for special cases)
    - Step function (Compton edge effect)
    
    The peak function is:
        f(x) = Gaussian(x) + LeftTail(x) + RightTail(x) + Step(x)
    
    where:
        Gaussian = A * exp(-(x-μ)²/(2σ²))
        LeftTail = A * η_L * exp((x-μ)/β_L) * erfc((x-μ)/(√2·σ) + σ/(√2·β_L))
        Step = A * η_S * erfc((x-μ)/(√2·σ))
    
    Attributes
    ----------
    centroid : float
        Peak centroid (energy or channel)
    amplitude : float
        Peak height (counts)
    sigma : float
        Gaussian standard deviation
    tail_left : float
        Left tail parameter β_L (decay constant). None = no tail.
    tail_left_fraction : float
        Left tail intensity fraction η_L
    tail_right : float
        Right tail parameter β_R. None = no tail.
    tail_right_fraction : float
        Right tail intensity fraction η_R
    step_height : float
        Step function height η_S. None = no step.
    """
    
    centroid: float
    amplitude: float
    sigma: float
    tail_left: Optional[float] = None
    tail_left_fraction: float = 0.0
    tail_right: Optional[float] = None
    tail_right_fraction: float = 0.0
    step_height: Optional[float] = None
    
    # Uncertainties
    centroid_unc: float = 0.0
    amplitude_unc: float = 0.0
    sigma_unc: float = 0.0
    
    @property
    def fwhm(self) -> float:
        """Full width at half maximum (Gaussian component)."""
        return 2.355 * self.sigma
    
    @property
    def has_left_tail(self) -> bool:
        return self.tail_left is not None and self.tail_left > 0
    
    @property
    def has_right_tail(self) -> bool:
        return self.tail_right is not None and self.tail_right > 0
    
    @property
    def has_step(self) -> bool:
        return self.step_height is not None and self.step_height > 0
    
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate Hypermet function at x."""
        from scipy.special import erfc
        
        x = np.asarray(x)
        mu = self.centroid
        sigma = self.sigma
        A = self.amplitude
        
        # Gaussian component
        z = (x - mu) / sigma
        result = A * np.exp(-0.5 * z**2)
        
        # Left tail (low-energy)
        if self.has_left_tail:
            beta_L = self.tail_left
            eta_L = self.tail_left_fraction
            # Convolution of Gaussian with exponential tail
            arg1 = (x - mu) / beta_L
            arg2 = z / np.sqrt(2) + sigma / (np.sqrt(2) * beta_L)
            tail = A * eta_L * np.exp(arg1 + sigma**2 / (2 * beta_L**2)) * erfc(arg2)
            # Only add where x < mu (left side)
            result = result + 0.5 * tail
        
        # Right tail (high-energy) - less common
        if self.has_right_tail:
            beta_R = self.tail_right
            eta_R = self.tail_right_fraction
            arg1 = -(x - mu) / beta_R
            arg2 = -z / np.sqrt(2) + sigma / (np.sqrt(2) * beta_R)
            tail = A * eta_R * np.exp(arg1 + sigma**2 / (2 * beta_R**2)) * erfc(arg2)
            result = result + 0.5 * tail
        
        # Step function (Compton edge)
        if self.has_step:
            eta_S = self.step_height
            step = A * eta_S * 0.5 * erfc(z / np.sqrt(2))
            result = result + step
        
        return result
    
    @property
    def area(self) -> float:
        """Approximate integrated area (numerical for complex shapes)."""
        # For pure Gaussian: A * sigma * sqrt(2π)
        # Tails add extra area
        gaussian_area = self.amplitude * self.sigma * np.sqrt(2 * np.pi)
        
        # Approximate tail contributions
        if self.has_left_tail:
            gaussian_area *= (1 + self.tail_left_fraction)
        if self.has_right_tail:
            gaussian_area *= (1 + self.tail_right_fraction)
        
        return gaussian_area
    
    def __repr__(self) -> str:
        parts = [f"HypermetPeak(centroid={self.centroid:.2f}, FWHM={self.fwhm:.3f}"]
        if self.has_left_tail:
            parts.append(f", tail_L={self.tail_left:.2f}")
        if self.has_right_tail:
            parts.append(f", tail_R={self.tail_right:.2f}")
        if self.has_step:
            parts.append(f", step={self.step_height:.3f}")
        parts.append(")")
        return "".join(parts)


@dataclass
class PeakFitResult:
    """
    Complete peak fitting result.
    
    Attributes
    ----------
    peak : GaussianPeak
        Fitted peak parameters
    background : np.ndarray
        Background under peak
    background_model : str
        Background model type used
    residuals : np.ndarray
        Fit residuals
    chi_squared : float
        Chi-squared of fit
    dof : int
        Degrees of freedom
    fit_region : Tuple[int, int]
        Channel range of fit
    covariance : Optional[np.ndarray]
        Parameter covariance matrix
    success : bool
        Whether fit converged
    message : str
        Fit status message
    """
    
    peak: GaussianPeak
    background: np.ndarray
    background_model: str = 'linear'
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_squared: float = 0.0
    dof: int = 1
    fit_region: Tuple[int, int] = (0, 0)
    covariance: Optional[np.ndarray] = None
    success: bool = True
    message: str = ""
    
    @property
    def reduced_chi_squared(self) -> float:
        """Reduced chi-squared."""
        return self.chi_squared / self.dof if self.dof > 0 else 0.0
    
    @property
    def net_counts(self) -> float:
        """Net peak counts (area - background)."""
        return self.peak.area
    
    @property
    def net_counts_uncertainty(self) -> float:
        """Net counts uncertainty."""
        return self.peak.area_uncertainty


def five_point_smooth(counts: np.ndarray) -> np.ndarray:
    """Apply five-point smoothing (Phillips, 1978) for low-statistics spectra."""
    if len(counts) < 5:
        raise ValueError("Input array must have at least 5 elements for smoothing.")

    smoothed = counts.astype(float).copy()
    for i in range(2, len(counts) - 2):
        smoothed[i] = (
            counts[i - 2]
            + counts[i + 2]
            + 2.0 * counts[i - 1]
            + 2.0 * counts[i + 1]
            + 3.0 * counts[i]
        ) / 9.0
    return smoothed


def gaussian(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float
) -> np.ndarray:
    """Gaussian function."""
    return amplitude * np.exp(-(x - centroid)**2 / (2 * sigma**2))


def gaussian_with_linear_bg(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    bg_slope: float,
    bg_intercept: float
) -> np.ndarray:
    """Gaussian plus linear background."""
    return gaussian(x, amplitude, centroid, sigma) + bg_slope * x + bg_intercept


def gaussian_with_step_bg(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    bg_left: float,
    bg_right: float,
    step_fraction: float = 0.5
) -> np.ndarray:
    """
    Gaussian with step background (Compton edge model).
    
    Step function transitions from bg_left to bg_right at centroid.
    """
    from scipy.special import erfc
    
    peak = gaussian(x, amplitude, centroid, sigma)
    
    # Error function step
    step = 0.5 * (bg_left + bg_right) + 0.5 * (bg_left - bg_right) * erfc(
        (x - centroid) / (np.sqrt(2) * sigma)
    )
    
    return peak + step


# =============================================================================
# Advanced Peak Shapes (Epic R - Becquerel Parity)
# =============================================================================

SQRT_TWO = np.sqrt(2.0)
FWHM_SIG_RATIO = np.sqrt(8 * np.log(2))  # ≈ 2.35482


def expgauss(
    x: np.ndarray,
    amplitude: float = 1.0,
    centroid: float = 0.0,
    sigma: float = 1.0,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Exponentially-modified Gaussian (EMG).
    
    Convolution of a Gaussian with an exponential decay. Commonly used
    for chromatography peaks and asymmetric detector responses.
    
    The EMG is defined as:
        f(x) = (A*γ/2) * exp(γ*(μ - x + γσ²/2)) * erfc((μ + γσ² - x)/(√2·σ))
    
    Parameters
    ----------
    x : np.ndarray
        X-values (energy or channel)
    amplitude : float
        Peak amplitude
    centroid : float
        Peak center (mu)
    sigma : float
        Gaussian width parameter
    gamma : float
        Exponential modifier. Positive = right tail, negative = left tail.
        |gamma| is the exponential decay rate.
    
    Returns
    -------
    np.ndarray
        EMG function values
    
    Notes
    -----
    Based on becquerel.core.fitting.expgauss implementation.
    """
    from scipy.special import erfc
    
    sign = np.sign(gamma)
    gamma_abs = np.abs(gamma)
    gss = gamma_abs * sigma * sigma
    
    arg1 = sign * gamma_abs * (centroid - x + gss / 2.0)
    arg2 = sign * (centroid + gss - x) / (SQRT_TWO * sigma)
    
    return amplitude * (gamma_abs / 2) * np.exp(arg1) * erfc(arg2)


def gauss_dbl_exp(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    ltail_ratio: float = 0.0,
    ltail_slope: float = 0.01,
    ltail_cutoff: float = 1.0,
    rtail_ratio: float = 0.0,
    rtail_slope: float = 0.01,
    rtail_cutoff: float = 1.0
) -> np.ndarray:
    """
    Gaussian with exponential tails on both sides.
    
    Extension of Namboodiri et al. (https://www.osti.gov/biblio/392720)
    for asymmetric peaks such as those from CZT detectors.
    
    Parameters
    ----------
    x : np.ndarray
        X-values (energy or channel)
    amplitude : float
        Peak amplitude
    centroid : float
        Peak center (mu)
    sigma : float
        Gaussian width
    ltail_ratio : float
        Left tail amplitude ratio (tail_amp / peak_amp)
    ltail_slope : float
        Left tail exponential slope (steepness)
    ltail_cutoff : float
        Left tail cutoff parameter (length)
    rtail_ratio : float
        Right tail amplitude ratio
    rtail_slope : float
        Right tail exponential slope
    rtail_cutoff : float
        Right tail cutoff parameter
    
    Returns
    -------
    np.ndarray
        Peak function values
    
    Notes
    -----
    Based on becquerel.core.fitting.gauss_dbl_exp implementation.
    
    The function uses a "Heaviside convolution" approach where:
    - Left tail is only added for x < centroid
    - Right tail is only added for x > centroid
    - Both tails smoothly join the Gaussian via the cutoff parameter
    """
    x = np.asarray(x)
    alpha = -1.0 / (2.0 * sigma**2)
    
    # Initialize tail functions
    ltail_func = np.zeros_like(x, dtype=float)
    rtail_func = np.zeros_like(x, dtype=float)
    
    # Left tail (x < centroid)
    if ltail_ratio > 0:
        mask = x < centroid
        if np.any(mask):
            ltail_func[mask] = amplitude * ltail_ratio * np.exp(ltail_slope * (x[mask] - centroid))
            ltail_func[mask] *= -np.expm1(ltail_cutoff * alpha * ((x[mask] - centroid) ** 2))
    
    # Right tail (x > centroid)
    if rtail_ratio > 0:
        mask = x > centroid
        if np.any(mask):
            rtail_func[mask] = amplitude * rtail_ratio * np.exp(rtail_slope * (x[mask] - centroid))
            rtail_func[mask] *= -np.expm1(rtail_cutoff * alpha * ((x[mask] - centroid) ** 2))
    
    # Gaussian core + tails
    return amplitude * np.exp(alpha * ((x - centroid) ** 2)) + ltail_func + rtail_func


def gauss_with_erf(
    x: np.ndarray,
    amp_gauss: float,
    amp_erf: float,
    centroid: float,
    sigma: float
) -> np.ndarray:
    """
    Gaussian plus error function step.
    
    Combines a Gaussian peak with an error function step, useful for
    modeling peaks on a Compton edge.
    
    Parameters
    ----------
    x : np.ndarray
        X-values
    amp_gauss : float
        Gaussian amplitude
    amp_erf : float
        Error function amplitude
    centroid : float
        Peak center (mu)
    sigma : float
        Width parameter
    
    Returns
    -------
    np.ndarray
        Combined function values
    
    Notes
    -----
    Based on becquerel.core.fitting.gausserf implementation.
    """
    from scipy.special import erf
    
    gauss_part = (amp_gauss / sigma / np.sqrt(2.0 * np.pi) * 
                  np.exp(-((x - centroid) ** 2.0) / (2.0 * sigma**2.0)))
    erf_part = amp_erf * 0.5 * (1.0 - erf((x - centroid) / (SQRT_TWO * sigma)))
    
    return gauss_part + erf_part


# =============================================================================
# Poisson Likelihood Fitting (Epic R - Becquerel Parity)
# =============================================================================


def poisson_neg_log_likelihood(
    y_model: np.ndarray,
    y_data: np.ndarray
) -> float:
    """
    Negative log-likelihood for Poisson-distributed data.
    
    NLL = sum(y_model - y_data * log(y_model))
    
    This is the correct objective for fitting count data, as it properly
    weights low-count bins. Minimizing this is equivalent to maximizing
    the Poisson likelihood.
    
    Parameters
    ----------
    y_model : np.ndarray
        Model predictions (must be positive)
    y_data : np.ndarray
        Observed counts (integer or float)
    
    Returns
    -------
    float
        Negative log-likelihood value
    
    Notes
    -----
    Uses scipy.special.xlogy for numerical stability when y_data = 0.
    Based on becquerel.core.fitting.poisson_loss implementation.
    """
    from scipy.special import xlogy
    
    # Ensure model is positive to avoid log(0)
    y_model = np.maximum(y_model, 1e-10)
    return np.sum(y_model - xlogy(y_data, y_model))


def fit_peak_poisson(
    channels: np.ndarray,
    counts: np.ndarray,
    initial_centroid: float,
    initial_sigma: float = 3.0,
    model: str = 'gaussian',
    background: str = 'linear',
    fit_range: Optional[Tuple[float, float]] = None,
    **model_kwargs
) -> PeakFitResult:
    """
    Fit a peak using Poisson maximum likelihood.
    
    Unlike chi-squared fitting, Poisson likelihood properly handles
    low-count bins and doesn't require binning or uncertainty estimates.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Observed counts per channel
    initial_centroid : float
        Initial guess for peak centroid
    initial_sigma : float
        Initial guess for peak width
    model : str
        Peak model: 'gaussian', 'expgauss', 'gauss_dbl_exp'
    background : str
        Background model: 'linear', 'constant', 'none'
    fit_range : tuple, optional
        (low, high) channel range to fit. Defaults to centroid ± 5*sigma.
    **model_kwargs
        Additional model parameters (e.g., gamma for expgauss)
    
    Returns
    -------
    PeakFitResult
        Fitted peak result with uncertainties
    
    Notes
    -----
    Uses scipy.optimize.minimize with Nelder-Mead for robustness.
    """
    # Determine fit range
    if fit_range is None:
        half_width = max(5 * initial_sigma, 10)
        low = max(0, int(initial_centroid - half_width))
        high = min(len(counts), int(initial_centroid + half_width))
    else:
        low, high = int(fit_range[0]), int(fit_range[1])
    
    x_fit = channels[low:high]
    y_fit = counts[low:high]
    
    # Initial amplitude estimate
    bg_estimate = np.mean([np.mean(y_fit[:3]), np.mean(y_fit[-3:])])
    peak_counts = y_fit - bg_estimate
    initial_amplitude = np.max(peak_counts) if np.max(peak_counts) > 0 else 1.0
    
    # Build model function based on type
    if model == 'gaussian':
        if background == 'linear':
            def model_func(params):
                amp, cent, sig, m, b = params
                return gaussian(x_fit, amp, cent, sig) + m * x_fit + b
            p0 = [initial_amplitude, initial_centroid, initial_sigma, 0.0, bg_estimate]
            param_names = ['amplitude', 'centroid', 'sigma', 'bg_slope', 'bg_intercept']
        else:  # constant background
            def model_func(params):
                amp, cent, sig, b = params
                return gaussian(x_fit, amp, cent, sig) + b
            p0 = [initial_amplitude, initial_centroid, initial_sigma, bg_estimate]
            param_names = ['amplitude', 'centroid', 'sigma', 'bg_const']
    
    elif model == 'expgauss':
        gamma = model_kwargs.get('gamma', 0.1)
        if background == 'linear':
            def model_func(params):
                amp, cent, sig, gam, m, b = params
                return expgauss(x_fit, amp, cent, sig, gam) + m * x_fit + b
            p0 = [initial_amplitude, initial_centroid, initial_sigma, gamma, 0.0, bg_estimate]
            param_names = ['amplitude', 'centroid', 'sigma', 'gamma', 'bg_slope', 'bg_intercept']
        else:
            def model_func(params):
                amp, cent, sig, gam, b = params
                return expgauss(x_fit, amp, cent, sig, gam) + b
            p0 = [initial_amplitude, initial_centroid, initial_sigma, gamma, bg_estimate]
            param_names = ['amplitude', 'centroid', 'sigma', 'gamma', 'bg_const']
    
    elif model == 'gauss_dbl_exp':
        if background == 'linear':
            def model_func(params):
                amp, cent, sig, lr, ls, lc, rr, rs, rc, m, b = params
                return gauss_dbl_exp(x_fit, amp, cent, sig, lr, ls, lc, rr, rs, rc) + m * x_fit + b
            p0 = [initial_amplitude, initial_centroid, initial_sigma,
                  0.1, 0.01, 1.0, 0.1, 0.01, 1.0, 0.0, bg_estimate]
            param_names = ['amplitude', 'centroid', 'sigma',
                          'ltail_ratio', 'ltail_slope', 'ltail_cutoff',
                          'rtail_ratio', 'rtail_slope', 'rtail_cutoff',
                          'bg_slope', 'bg_intercept']
        else:
            def model_func(params):
                amp, cent, sig, lr, ls, lc, rr, rs, rc, b = params
                return gauss_dbl_exp(x_fit, amp, cent, sig, lr, ls, lc, rr, rs, rc) + b
            p0 = [initial_amplitude, initial_centroid, initial_sigma,
                  0.1, 0.01, 1.0, 0.1, 0.01, 1.0, bg_estimate]
            param_names = ['amplitude', 'centroid', 'sigma',
                          'ltail_ratio', 'ltail_slope', 'ltail_cutoff',
                          'rtail_ratio', 'rtail_slope', 'rtail_cutoff', 'bg_const']
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Objective function
    def objective(params):
        y_model = model_func(params)
        # Ensure positive model values
        y_model = np.maximum(y_model, 1e-10)
        return poisson_neg_log_likelihood(y_model, y_fit)
    
    # Fit using Nelder-Mead (robust for Poisson likelihood)
    result = optimize.minimize(objective, p0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Extract results
    popt = result.x
    y_model = model_func(popt)
    residuals = y_fit - y_model
    
    # Compute chi-squared equivalent for diagnostics
    # Using Pearson chi-squared for comparison
    chi2 = np.sum((residuals ** 2) / np.maximum(y_model, 1))
    dof = len(y_fit) - len(p0)
    
    # Build peak object
    if model == 'gaussian':
        amp_idx, cent_idx, sig_idx = 0, 1, 2
    elif model == 'expgauss':
        amp_idx, cent_idx, sig_idx = 0, 1, 2
    else:  # gauss_dbl_exp
        amp_idx, cent_idx, sig_idx = 0, 1, 2
    
    peak = GaussianPeak(
        centroid=popt[cent_idx],
        amplitude=popt[amp_idx],
        sigma=popt[sig_idx],
        centroid_unc=0.0,  # Would need Hessian for proper uncertainties
        amplitude_unc=0.0,
        sigma_unc=0.0
    )
    
    # Background
    if background == 'linear':
        bg = popt[-2] * x_fit + popt[-1]
    else:
        bg = np.full_like(x_fit, popt[-1], dtype=float)
    
    return PeakFitResult(
        peak=peak,
        background=bg,
        background_model=background,
        residuals=residuals,
        chi_squared=chi2,
        dof=dof,
        fit_region=(low, high),
        covariance=None,
        success=result.success,
        message=f"Poisson MLE fit ({model}): {result.message}"
    )


@dataclass
class ExpGaussPeak:
    """
    Exponentially-modified Gaussian peak representation.
    
    Attributes
    ----------
    centroid : float
        Peak center (mu)
    amplitude : float
        Peak amplitude
    sigma : float
        Gaussian width
    gamma : float
        Exponential modifier (positive = right tail)
    """
    centroid: float
    amplitude: float
    sigma: float
    gamma: float = 0.0
    centroid_unc: float = 0.0
    amplitude_unc: float = 0.0
    sigma_unc: float = 0.0
    gamma_unc: float = 0.0
    
    @property
    def fwhm(self) -> float:
        """Approximate FWHM (exact for gamma → 0)."""
        # For pure Gaussian: 2.355 * sigma
        # EMG FWHM depends on gamma but this is a reasonable approximation
        return FWHM_SIG_RATIO * self.sigma
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ExpGauss at x."""
        return expgauss(x, self.amplitude, self.centroid, self.sigma, self.gamma)
    
    def __repr__(self) -> str:
        return (f"ExpGaussPeak(centroid={self.centroid:.2f}, "
                f"sigma={self.sigma:.3f}, gamma={self.gamma:.4f})")


@dataclass
class DoubleExpPeak:
    """
    Gaussian with double exponential tails peak representation.
    
    Suitable for asymmetric peaks from CZT or similar detectors.
    """
    centroid: float
    amplitude: float
    sigma: float
    ltail_ratio: float = 0.0
    ltail_slope: float = 0.01
    ltail_cutoff: float = 1.0
    rtail_ratio: float = 0.0
    rtail_slope: float = 0.01
    rtail_cutoff: float = 1.0
    
    # Uncertainties
    centroid_unc: float = 0.0
    amplitude_unc: float = 0.0
    sigma_unc: float = 0.0
    
    @property
    def fwhm(self) -> float:
        """Approximate FWHM (Gaussian core)."""
        return FWHM_SIG_RATIO * self.sigma
    
    @property
    def has_left_tail(self) -> bool:
        return self.ltail_ratio > 0
    
    @property
    def has_right_tail(self) -> bool:
        return self.rtail_ratio > 0
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate at x."""
        return gauss_dbl_exp(
            x, self.amplitude, self.centroid, self.sigma,
            self.ltail_ratio, self.ltail_slope, self.ltail_cutoff,
            self.rtail_ratio, self.rtail_slope, self.rtail_cutoff
        )
    
    def __repr__(self) -> str:
        parts = [f"DoubleExpPeak(centroid={self.centroid:.2f}, FWHM={self.fwhm:.3f}"]
        if self.has_left_tail:
            parts.append(f", L_tail={self.ltail_ratio:.2%}")
        if self.has_right_tail:
            parts.append(f", R_tail={self.rtail_ratio:.2%}")
        parts.append(")")
        return "".join(parts)


def estimate_background(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_regions: Optional[List[Tuple[int, int]]] = None,
    method: str = 'snip',
    iterations: Union[int, List[int]] = 20,
    window_size: int = 10
) -> np.ndarray:
    """
    Estimate background under spectrum.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    peak_regions : list of (start, end) tuples, optional
        Known peak regions to exclude
    method : str
        Background method: 'snip', 'linear', 'rolling_min'
    iterations : int
        Number of SNIP iterations
    window_size : int
        Window size for rolling methods
    
    Returns
    -------
    np.ndarray
        Estimated background
    """
    if method == 'snip':
        return _snip_background(counts, iterations)
    elif method == 'rolling_min':
        return _rolling_min_background(counts, window_size)
    elif method == 'linear':
        return _linear_background(channels, counts, peak_regions)
    else:
        raise ValueError(f"Unknown background method: {method}")


def _lls_transform(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return np.log(np.log(np.sqrt(values + 1.0) + 1.0) + 1.0)


def _inverse_lls_transform(values: np.ndarray) -> np.ndarray:
    return (np.exp(np.exp(values) - 1.0) - 1.0) ** 2 - 1.0


def _snip_background(counts: np.ndarray, iterations: Union[int, List[int]] = 20) -> np.ndarray:
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm.

    Uses the LLS transform (as in peakingduck) for better Poisson behavior.
    """
    iteration_list = (
        list(range(1, int(iterations) + 1))
        if isinstance(iterations, int)
        else [int(val) for val in iterations]
    )
    if not iteration_list:
        return counts.astype(float).copy()

    snipped = _lls_transform(counts.astype(float))
    for order in iteration_list:
        if order <= 0:
            continue
        updated = snipped.copy()
        for i in range(order, len(snipped) - order):
            avg = 0.5 * (snipped[i - order] + snipped[i + order])
            if avg < snipped[i]:
                updated[i] = avg
        snipped = updated

    background = _inverse_lls_transform(snipped)
    return np.maximum(background, 0.0)


def _rolling_min_background(counts: np.ndarray, window_size: int = 10) -> np.ndarray:
    """Simple rolling minimum background."""
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(counts, size=window_size)


def _linear_background(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_regions: Optional[List[Tuple[int, int]]] = None
) -> np.ndarray:
    """Fit linear background excluding peak regions."""
    mask = np.ones(len(channels), dtype=bool)
    
    if peak_regions:
        for start, end in peak_regions:
            mask[(channels >= start) & (channels <= end)] = False
    
    if np.sum(mask) < 2:
        # Not enough points, return flat background
        return np.full_like(counts, np.median(counts), dtype=float)
    
    coeffs = np.polyfit(channels[mask], counts[mask], 1)
    return np.polyval(coeffs, channels)


def subtract_background(
    counts: np.ndarray,
    background: np.ndarray,
    minimum: float = 0.0
) -> np.ndarray:
    """
    Subtract background from spectrum.
    
    Parameters
    ----------
    counts : np.ndarray
        Spectrum counts
    background : np.ndarray
        Background to subtract
    minimum : float
        Minimum value for result (default 0)
    
    Returns
    -------
    np.ndarray
        Background-subtracted spectrum
    """
    return np.maximum(counts - background, minimum)


def window_peak_finder(
    channels: np.ndarray,
    counts: np.ndarray,
    threshold: float = 2.0,
    inner_window: int = 2,
    outer_window: int = 40,
    enforce_maximum: bool = True,
    min_distance: int = 3,
) -> List[Tuple[int, float, float]]:
    """
    Window-based peak finder (inspired by peakingduck WindowPeakFinder).
    
    For each channel, computes the local mean and standard deviation from
    surrounding channels (excluding the inner window around the test point).
    A peak is detected when the count exceeds mean + threshold * stddev.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    threshold : float
        Number of sigma above local background for peak detection
    inner_window : int
        Number of channels to exclude on each side of test point
    outer_window : int  
        Number of channels to include on each side for statistics
    enforce_maximum : bool
        If True, require point to be local maximum
    min_distance : int
        Minimum distance between consecutive peak detections
    
    Returns
    -------
    list of (channel, counts, significance) tuples
        Detected peak positions with their counts and significance
    """
    n = len(counts)
    if n < 2 * outer_window + 1:
        return []
    
    peaks = []
    last_peak_idx = -min_distance - 1
    
    for i in range(outer_window, n - outer_window):
        # Extract window values (excluding inner region)
        left_start = max(0, i - outer_window)
        left_end = max(0, i - inner_window)
        right_start = min(n, i + inner_window + 1)
        right_end = min(n, i + outer_window + 1)
        
        window_vals = np.concatenate([
            counts[left_start:left_end],
            counts[right_start:right_end]
        ])
        
        if len(window_vals) < 3:
            continue
        
        mean_val = np.mean(window_vals)
        std_val = np.std(window_vals, ddof=1)
        
        if std_val <= 0:
            std_val = np.sqrt(max(mean_val, 1))  # Poisson approximation
        
        local_threshold = mean_val + threshold * std_val
        value = counts[i]
        
        if value > local_threshold:
            # Check local maximum condition
            if enforce_maximum:
                if i > 0 and counts[i-1] >= value:
                    continue
                if i < n - 1 and counts[i+1] >= value:
                    continue
            
            # Check minimum distance from last peak
            if i - last_peak_idx < min_distance:
                continue
            
            significance = (value - mean_val) / std_val
            peaks.append((int(channels[i]), float(value), float(significance)))
            last_peak_idx = i
    
    return peaks


def chunked_peak_finder(
    channels: np.ndarray,
    counts: np.ndarray,
    threshold: float = 3.0,
    n_chunks: int = 10,
    min_distance: int = 5,
) -> List[Tuple[int, float]]:
    """
    Chunked peak finder (inspired by peakingduck ChunkedSimplePeakFinder).
    
    Breaks spectrum into chunks and applies threshold relative to each chunk's
    local statistics. Useful for spectra with varying background levels.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    threshold : float
        Number of sigma above local chunk background
    n_chunks : int
        Number of chunks to divide spectrum into
    min_distance : int
        Minimum distance between peaks
    
    Returns
    -------
    list of (channel, significance) tuples
        Detected peak positions and significances
    """
    n = len(counts)
    chunk_size = n // n_chunks
    if chunk_size < 10:
        chunk_size = 10
        n_chunks = max(1, n // chunk_size)
    
    all_peaks = []
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n) if chunk_idx < n_chunks - 1 else n
        
        chunk_channels = channels[start:end]
        chunk_counts = counts[start:end]
        
        if len(chunk_counts) < 5:
            continue
        
        # Local statistics for this chunk
        chunk_mean = np.mean(chunk_counts)
        chunk_std = np.std(chunk_counts, ddof=1)
        if chunk_std <= 0:
            chunk_std = np.sqrt(max(chunk_mean, 1))
        
        local_threshold = chunk_mean + threshold * chunk_std
        
        # Find peaks in chunk using scipy
        peaks, properties = signal.find_peaks(
            chunk_counts,
            height=local_threshold,
            distance=min_distance
        )
        
        # Add to all peaks with global channel indices
        for p in peaks:
            global_ch = int(chunk_channels[p])
            significance = (chunk_counts[p] - chunk_mean) / chunk_std
            all_peaks.append((global_ch, float(significance)))
    
    # Sort by channel and remove duplicates within min_distance
    all_peaks.sort(key=lambda x: x[0])
    filtered = []
    last_ch = -min_distance - 1
    for ch, sig in all_peaks:
        if ch - last_ch >= min_distance:
            filtered.append((ch, sig))
            last_ch = ch
    
    return filtered


def auto_find_peaks(
    channels: np.ndarray,
    counts: np.ndarray,
    threshold: float = 3.0,
    min_distance: int = 5,
    background_method: str = 'snip',
    finder_method: str = 'scipy',
) -> List[Tuple[int, float]]:
    """
    Automatically find peaks in spectrum.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    threshold : float
        Detection threshold (number of sigma above background)
    min_distance : int
        Minimum distance between peaks (channels)
    background_method : str
        Method for background estimation
    finder_method : str
        Peak finding algorithm: 'scipy' (default), 'window', or 'chunked'
    
    Returns
    -------
    list of (channel, significance) tuples
        Detected peak positions and significances
    """
    if finder_method == 'window':
        # Window method returns 3-tuple, convert to 2-tuple
        result = window_peak_finder(channels, counts, threshold, min_distance=min_distance)
        return [(ch, sig) for ch, _, sig in result]
    
    elif finder_method == 'chunked':
        return chunked_peak_finder(channels, counts, threshold, min_distance=min_distance)
    
    # Default: scipy-based with SNIP background
    # Estimate background
    background = estimate_background(channels, counts, method=background_method)
    
    # Calculate significance
    net = counts - background
    sigma = np.sqrt(np.maximum(background, 1))  # Poisson uncertainty
    significance = net / sigma
    
    # Find local maxima
    peaks, properties = signal.find_peaks(
        significance,
        height=threshold,
        distance=min_distance
    )
    
    # Get peak significances
    heights = properties['peak_heights']
    
    return [(int(channels[p]), float(h)) for p, h in zip(peaks, heights)]


def fit_single_peak(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channel: int,
    fit_width: int = 10,
    background_model: str = 'linear',
    fix_centroid: bool = False,
    initial_sigma: Optional[float] = None
) -> PeakFitResult:
    """
    Fit single Gaussian peak to spectrum region.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers (full spectrum)
    counts : np.ndarray
        Spectrum counts (full spectrum)
    peak_channel : int
        Approximate peak channel
    fit_width : int
        Half-width of fit region in channels
    background_model : str
        Background model: 'linear', 'constant', 'step'
    fix_centroid : bool
        If True, fix centroid at peak_channel
    initial_sigma : float, optional
        Initial guess for sigma
    
    Returns
    -------
    PeakFitResult
        Fitting result with peak parameters and uncertainties
    """
    # Extract fit region
    idx_peak = np.argmin(np.abs(channels - peak_channel))
    
    ch_lo = max(0, idx_peak - fit_width)
    ch_hi = min(len(channels), idx_peak + fit_width + 1)
    
    x = channels[ch_lo:ch_hi].astype(float)
    y = counts[ch_lo:ch_hi].astype(float)
    
    # Weights for chi-squared (Poisson uncertainty)
    weights = 1.0 / np.maximum(np.sqrt(y), 1.0)
    
    # Initial guesses
    amplitude_guess = y.max() - y.min()
    centroid_guess = float(peak_channel)
    sigma_guess = initial_sigma if initial_sigma else fit_width / 3.0
    
    # Background initial guess
    bg_left = np.mean(y[:3])
    bg_right = np.mean(y[-3:])
    bg_slope = (bg_right - bg_left) / (x[-1] - x[0])
    bg_intercept = bg_left - bg_slope * x[0]
    
    # Define model and fit
    if background_model == 'linear':
        def model(x, amp, cent, sig, slope, intercept):
            return gaussian(x, amp, cent, sig) + slope * x + intercept
        
        p0 = [amplitude_guess, centroid_guess, sigma_guess, bg_slope, bg_intercept]
        
        bounds_lower = [0, x[0], 0.5, -np.inf, -np.inf]
        bounds_upper = [np.inf, x[-1], fit_width, np.inf, np.inf]
        
        if fix_centroid:
            bounds_lower[1] = centroid_guess - 0.01
            bounds_upper[1] = centroid_guess + 0.01
    
    elif background_model == 'constant':
        def model(x, amp, cent, sig, bg):
            return gaussian(x, amp, cent, sig) + bg
        
        p0 = [amplitude_guess, centroid_guess, sigma_guess, (bg_left + bg_right) / 2]
        
        bounds_lower = [0, x[0], 0.5, 0]
        bounds_upper = [np.inf, x[-1], fit_width, np.inf]
        
        if fix_centroid:
            bounds_lower[1] = centroid_guess - 0.01
            bounds_upper[1] = centroid_guess + 0.01
    
    elif background_model == 'step':
        def model(x, amp, cent, sig, bg_l, bg_r):
            return gaussian_with_step_bg(x, amp, cent, sig, bg_l, bg_r)
        
        p0 = [amplitude_guess, centroid_guess, sigma_guess, bg_left, bg_right]
        
        bounds_lower = [0, x[0], 0.5, 0, 0]
        bounds_upper = [np.inf, x[-1], fit_width, np.inf, np.inf]
        
        if fix_centroid:
            bounds_lower[1] = centroid_guess - 0.01
            bounds_upper[1] = centroid_guess + 0.01
    
    else:
        raise ValueError(f"Unknown background model: {background_model}")
    
    # Perform fit
    try:
        popt, pcov = optimize.curve_fit(
            model, x, y,
            p0=p0,
            sigma=1.0 / weights,
            absolute_sigma=True,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        
        perr = np.sqrt(np.diag(pcov))
        success = True
        message = "Fit converged"
        
    except Exception as e:
        # Fit failed, return initial guess with large uncertainties
        popt = p0
        perr = np.array([1e6] * len(p0))
        pcov = None
        success = False
        message = str(e)
    
    # Extract parameters
    amplitude, centroid, sigma = popt[0], popt[1], popt[2]
    amp_err, cent_err, sig_err = perr[0], perr[1], perr[2]
    
    peak = GaussianPeak(
        centroid=centroid,
        amplitude=amplitude,
        sigma=sigma,
        centroid_unc=cent_err,
        amplitude_unc=amp_err,
        sigma_unc=sig_err,
    )
    
    # Calculate background array
    if background_model == 'linear':
        background = popt[3] * x + popt[4]
    elif background_model == 'constant':
        background = np.full_like(x, popt[3])
    elif background_model == 'step':
        from scipy.special import erfc
        background = 0.5 * (popt[3] + popt[4]) + 0.5 * (popt[3] - popt[4]) * erfc(
            (x - centroid) / (np.sqrt(2) * sigma)
        )
    
    # Calculate residuals and chi-squared
    y_fit = model(x, *popt)
    residuals = y - y_fit
    chi_sq = np.sum((residuals * weights)**2)
    dof = len(y) - len(popt)
    
    return PeakFitResult(
        peak=peak,
        background=background,
        background_model=background_model,
        residuals=residuals,
        chi_squared=chi_sq,
        dof=dof,
        fit_region=(int(x[0]), int(x[-1])),
        covariance=pcov,
        success=success,
        message=message,
    )


def fit_multiple_peaks(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channels: List[int],
    fit_width: int = 10,
    background_model: str = 'linear',
    share_sigma: bool = False
) -> List[PeakFitResult]:
    """
    Fit multiple peaks simultaneously.
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers
    counts : np.ndarray
        Spectrum counts
    peak_channels : list of int
        Approximate peak channels
    fit_width : int
        Half-width of fit region per peak
    background_model : str
        Background model
    share_sigma : bool
        If True, all peaks share same sigma
    
    Returns
    -------
    list of PeakFitResult
        Fitting results for each peak
    """
    # For well-separated peaks, fit individually
    results = []
    
    for peak_ch in sorted(peak_channels):
        result = fit_single_peak(
            channels, counts, peak_ch,
            fit_width=fit_width,
            background_model=background_model
        )
        results.append(result)
    
    return results


def calculate_activity(
    net_counts: float,
    net_counts_unc: float,
    live_time: float,
    efficiency: float,
    efficiency_unc: float,
    emission_probability: float,
    emission_probability_unc: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate activity from peak net counts.
    
    A = N / (t_live * ε * I_γ)
    
    Parameters
    ----------
    net_counts : float
        Net peak counts
    net_counts_unc : float
        Net counts uncertainty
    live_time : float
        Measurement live time (seconds)
    efficiency : float
        Detection efficiency at peak energy
    efficiency_unc : float
        Efficiency uncertainty
    emission_probability : float
        Gamma emission probability
    emission_probability_unc : float
        Emission probability uncertainty
    
    Returns
    -------
    activity : float
        Activity in Bq
    activity_unc : float
        Activity uncertainty in Bq
    """
    # Activity calculation
    denominator = live_time * efficiency * emission_probability
    activity = net_counts / denominator
    
    # Relative uncertainty propagation
    rel_unc_sq = (net_counts_unc / net_counts)**2
    rel_unc_sq += (efficiency_unc / efficiency)**2
    
    if emission_probability_unc > 0:
        rel_unc_sq += (emission_probability_unc / emission_probability)**2
    
    activity_unc = activity * np.sqrt(rel_unc_sq)
    
    return activity, activity_unc


def peak_report(
    results: List[PeakFitResult],
    energies: Optional[np.ndarray] = None,
    energy_calibration: Optional[List[float]] = None
) -> str:
    """
    Generate text report of peak fitting results.
    
    Parameters
    ----------
    results : list of PeakFitResult
        Peak fitting results
    energies : np.ndarray, optional
        Energy array (if already calibrated)
    energy_calibration : list of float, optional
        Energy calibration coefficients [a0, a1, ...]
    
    Returns
    -------
    str
        Formatted report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PEAK FITTING REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    header = f"{'Peak':^4}  {'Centroid':^12}  {'Energy':^10}  {'Area':^14}  {'FWHM':^8}  {'χ²/dof':^8}"
    lines.append(header)
    lines.append("-" * 80)
    
    for i, result in enumerate(results, 1):
        peak = result.peak
        
        # Calculate energy if calibration provided
        if energy_calibration:
            energy = sum(c * peak.centroid**j for j, c in enumerate(energy_calibration))
            energy_str = f"{energy:.2f} keV"
        elif energies is not None:
            ch = int(peak.centroid)
            if 0 <= ch < len(energies):
                energy_str = f"{energies[ch]:.2f} keV"
            else:
                energy_str = "N/A"
        else:
            energy_str = "N/A"
        
        chi2_str = f"{result.reduced_chi_squared:.2f}" if result.dof > 0 else "N/A"
        
        line = (
            f"{i:^4}  "
            f"{peak.centroid:.2f}±{peak.centroid_unc:.2f}  "
            f"{energy_str:^10}  "
            f"{peak.area:.1f}±{peak.area_uncertainty:.1f}  "
            f"{peak.fwhm:.3f}  "
            f"{chi2_str:^8}"
        )
        lines.append(line)
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# Hypermet Peak Fitting (Theuerkauf model inspired by hdtv)
# =============================================================================

def hypermet_function(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    tail_left: float = 0.0,
    tail_left_frac: float = 0.0,
    step_height: float = 0.0,
) -> np.ndarray:
    """
    Hypermet peak function with left tail and step.
    
    Parameters
    ----------
    x : np.ndarray
        x values (channels or energy)
    amplitude : float
        Peak amplitude
    centroid : float
        Peak centroid
    sigma : float
        Gaussian sigma
    tail_left : float
        Left tail decay parameter (β_L). 0 = no tail.
    tail_left_frac : float
        Left tail intensity fraction (η_L)
    step_height : float
        Step height (η_S). 0 = no step.
    
    Returns
    -------
    np.ndarray
        Function values
    """
    from scipy.special import erfc
    
    z = (x - centroid) / sigma
    
    # Gaussian component
    result = amplitude * np.exp(-0.5 * z**2)
    
    # Left tail (if enabled) - require reasonable parameters to avoid overflow
    if tail_left > 0.1 and tail_left_frac > 0.001:
        arg1 = (x - centroid) / tail_left + sigma**2 / (2 * tail_left**2)
        arg2 = z / np.sqrt(2) + sigma / (np.sqrt(2) * tail_left)
        # Clip arg1 to avoid overflow in exp
        arg1_clipped = np.clip(arg1, -500, 500)
        tail = amplitude * tail_left_frac * np.exp(arg1_clipped) * erfc(arg2)
        result = result + 0.5 * tail
    
    # Step function (if enabled)
    if step_height > 0.0001:
        step = amplitude * step_height * 0.5 * erfc(z / np.sqrt(2))
        result = result + step
    
    return result


def hypermet_with_linear_bg(
    x: np.ndarray,
    amplitude: float,
    centroid: float,
    sigma: float,
    tail_left: float,
    tail_left_frac: float,
    step_height: float,
    bg_slope: float,
    bg_intercept: float,
) -> np.ndarray:
    """Hypermet function with linear background."""
    return hypermet_function(
        x, amplitude, centroid, sigma,
        tail_left, tail_left_frac, step_height
    ) + bg_slope * x + bg_intercept


def fit_hypermet_peak(
    channels: np.ndarray,
    counts: np.ndarray,
    peak_channel: int,
    fit_width: int = 15,
    enable_tail: bool = True,
    enable_step: bool = False,
    initial_sigma: Optional[float] = None,
) -> Tuple[HypermetPeak, PeakFitResult]:
    """
    Fit Hypermet peak to spectrum region.
    
    The Hypermet model includes a Gaussian core plus optional left tail
    (incomplete charge collection) and step function (Compton edge).
    
    Parameters
    ----------
    channels : np.ndarray
        Channel numbers (full spectrum)
    counts : np.ndarray
        Spectrum counts (full spectrum)
    peak_channel : int
        Approximate peak channel
    fit_width : int
        Half-width of fit region in channels
    enable_tail : bool
        Enable left tail fitting
    enable_step : bool
        Enable step function fitting
    initial_sigma : float, optional
        Initial guess for sigma
    
    Returns
    -------
    peak : HypermetPeak
        Fitted Hypermet peak
    result : PeakFitResult
        Full fitting result (uses Gaussian representation for compatibility)
    """
    # Extract fit region
    idx_peak = np.argmin(np.abs(channels - peak_channel))
    ch_lo = max(0, idx_peak - fit_width)
    ch_hi = min(len(channels), idx_peak + fit_width + 1)
    
    x = channels[ch_lo:ch_hi].astype(float)
    y = counts[ch_lo:ch_hi].astype(float)
    
    # Weights for chi-squared
    weights = 1.0 / np.maximum(np.sqrt(y), 1.0)
    
    # Initial guesses
    amplitude_guess = y.max() - y.min()
    centroid_guess = float(peak_channel)
    sigma_guess = initial_sigma if initial_sigma else fit_width / 4.0
    
    bg_left = np.mean(y[:3])
    bg_right = np.mean(y[-3:])
    bg_slope = (bg_right - bg_left) / (x[-1] - x[0])
    bg_intercept = bg_left - bg_slope * x[0]
    
    # Initial tail parameters - use small but non-zero values for bounds
    tail_guess = sigma_guess * 2 if enable_tail else 0.05
    tail_frac_guess = 0.1 if enable_tail else 0.0005
    step_guess = 0.01 if enable_step else 0.00005
    
    # Parameter setup
    p0 = [
        amplitude_guess,
        centroid_guess,
        sigma_guess,
        tail_guess,
        tail_frac_guess,
        step_guess,
        bg_slope,
        bg_intercept,
    ]
    
    # Bounds - allow small non-zero values to avoid singularities
    bounds_lower = [
        0,                    # amplitude
        x[0],                 # centroid
        0.5,                  # sigma
        0,                    # tail_left
        0,                    # tail_left_frac
        0,                    # step_height
        -np.inf,              # bg_slope
        -np.inf,              # bg_intercept
    ]
    
    bounds_upper = [
        np.inf,               # amplitude
        x[-1],                # centroid
        fit_width,            # sigma
        fit_width * 3 if enable_tail else 0.1,    # tail_left
        1.0 if enable_tail else 0.001,            # tail_left_frac
        0.5 if enable_step else 0.0001,           # step_height
        np.inf,               # bg_slope
        np.inf,               # bg_intercept
    ]
    
    # Perform fit
    try:
        popt, pcov = optimize.curve_fit(
            hypermet_with_linear_bg, x, y,
            p0=p0,
            sigma=1.0 / weights,
            absolute_sigma=True,
            bounds=(bounds_lower, bounds_upper),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        success = True
        message = "Fit converged"
    except Exception as e:
        popt = p0
        perr = np.array([1e6] * len(p0))
        pcov = None
        success = False
        message = str(e)
    
    # Extract parameters
    amplitude = popt[0]
    centroid = popt[1]
    sigma = popt[2]
    tail_left = popt[3] if enable_tail else None
    tail_left_frac = popt[4]
    step_height = popt[5] if enable_step else None
    
    # Create Hypermet peak
    hypermet_peak = HypermetPeak(
        centroid=centroid,
        amplitude=amplitude,
        sigma=sigma,
        tail_left=tail_left,
        tail_left_fraction=tail_left_frac,
        step_height=step_height,
        centroid_unc=perr[1],
        amplitude_unc=perr[0],
        sigma_unc=perr[2],
    )
    
    # Create Gaussian equivalent for compatibility
    gaussian_peak = GaussianPeak(
        centroid=centroid,
        amplitude=amplitude,
        sigma=sigma,
        centroid_unc=perr[1],
        amplitude_unc=perr[0],
        sigma_unc=perr[2],
    )
    
    # Background and residuals
    background = popt[6] * x + popt[7]
    y_fit = hypermet_with_linear_bg(x, *popt)
    residuals = y - y_fit
    chi_sq = np.sum((residuals * weights)**2)
    dof = len(y) - len(popt)
    
    result = PeakFitResult(
        peak=gaussian_peak,
        background=background,
        background_model='hypermet',
        residuals=residuals,
        chi_squared=chi_sq,
        dof=dof,
        fit_region=(int(x[0]), int(x[-1])),
        covariance=pcov,
        success=success,
        message=message,
    )
    
    return hypermet_peak, result
