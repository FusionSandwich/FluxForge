"""
Prior covariance models for GLS spectral adjustment.

This module provides explicit prior covariance models as required
for STAYSL parity. The GLS adjustment must run with an explicit
prior covariance model, either user-supplied or from a structured default.

Reference: STAYSL PNNL, GLS adjustment methodology
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.linalg import cholesky, eigh


class PriorCovarianceModel(Enum):
    """Types of prior covariance models."""
    
    USER_SUPPLIED = "user_supplied"  # User provides full matrix
    DIAGONAL = "diagonal"  # Only diagonal (uncorrelated groups)
    FRACTIONAL_UNIFORM = "fractional_uniform"  # Same relative uncertainty all groups
    FRACTIONAL_REGIONAL = "fractional_regional"  # Different uncertainty by energy region
    LETHARGY_CORRELATED = "lethargy_correlated"  # Correlation decays with lethargy distance
    MCNP_STATISTICS = "mcnp_statistics"  # Based on MCNP tally statistics


class ResponseUncertaintyPolicy(Enum):
    """Policy for handling response (cross section) uncertainty."""
    
    IGNORE = "ignore"  # Do not include response uncertainty
    AUGMENT_VY = "augment_vy"  # Add propagated response uncertainty to V_y
    NUISANCE_PARAMETERS = "nuisance"  # Treat as nuisance parameters
    MONTE_CARLO = "monte_carlo"  # Monte Carlo propagation


@dataclass
class EnergyRegion:
    """
    Energy region for regional uncertainty specification.
    
    Attributes:
        name: Region name (e.g., "thermal", "epithermal", "fast")
        energy_low_ev: Lower energy bound
        energy_high_ev: Upper energy bound
        fractional_uncertainty: Relative uncertainty (σ/μ)
    """
    
    name: str
    energy_low_ev: float
    energy_high_ev: float
    fractional_uncertainty: float


# Standard energy regions for TRIGA-type spectra
STANDARD_REGIONS = [
    EnergyRegion("thermal", 0.0, 0.5, 0.30),  # 30% uncertainty
    EnergyRegion("epithermal", 0.5, 1e5, 0.25),  # 25% uncertainty
    EnergyRegion("fast_low", 1e5, 1e6, 0.20),  # 20% uncertainty
    EnergyRegion("fast_high", 1e6, 2e7, 0.35),  # 35% uncertainty
]


def create_diagonal_prior_covariance(
    prior_flux: np.ndarray,
    fractional_uncertainty: float = 0.25,
) -> np.ndarray:
    """
    Create diagonal prior covariance matrix.
    
    V_ij = 0 for i ≠ j
    V_ii = (f * φ_i)²
    
    Args:
        prior_flux: Prior flux values
        fractional_uncertainty: Relative uncertainty (σ/μ)
        
    Returns:
        Diagonal covariance matrix
    """
    n = len(prior_flux)
    V = np.zeros((n, n))
    for i in range(n):
        sigma_i = fractional_uncertainty * max(prior_flux[i], 1e-30)
        V[i, i] = sigma_i ** 2
    return V


def create_regional_prior_covariance(
    prior_flux: np.ndarray,
    energy_bounds_ev: np.ndarray,
    regions: Optional[List[EnergyRegion]] = None,
) -> np.ndarray:
    """
    Create prior covariance with regional fractional uncertainties.
    
    Different energy regions get different relative uncertainties,
    but groups within regions are uncorrelated.
    
    Args:
        prior_flux: Prior flux values
        energy_bounds_ev: Group energy boundaries
        regions: Energy regions with uncertainties (default: STANDARD_REGIONS)
        
    Returns:
        Diagonal covariance matrix with regional uncertainties
    """
    if regions is None:
        regions = STANDARD_REGIONS
    
    n = len(prior_flux)
    V = np.zeros((n, n))
    
    # Get group midpoint energies
    group_energies = np.sqrt(energy_bounds_ev[:-1] * energy_bounds_ev[1:])
    
    for i in range(n):
        E_i = group_energies[i]
        
        # Find region for this group
        frac_unc = 0.25  # Default
        for region in regions:
            if region.energy_low_ev <= E_i < region.energy_high_ev:
                frac_unc = region.fractional_uncertainty
                break
        
        sigma_i = frac_unc * max(prior_flux[i], 1e-30)
        V[i, i] = sigma_i ** 2
    
    return V


def create_lethargy_correlated_covariance(
    prior_flux: np.ndarray,
    energy_bounds_ev: np.ndarray,
    fractional_uncertainty: float = 0.25,
    correlation_length: float = 1.0,  # lethargy units
) -> np.ndarray:
    """
    Create prior covariance with lethargy-based correlation.
    
    Groups are correlated with correlation decaying exponentially
    with lethargy distance:
    
    ρ_ij = exp(-|u_i - u_j| / λ)
    
    where u = ln(E_max/E) is lethargy and λ is correlation length.
    
    Args:
        prior_flux: Prior flux values
        energy_bounds_ev: Group energy boundaries
        fractional_uncertainty: Base relative uncertainty
        correlation_length: Correlation decay length in lethargy units
        
    Returns:
        Full covariance matrix with lethargy correlation
    """
    n = len(prior_flux)
    
    # Calculate group midpoint lethargies
    E_max = energy_bounds_ev[-1]
    group_energies = np.sqrt(energy_bounds_ev[:-1] * energy_bounds_ev[1:])
    lethargies = np.log(E_max / group_energies)
    
    # Standard deviations
    sigmas = fractional_uncertainty * np.maximum(prior_flux, 1e-30)
    
    # Build correlation matrix
    rho = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            du = abs(lethargies[i] - lethargies[j])
            rho[i, j] = math.exp(-du / correlation_length)
    
    # Convert to covariance
    V = np.outer(sigmas, sigmas) * rho
    
    return V


def create_mcnp_statistics_covariance(
    prior_flux: np.ndarray,
    relative_errors: np.ndarray,
    correlation: float = 0.0,  # Inter-group correlation from tally
) -> np.ndarray:
    """
    Create prior covariance from MCNP tally statistics.
    
    Args:
        prior_flux: Prior flux values from MCNP
        relative_errors: Relative errors from MCNP (1σ)
        correlation: Assumed inter-group correlation
        
    Returns:
        Covariance matrix based on MCNP statistics
    """
    n = len(prior_flux)
    sigmas = relative_errors * np.maximum(prior_flux, 1e-30)
    
    if correlation == 0.0:
        # Diagonal
        return np.diag(sigmas ** 2)
    else:
        # Uniform correlation
        rho = np.full((n, n), correlation)
        np.fill_diagonal(rho, 1.0)
        return np.outer(sigmas, sigmas) * rho


@dataclass
class PriorCovarianceConfig:
    """
    Configuration for prior covariance model.
    
    Attributes:
        model_type: Type of prior covariance model
        fractional_uncertainty: Base relative uncertainty (for applicable models)
        correlation_length: Lethargy correlation length (for correlated model)
        regions: Energy regions (for regional model)
        user_covariance: User-supplied covariance matrix
    """
    
    model_type: PriorCovarianceModel = PriorCovarianceModel.FRACTIONAL_REGIONAL
    fractional_uncertainty: float = 0.25
    correlation_length: float = 1.0
    regions: Optional[List[EnergyRegion]] = None
    user_covariance: Optional[np.ndarray] = None
    
    def build_covariance(
        self,
        prior_flux: np.ndarray,
        energy_bounds_ev: np.ndarray,
        mcnp_errors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build prior covariance matrix based on configuration.
        
        Args:
            prior_flux: Prior flux values
            energy_bounds_ev: Group energy boundaries
            mcnp_errors: Relative errors from MCNP (if available)
            
        Returns:
            Prior covariance matrix V_phi0
        """
        if self.model_type == PriorCovarianceModel.USER_SUPPLIED:
            if self.user_covariance is None:
                raise ValueError("User covariance required for USER_SUPPLIED model")
            return self.user_covariance
        
        elif self.model_type == PriorCovarianceModel.DIAGONAL:
            return create_diagonal_prior_covariance(
                prior_flux, self.fractional_uncertainty
            )
        
        elif self.model_type == PriorCovarianceModel.FRACTIONAL_UNIFORM:
            return create_diagonal_prior_covariance(
                prior_flux, self.fractional_uncertainty
            )
        
        elif self.model_type == PriorCovarianceModel.FRACTIONAL_REGIONAL:
            return create_regional_prior_covariance(
                prior_flux, energy_bounds_ev, self.regions
            )
        
        elif self.model_type == PriorCovarianceModel.LETHARGY_CORRELATED:
            return create_lethargy_correlated_covariance(
                prior_flux, energy_bounds_ev,
                self.fractional_uncertainty, self.correlation_length
            )
        
        elif self.model_type == PriorCovarianceModel.MCNP_STATISTICS:
            if mcnp_errors is None:
                raise ValueError("MCNP errors required for MCNP_STATISTICS model")
            return create_mcnp_statistics_covariance(prior_flux, mcnp_errors)
        
        else:
            raise ValueError(f"Unknown covariance model: {self.model_type}")


@dataclass
class ResponseUncertaintyConfig:
    """
    Configuration for response uncertainty treatment.
    
    Attributes:
        policy: How to handle response uncertainty
        xs_relative_uncertainty: Relative cross section uncertainty (if not from library)
        n_monte_carlo: Number of MC samples (if using MC propagation)
    """
    
    policy: ResponseUncertaintyPolicy = ResponseUncertaintyPolicy.AUGMENT_VY
    xs_relative_uncertainty: float = 0.05  # 5% default
    n_monte_carlo: int = 100


def propagate_response_uncertainty_to_vy(
    R: np.ndarray,
    phi: np.ndarray,
    sigma_R: np.ndarray,
    V_y: np.ndarray,
) -> np.ndarray:
    """
    Augment measurement covariance with propagated response uncertainty.
    
    V_y_augmented = V_y + R' * V_R * R (approximately)
    
    For diagonal response uncertainty:
    V_y_aug[i,i] += sum_g (dR_i/dR_ig * σ_Rig * φ_g)²
    
    Args:
        R: Response matrix [n_reactions × n_groups]
        phi: Prior flux
        sigma_R: Relative uncertainty in R (same shape as R)
        V_y: Original measurement covariance
        
    Returns:
        Augmented covariance matrix
    """
    n_reactions = R.shape[0]
    
    # Propagate response uncertainty through y = R @ phi
    # δy_i = sum_g (R_ig * δphi_g + phi_g * δR_ig)
    # For response uncertainty only: δy_i ≈ sum_g (phi_g * σ_Rig * R_ig)
    
    V_aug = V_y.copy()
    
    for i in range(n_reactions):
        var_from_R = 0.0
        for g in range(len(phi)):
            delta_y = phi[g] * sigma_R[i, g] * R[i, g]
            var_from_R += delta_y ** 2
        V_aug[i, i] += var_from_R
    
    return V_aug


def validate_covariance_matrix(
    V: np.ndarray,
    name: str = "covariance",
    fix_issues: bool = True,
    nugget: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, bool]]:
    """
    Validate and optionally fix covariance matrix issues.
    
    Checks:
    1. Symmetry
    2. Positive definiteness
    3. Near-singularity (condition number)
    
    Args:
        V: Covariance matrix to validate
        name: Name for error messages
        fix_issues: Whether to apply fixes
        nugget: Small value to add to diagonal for conditioning
        
    Returns:
        Tuple of (fixed_matrix, issues_dict)
    """
    issues = {
        "asymmetric": False,
        "not_positive_definite": False,
        "near_singular": False,
        "fixed": False,
    }
    
    V_fixed = V.copy()
    
    # Check symmetry
    if not np.allclose(V, V.T, rtol=1e-10):
        issues["asymmetric"] = True
        if fix_issues:
            V_fixed = 0.5 * (V_fixed + V_fixed.T)
            issues["fixed"] = True
    
    # Check positive definiteness
    try:
        eigenvalues = np.linalg.eigvalsh(V_fixed)
        if np.any(eigenvalues <= 0):
            issues["not_positive_definite"] = True
            if fix_issues:
                # Add nugget to diagonal
                V_fixed += nugget * np.eye(len(V)) * np.max(np.abs(np.diag(V)))
                issues["fixed"] = True
    except np.linalg.LinAlgError:
        issues["not_positive_definite"] = True
        if fix_issues:
            V_fixed += nugget * np.eye(len(V)) * np.max(np.abs(np.diag(V)))
            issues["fixed"] = True
    
    # Check condition number
    try:
        cond = np.linalg.cond(V_fixed)
        if cond > 1e12:
            issues["near_singular"] = True
            if fix_issues:
                # SVD-based conditioning
                U, s, Vh = np.linalg.svd(V_fixed)
                s_min = np.max(s) / 1e10  # Condition limit of 10^10
                s_fixed = np.maximum(s, s_min)
                V_fixed = U @ np.diag(s_fixed) @ Vh
                issues["fixed"] = True
    except np.linalg.LinAlgError:
        issues["near_singular"] = True
    
    return V_fixed, issues


@dataclass
class CovarianceValidationResult:
    """Result of covariance validation."""
    
    is_valid: bool
    issues: Dict[str, bool]
    condition_number: float
    min_eigenvalue: float
    was_fixed: bool
    fixed_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "condition_number": self.condition_number,
            "min_eigenvalue": self.min_eigenvalue,
            "was_fixed": self.was_fixed,
        }


def full_covariance_validation(
    V: np.ndarray,
    name: str = "covariance",
    fix: bool = True,
) -> CovarianceValidationResult:
    """
    Perform full covariance validation with detailed results.
    
    Args:
        V: Covariance matrix
        name: Matrix name for reporting
        fix: Whether to attempt fixes
        
    Returns:
        CovarianceValidationResult
    """
    V_fixed, issues = validate_covariance_matrix(V, name, fix)
    
    # Compute diagnostics
    try:
        eigenvalues = np.linalg.eigvalsh(V_fixed)
        min_eig = float(np.min(eigenvalues))
        cond = float(np.linalg.cond(V_fixed))
    except:
        min_eig = 0.0
        cond = float('inf')
    
    is_valid = not any([
        issues["asymmetric"],
        issues["not_positive_definite"],
        issues["near_singular"],
    ])
    
    return CovarianceValidationResult(
        is_valid=is_valid,
        issues=issues,
        condition_number=cond,
        min_eigenvalue=min_eig,
        was_fixed=issues.get("fixed", False),
        fixed_matrix=V_fixed if issues.get("fixed", False) else None,
    )


def summarize_prior_uncertainty_by_region(
    V: np.ndarray,
    prior_flux: np.ndarray,
    energy_bounds_ev: np.ndarray,
    regions: Optional[List[EnergyRegion]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Summarize prior uncertainty magnitudes by energy region.
    
    Required for reports per STAYSL parity requirements.
    
    Args:
        V: Prior covariance matrix
        prior_flux: Prior flux values
        energy_bounds_ev: Group boundaries
        regions: Energy regions to summarize
        
    Returns:
        Dictionary with regional uncertainty summaries
    """
    if regions is None:
        regions = STANDARD_REGIONS
    
    n = len(prior_flux)
    group_energies = np.sqrt(energy_bounds_ev[:-1] * energy_bounds_ev[1:])
    sigmas = np.sqrt(np.diag(V))
    
    summary = {}
    
    for region in regions:
        # Find groups in this region
        mask = (group_energies >= region.energy_low_ev) & (group_energies < region.energy_high_ev)
        
        if np.any(mask):
            region_flux = prior_flux[mask]
            region_sigma = sigmas[mask]
            
            # Relative uncertainties
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_unc = np.where(region_flux > 0, region_sigma / region_flux, 0)
            
            summary[region.name] = {
                "n_groups": int(np.sum(mask)),
                "mean_rel_uncertainty": float(np.mean(rel_unc)),
                "min_rel_uncertainty": float(np.min(rel_unc)),
                "max_rel_uncertainty": float(np.max(rel_unc)),
                "total_flux": float(np.sum(region_flux)),
                "rms_uncertainty": float(np.sqrt(np.sum(region_sigma**2))),
            }
    
    return summary
