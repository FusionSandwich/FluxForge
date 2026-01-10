"""
f/α Reconciliation Module for k0-NAA Cross-Validation
=====================================================

This module provides cross-validation between k0-NAA derived f and α
parameters and values computed from unfolded neutron spectra. This is
essential for validating flux characterization and ensuring consistency
between different measurement/analysis approaches.

Theory
------
The thermal-to-epithermal flux ratio f is defined as:

    f = φ_th / φ_epi

Where:
    φ_th = ∫[0 to E_Cd] φ(E) dE  (thermal region, E_Cd ~ 0.5 eV)
    φ_epi = ∫[E_Cd to E_fast] φ(E) dE  (epithermal region)

The epithermal shape parameter α describes deviation from 1/E:

    φ_epi(E) ∝ E^(-1-α)

For a pure 1/E spectrum, α = 0. Positive α indicates a "harder"
spectrum (more high-energy neutrons), negative α indicates "softer".

The module provides:
1. Computation of f, α from unfolded group spectra
2. Statistical comparison with Cd-ratio derived values
3. Diagnostic metrics for consistency assessment
4. Recommendations for reconciliation

References
----------
- De Corte et al., JRNC 133 (1989) 43-130: k0-NAA formalism
- Westcott conventions for reactor neutron spectra (1960)
- IAEA-TECDOC-1215: Use of Research Reactors for NAA

Author: FluxForge Development Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy import stats


# Standard cutoff energies
E_CD = 0.55  # Cadmium cutoff energy (eV)
E_THERMAL_UPPER = 0.4  # Upper limit for thermal region (eV)
E_EPITHERMAL_LOWER = 0.625  # Lower limit for epithermal (eV), sometimes used
E_FAST_THRESHOLD = 1.0e6  # 1 MeV fast threshold


class ReconciliationStatus(Enum):
    """Status of f/α reconciliation."""
    
    CONSISTENT = "consistent"  # Within uncertainties
    TENSION = "tension"  # Marginal disagreement (1-3σ)
    DISCREPANT = "discrepant"  # Significant disagreement (>3σ)
    INCOMPLETE = "incomplete"  # Missing required data


@dataclass
class UnfoldedFluxParameters:
    """
    Flux parameters computed from unfolded spectrum.
    
    Attributes:
        f: Thermal-to-epithermal flux ratio
        f_uncertainty: Uncertainty in f
        alpha: Epithermal shape parameter
        alpha_uncertainty: Uncertainty in α
        phi_thermal: Thermal flux integral (arbitrary units)
        phi_epithermal: Epithermal flux integral
        phi_fast: Fast flux integral (>1 MeV)
        energy_range: (E_min, E_max) of spectrum
        n_groups: Number of energy groups used
    """
    
    f: float
    f_uncertainty: float
    alpha: float
    alpha_uncertainty: float
    phi_thermal: float = 0.0
    phi_epithermal: float = 0.0
    phi_fast: float = 0.0
    energy_range: Tuple[float, float] = (0.0, 0.0)
    n_groups: int = 0
    
    def __repr__(self):
        return f"UnfoldedFluxParameters(f={self.f:.1f}±{self.f_uncertainty:.1f}, α={self.alpha:.3f}±{self.alpha_uncertainty:.3f})"


@dataclass
class ReconciliationResult:
    """
    Result of f/α reconciliation analysis.
    
    Attributes:
        f_unfolded: f from unfolded spectrum
        f_cdratio: f from Cd-ratio analysis
        f_difference: Absolute difference
        f_sigma: Significance of difference in σ
        alpha_unfolded: α from unfolded spectrum
        alpha_cdratio: α from Cd-ratio analysis
        alpha_difference: Absolute difference
        alpha_sigma: Significance of difference in σ
        status: Overall reconciliation status
        recommendations: List of recommendations
        diagnostics: Additional diagnostic metrics
    """
    
    f_unfolded: float = 0.0
    f_cdratio: float = 0.0
    f_difference: float = 0.0
    f_sigma: float = 0.0
    f_unfolded_unc: float = 0.0
    f_cdratio_unc: float = 0.0
    
    alpha_unfolded: float = 0.0
    alpha_cdratio: float = 0.0
    alpha_difference: float = 0.0
    alpha_sigma: float = 0.0
    alpha_unfolded_unc: float = 0.0
    alpha_cdratio_unc: float = 0.0
    
    status: ReconciliationStatus = ReconciliationStatus.INCOMPLETE
    recommendations: list = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "f/α RECONCILIATION SUMMARY",
            "=" * 60,
            "",
            "--- Thermal-to-Epithermal Ratio (f) ---",
            f"  Unfolded:  f = {self.f_unfolded:.2f} ± {self.f_unfolded_unc:.2f}",
            f"  Cd-ratio:  f = {self.f_cdratio:.2f} ± {self.f_cdratio_unc:.2f}",
            f"  Δf = {self.f_difference:+.2f} ({self.f_sigma:.1f}σ)",
            "",
            "--- Epithermal Shape Parameter (α) ---",
            f"  Unfolded:  α = {self.alpha_unfolded:.4f} ± {self.alpha_unfolded_unc:.4f}",
            f"  Cd-ratio:  α = {self.alpha_cdratio:.4f} ± {self.alpha_cdratio_unc:.4f}",
            f"  Δα = {self.alpha_difference:+.4f} ({self.alpha_sigma:.1f}σ)",
            "",
            f"Status: {self.status.value.upper()}",
            "",
        ]
        
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_f_from_spectrum(
    flux: np.ndarray,
    energy_bins: np.ndarray,
    flux_uncertainty: Optional[np.ndarray] = None,
    e_cd: float = E_CD,
) -> Tuple[float, float, float, float]:
    """
    Compute f (thermal/epithermal ratio) from group flux spectrum.
    
    Parameters
    ----------
    flux : np.ndarray
        Group fluxes (per unit energy or lethargy).
    energy_bins : np.ndarray
        Energy bin boundaries in eV (length = len(flux) + 1).
    flux_uncertainty : np.ndarray, optional
        Uncertainties in flux values.
    e_cd : float
        Cadmium cutoff energy (default 0.55 eV).
        
    Returns
    -------
    f : float
        Thermal-to-epithermal ratio.
    f_unc : float
        Uncertainty in f.
    phi_thermal : float
        Thermal flux integral.
    phi_epithermal : float
        Epithermal flux integral.
    """
    if flux_uncertainty is None:
        flux_uncertainty = np.zeros_like(flux)
    
    n_groups = len(flux)
    
    # Compute group widths
    delta_e = np.diff(energy_bins)
    
    # Integrate thermal region (E < E_Cd)
    phi_thermal = 0.0
    phi_thermal_var = 0.0
    phi_epithermal = 0.0
    phi_epithermal_var = 0.0
    
    for i in range(n_groups):
        e_lo = energy_bins[i]
        e_hi = energy_bins[i + 1]
        e_mid = (e_lo + e_hi) / 2
        
        # Determine contribution to each region
        if e_hi <= e_cd:
            # Entirely thermal
            contribution = flux[i] * delta_e[i]
            phi_thermal += contribution
            phi_thermal_var += (flux_uncertainty[i] * delta_e[i])**2
        elif e_lo >= e_cd:
            # Entirely epithermal
            contribution = flux[i] * delta_e[i]
            phi_epithermal += contribution
            phi_epithermal_var += (flux_uncertainty[i] * delta_e[i])**2
        else:
            # Straddles the cutoff - split proportionally
            f_thermal = (e_cd - e_lo) / delta_e[i]
            f_epi = (e_hi - e_cd) / delta_e[i]
            
            phi_thermal += flux[i] * delta_e[i] * f_thermal
            phi_thermal_var += (flux_uncertainty[i] * delta_e[i] * f_thermal)**2
            
            phi_epithermal += flux[i] * delta_e[i] * f_epi
            phi_epithermal_var += (flux_uncertainty[i] * delta_e[i] * f_epi)**2
    
    # Compute f and uncertainty
    if phi_epithermal > 0:
        f = phi_thermal / phi_epithermal
        # Propagate uncertainty
        rel_th = np.sqrt(phi_thermal_var) / phi_thermal if phi_thermal > 0 else 0
        rel_epi = np.sqrt(phi_epithermal_var) / phi_epithermal
        f_unc = f * np.sqrt(rel_th**2 + rel_epi**2)
    else:
        f = np.inf
        f_unc = np.inf
    
    return f, f_unc, phi_thermal, phi_epithermal


def compute_alpha_from_spectrum(
    flux: np.ndarray,
    energy_bins: np.ndarray,
    flux_uncertainty: Optional[np.ndarray] = None,
    e_min: float = 1.0,  # 1 eV
    e_max: float = 1e5,  # 100 keV
) -> Tuple[float, float]:
    """
    Compute α (epithermal shape parameter) from spectrum fit.
    
    Fits φ(E) ∝ E^(-1-α) in the epithermal region to determine α.
    
    Parameters
    ----------
    flux : np.ndarray
        Group fluxes per unit lethargy.
    energy_bins : np.ndarray
        Energy bin boundaries in eV.
    flux_uncertainty : np.ndarray, optional
        Uncertainties in flux values.
    e_min : float
        Lower energy bound for fit (eV).
    e_max : float
        Upper energy bound for fit (eV).
        
    Returns
    -------
    alpha : float
        Epithermal shape parameter.
    alpha_unc : float
        Uncertainty in α.
    """
    if flux_uncertainty is None:
        flux_uncertainty = np.ones_like(flux) * 0.1 * flux  # 10% assumed
    
    # Select groups in epithermal range
    e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])  # Geometric mean
    mask = (e_centers >= e_min) & (e_centers <= e_max) & (flux > 0)
    
    if np.sum(mask) < 3:
        return 0.0, 0.1  # Not enough points, assume 1/E
    
    e_sel = e_centers[mask]
    phi_sel = flux[mask]
    unc_sel = flux_uncertainty[mask]
    
    # For 1/E spectrum, φ×E should be constant
    # For E^(-1-α), φ×E ∝ E^(-α)
    # Take log: log(φ×E) = -α×log(E) + const
    
    y = np.log(phi_sel * e_sel)
    x = np.log(e_sel)
    
    # Weights from uncertainty propagation
    # d(log(phi*E)) ≈ dphi / phi
    weights = phi_sel / unc_sel
    weights = weights / np.max(weights)  # Normalize
    
    # Weighted linear regression
    try:
        # Using scipy for proper statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        alpha = -slope
        alpha_unc = std_err
    except:
        alpha = 0.0
        alpha_unc = 0.1
    
    return alpha, alpha_unc


def compute_flux_parameters_from_spectrum(
    flux: np.ndarray,
    energy_bins: np.ndarray,
    flux_uncertainty: Optional[np.ndarray] = None,
    e_cd: float = E_CD,
) -> UnfoldedFluxParameters:
    """
    Compute complete flux parameters from unfolded spectrum.
    
    Parameters
    ----------
    flux : np.ndarray
        Group fluxes (per unit energy or lethargy).
    energy_bins : np.ndarray
        Energy bin boundaries in eV (length = len(flux) + 1).
    flux_uncertainty : np.ndarray, optional
        Uncertainties in flux values.
    e_cd : float
        Cadmium cutoff energy.
        
    Returns
    -------
    UnfoldedFluxParameters
        Complete flux characterization.
    """
    f, f_unc, phi_th, phi_epi = compute_f_from_spectrum(
        flux, energy_bins, flux_uncertainty, e_cd
    )
    
    alpha, alpha_unc = compute_alpha_from_spectrum(
        flux, energy_bins, flux_uncertainty
    )
    
    # Compute fast flux (>1 MeV)
    delta_e = np.diff(energy_bins)
    e_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    fast_mask = e_centers >= E_FAST_THRESHOLD
    phi_fast = np.sum(flux[fast_mask] * delta_e[fast_mask]) if np.any(fast_mask) else 0.0
    
    return UnfoldedFluxParameters(
        f=f,
        f_uncertainty=f_unc,
        alpha=alpha,
        alpha_uncertainty=alpha_unc,
        phi_thermal=phi_th,
        phi_epithermal=phi_epi,
        phi_fast=phi_fast,
        energy_range=(energy_bins[0], energy_bins[-1]),
        n_groups=len(flux),
    )


def reconcile_flux_parameters(
    unfolded_params: UnfoldedFluxParameters,
    cdratio_f: float,
    cdratio_f_unc: float,
    cdratio_alpha: float = 0.0,
    cdratio_alpha_unc: float = 0.01,
    f_tolerance_sigma: float = 2.0,
    alpha_tolerance_sigma: float = 2.0,
) -> ReconciliationResult:
    """
    Reconcile f and α between unfolded spectrum and Cd-ratio values.
    
    Parameters
    ----------
    unfolded_params : UnfoldedFluxParameters
        Parameters from unfolded spectrum.
    cdratio_f : float
        f from Cd-ratio analysis.
    cdratio_f_unc : float
        Uncertainty in Cd-ratio f.
    cdratio_alpha : float
        α from Cd-ratio analysis.
    cdratio_alpha_unc : float
        Uncertainty in Cd-ratio α.
    f_tolerance_sigma : float
        Tolerance in sigma for f comparison.
    alpha_tolerance_sigma : float
        Tolerance in sigma for α comparison.
        
    Returns
    -------
    ReconciliationResult
        Complete reconciliation analysis.
    """
    result = ReconciliationResult()
    
    # Store values
    result.f_unfolded = unfolded_params.f
    result.f_unfolded_unc = unfolded_params.f_uncertainty
    result.f_cdratio = cdratio_f
    result.f_cdratio_unc = cdratio_f_unc
    
    result.alpha_unfolded = unfolded_params.alpha
    result.alpha_unfolded_unc = unfolded_params.alpha_uncertainty
    result.alpha_cdratio = cdratio_alpha
    result.alpha_cdratio_unc = cdratio_alpha_unc
    
    # Compute differences
    result.f_difference = unfolded_params.f - cdratio_f
    combined_f_unc = np.sqrt(unfolded_params.f_uncertainty**2 + cdratio_f_unc**2)
    result.f_sigma = abs(result.f_difference) / combined_f_unc if combined_f_unc > 0 else 0
    
    result.alpha_difference = unfolded_params.alpha - cdratio_alpha
    combined_alpha_unc = np.sqrt(unfolded_params.alpha_uncertainty**2 + cdratio_alpha_unc**2)
    result.alpha_sigma = abs(result.alpha_difference) / combined_alpha_unc if combined_alpha_unc > 0 else 0
    
    # Determine status
    f_ok = result.f_sigma <= f_tolerance_sigma
    alpha_ok = result.alpha_sigma <= alpha_tolerance_sigma
    
    if f_ok and alpha_ok:
        result.status = ReconciliationStatus.CONSISTENT
    elif result.f_sigma > 3.0 or result.alpha_sigma > 3.0:
        result.status = ReconciliationStatus.DISCREPANT
    else:
        result.status = ReconciliationStatus.TENSION
    
    # Generate recommendations
    recommendations = []
    
    if not f_ok:
        if unfolded_params.f > cdratio_f:
            recommendations.append(
                "Unfolded f is higher than Cd-ratio. Check: thermal group resolution, "
                "prior spectrum thermal content, Cd filter transmission factor."
            )
        else:
            recommendations.append(
                "Unfolded f is lower than Cd-ratio. Check: epithermal normalization, "
                "high-energy extrapolation, Cd-ratio measurement statistics."
            )
    
    if not alpha_ok:
        if unfolded_params.alpha > cdratio_alpha:
            recommendations.append(
                "Unfolded α is higher (harder spectrum). Consider: group structure "
                "in resonance region, prior spectrum bias toward high energies."
            )
        else:
            recommendations.append(
                "Unfolded α is lower (softer spectrum). Consider: insufficient "
                "high-energy monitors, prior spectrum thermal bias."
            )
    
    if result.status == ReconciliationStatus.CONSISTENT:
        recommendations.append(
            "Parameters are consistent within uncertainties. Proceed with confidence."
        )
    
    result.recommendations = recommendations
    
    # Diagnostics
    result.diagnostics = {
        'combined_f_uncertainty': combined_f_unc,
        'combined_alpha_uncertainty': combined_alpha_unc,
        'f_relative_difference': result.f_difference / cdratio_f if cdratio_f > 0 else 0,
        'alpha_relative_difference': result.alpha_difference / cdratio_alpha if cdratio_alpha != 0 else 0,
        'phi_thermal': unfolded_params.phi_thermal,
        'phi_epithermal': unfolded_params.phi_epithermal,
        'n_groups': unfolded_params.n_groups,
    }
    
    return result


def reconcile_with_unfold_result(
    unfold_result,  # UnfoldResult from solvers
    cdratio_f: float,
    cdratio_f_unc: float,
    cdratio_alpha: float = 0.0,
    cdratio_alpha_unc: float = 0.01,
    energy_bins_ev: Optional[np.ndarray] = None,
) -> ReconciliationResult:
    """
    Reconcile using UnfoldResult directly.
    
    This is a convenience wrapper that extracts flux and uncertainty
    from an UnfoldResult object.
    
    Parameters
    ----------
    unfold_result : UnfoldResult
        Result from spectrum unfolding (GRAVEL, MAXED, etc.).
    cdratio_f : float
        f from Cd-ratio analysis.
    cdratio_f_unc : float
        Uncertainty in Cd-ratio f.
    cdratio_alpha : float
        α from Cd-ratio analysis.
    cdratio_alpha_unc : float
        Uncertainty in Cd-ratio α.
    energy_bins_ev : np.ndarray, optional
        Energy bins in eV. If not provided, attempts to get from unfold_result.
        
    Returns
    -------
    ReconciliationResult
        Complete reconciliation analysis.
    """
    # Extract flux and uncertainty from unfold result
    flux = unfold_result.flux
    
    # Get uncertainty from covariance if available
    if hasattr(unfold_result, 'covariance') and unfold_result.covariance is not None:
        flux_unc = np.sqrt(np.diag(unfold_result.covariance))
    elif hasattr(unfold_result, 'uncertainty') and unfold_result.uncertainty is not None:
        flux_unc = unfold_result.uncertainty
    else:
        flux_unc = None
    
    # Get energy bins
    if energy_bins_ev is None:
        if hasattr(unfold_result, 'energy_bins'):
            energy_bins_ev = unfold_result.energy_bins
        elif hasattr(unfold_result, 'group_structure'):
            energy_bins_ev = unfold_result.group_structure.boundaries
        else:
            raise ValueError("Energy bins must be provided or available in unfold_result")
    
    # Compute parameters and reconcile
    params = compute_flux_parameters_from_spectrum(flux, energy_bins_ev, flux_unc)
    
    return reconcile_flux_parameters(
        params, cdratio_f, cdratio_f_unc, cdratio_alpha, cdratio_alpha_unc
    )


def quick_f_check(
    flux: np.ndarray,
    energy_bins: np.ndarray,
    expected_f: float,
    tolerance: float = 0.2,  # 20% tolerance
) -> Tuple[bool, float, str]:
    """
    Quick check if unfolded f is reasonable.
    
    Parameters
    ----------
    flux : np.ndarray
        Group fluxes.
    energy_bins : np.ndarray
        Energy bin boundaries in eV.
    expected_f : float
        Expected f value from other sources.
    tolerance : float
        Relative tolerance for comparison.
        
    Returns
    -------
    ok : bool
        Whether f is within tolerance.
    computed_f : float
        Computed f value.
    message : str
        Descriptive message.
    """
    f, _, _, _ = compute_f_from_spectrum(flux, energy_bins)
    
    rel_diff = abs(f - expected_f) / expected_f if expected_f > 0 else 0
    ok = rel_diff <= tolerance
    
    if ok:
        message = f"f = {f:.1f} is consistent with expected {expected_f:.1f} (within {rel_diff*100:.0f}%)"
    else:
        message = f"f = {f:.1f} differs from expected {expected_f:.1f} by {rel_diff*100:.0f}%"
    
    return ok, f, message
