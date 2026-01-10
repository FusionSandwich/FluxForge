"""Transport code comparison for model validation (Criterion 6).

This module provides tools for comparing FluxForge unfolded spectra
and activation results against OpenMC and MCNP transport calculations.

Comparison Metrics:
- C/E ratio (Calculated-to-Experimental)
- Chi-squared goodness of fit
- Parity plots
- Residual analysis
- Energy-group-wise comparison

References:
    PNNL-22253 STAYSL PNNL User Manual
    ASTM E944 Standard Guide for Application of Neutron Spectrum
        Adjustment Methods in Reactor Surveillance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Any
import numpy as np


class TransportCode(Enum):
    """Supported transport codes."""
    
    OPENMC = "openmc"
    MCNP = "mcnp"
    ALARA = "alara"
    SERPENT = "serpent"
    SCALE = "scale"
    OTHER = "other"


@dataclass
class SpectrumComparison:
    """Result of spectrum-to-spectrum comparison.
    
    Attributes
    ----------
    energy_grid : ndarray
        Common energy grid (bin boundaries)
    unfolded : ndarray
        Unfolded spectrum (per lethargy or per MeV)
    transport : ndarray
        Transport calculation spectrum
    unfolded_unc : ndarray
        Unfolded spectrum uncertainties
    transport_unc : ndarray
        Transport spectrum uncertainties
    c_over_e : ndarray
        C/E ratio per energy group
    c_over_e_unc : ndarray
        C/E uncertainty
    chi2 : float
        Chi-squared statistic
    chi2_reduced : float
        Reduced chi-squared
    n_groups : int
        Number of energy groups compared
    transport_code : TransportCode
        Source of transport calculation
    """
    
    energy_grid: np.ndarray
    unfolded: np.ndarray
    transport: np.ndarray
    unfolded_unc: np.ndarray = field(default_factory=lambda: np.array([]))
    transport_unc: np.ndarray = field(default_factory=lambda: np.array([]))
    c_over_e: np.ndarray = field(default_factory=lambda: np.array([]))
    c_over_e_unc: np.ndarray = field(default_factory=lambda: np.array([]))
    chi2: float = 0.0
    chi2_reduced: float = 0.0
    n_groups: int = 0
    transport_code: TransportCode = TransportCode.OTHER
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.n_groups = len(self.unfolded)
        
        if len(self.c_over_e) == 0:
            self._compute_c_over_e()
        
        if self.chi2 == 0.0:
            self._compute_chi2()
    
    def _compute_c_over_e(self):
        """Compute C/E ratios."""
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            self.c_over_e = np.where(
                self.unfolded > 0,
                self.transport / self.unfolded,
                np.nan,
            )
        
        # Propagate uncertainties
        if len(self.unfolded_unc) > 0 and len(self.transport_unc) > 0:
            # Relative uncertainty of ratio
            rel_unc_e = np.where(self.unfolded > 0, self.unfolded_unc / self.unfolded, 0)
            rel_unc_c = np.where(self.transport > 0, self.transport_unc / self.transport, 0)
            rel_unc_ce = np.sqrt(rel_unc_e**2 + rel_unc_c**2)
            self.c_over_e_unc = np.where(np.isfinite(self.c_over_e), self.c_over_e * rel_unc_ce, np.nan)
        else:
            self.c_over_e_unc = np.full_like(self.c_over_e, np.nan)
    
    def _compute_chi2(self):
        """Compute chi-squared statistic."""
        if len(self.unfolded_unc) == 0:
            return
        
        # Combined variance
        var = self.unfolded_unc**2
        if len(self.transport_unc) > 0:
            var = var + self.transport_unc**2
        
        # Chi-squared
        valid = (var > 0) & np.isfinite(self.unfolded) & np.isfinite(self.transport)
        if np.sum(valid) > 0:
            residuals = (self.transport - self.unfolded)[valid]
            weights = 1.0 / var[valid]
            self.chi2 = float(np.sum(residuals**2 * weights))
            self.chi2_reduced = self.chi2 / np.sum(valid) if np.sum(valid) > 1 else 0.0
    
    @property
    def mean_c_over_e(self) -> float:
        """Mean C/E ratio (finite values only)."""
        valid = np.isfinite(self.c_over_e)
        if np.sum(valid) == 0:
            return np.nan
        return float(np.mean(self.c_over_e[valid]))
    
    @property
    def std_c_over_e(self) -> float:
        """Standard deviation of C/E ratios."""
        valid = np.isfinite(self.c_over_e)
        if np.sum(valid) < 2:
            return np.nan
        return float(np.std(self.c_over_e[valid]))
    
    @property
    def max_deviation(self) -> tuple[float, int]:
        """Maximum deviation and its group index."""
        valid = np.isfinite(self.c_over_e)
        if np.sum(valid) == 0:
            return np.nan, -1
        
        deviations = np.abs(self.c_over_e - 1.0)
        deviations = np.where(valid, deviations, 0)
        idx = int(np.argmax(deviations))
        return float(deviations[idx]), idx
    
    @property
    def within_uncertainty(self) -> np.ndarray:
        """Boolean array: is C/E within 1σ of unity?"""
        if len(self.c_over_e_unc) == 0:
            return np.full(len(self.c_over_e), False)
        
        return np.abs(self.c_over_e - 1.0) <= self.c_over_e_unc
    
    def summary(self) -> str:
        """Text summary of comparison."""
        lines = [
            f"Spectrum Comparison ({self.transport_code.value})",
            f"  Groups: {self.n_groups}",
            f"  Mean C/E: {self.mean_c_over_e:.4f} ± {self.std_c_over_e:.4f}",
            f"  Max deviation: {self.max_deviation[0]:.4f} at group {self.max_deviation[1]}",
            f"  χ²: {self.chi2:.2f} (reduced: {self.chi2_reduced:.2f})",
            f"  Groups within 1σ: {np.sum(self.within_uncertainty)}/{self.n_groups}",
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "transport_code": self.transport_code.value,
            "n_groups": self.n_groups,
            "mean_c_over_e": self.mean_c_over_e,
            "std_c_over_e": self.std_c_over_e,
            "chi2": self.chi2,
            "chi2_reduced": self.chi2_reduced,
            "max_deviation": self.max_deviation[0],
            "max_deviation_group": self.max_deviation[1],
            "fraction_within_1sigma": float(np.sum(self.within_uncertainty)) / self.n_groups if self.n_groups > 0 else 0,
        }


@dataclass
class ReactionRateComparison:
    """Comparison of reaction rates (measured vs. calculated).
    
    Attributes
    ----------
    reactions : list[str]
        Reaction identifiers (e.g., "Au-197(n,g)")
    measured : ndarray
        Measured reaction rates
    measured_unc : ndarray
        Measurement uncertainties
    calculated : ndarray
        Calculated reaction rates
    calculated_unc : ndarray
        Calculation uncertainties
    c_over_e : ndarray
        C/E ratios
    c_over_e_unc : ndarray
        C/E uncertainties
    """
    
    reactions: list[str]
    measured: np.ndarray
    measured_unc: np.ndarray
    calculated: np.ndarray
    calculated_unc: np.ndarray = field(default_factory=lambda: np.array([]))
    c_over_e: np.ndarray = field(default_factory=lambda: np.array([]))
    c_over_e_unc: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Compute C/E if not provided."""
        if len(self.c_over_e) == 0 and len(self.measured) > 0:
            self._compute_c_over_e()
    
    def _compute_c_over_e(self):
        """Compute C/E ratios with uncertainties."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.c_over_e = np.where(
                self.measured > 0,
                self.calculated / self.measured,
                np.nan,
            )
        
        # Uncertainty propagation
        rel_e = np.where(self.measured > 0, self.measured_unc / self.measured, 0)
        if len(self.calculated_unc) > 0:
            rel_c = np.where(self.calculated > 0, self.calculated_unc / self.calculated, 0)
        else:
            rel_c = np.zeros_like(rel_e)
        
        rel_ce = np.sqrt(rel_e**2 + rel_c**2)
        self.c_over_e_unc = np.where(np.isfinite(self.c_over_e), self.c_over_e * rel_ce, np.nan)
    
    @property
    def chi2(self) -> float:
        """Chi-squared goodness of fit."""
        var = self.measured_unc**2
        if len(self.calculated_unc) > 0:
            var = var + self.calculated_unc**2
        
        valid = (var > 0) & np.isfinite(self.measured) & np.isfinite(self.calculated)
        if np.sum(valid) == 0:
            return np.nan
        
        residuals = (self.calculated - self.measured)[valid]
        weights = 1.0 / var[valid]
        return float(np.sum(residuals**2 * weights))
    
    @property
    def mean_c_over_e(self) -> float:
        """Mean C/E ratio."""
        valid = np.isfinite(self.c_over_e)
        if np.sum(valid) == 0:
            return np.nan
        return float(np.mean(self.c_over_e[valid]))
    
    def summary_table(self) -> str:
        """Generate summary table."""
        lines = [
            "Reaction Rate Comparison",
            "-" * 70,
            f"{'Reaction':<20} {'Measured':>12} {'Calculated':>12} {'C/E':>8} {'Status':>10}",
            "-" * 70,
        ]
        
        for i, rxn in enumerate(self.reactions):
            status = "OK" if abs(self.c_over_e[i] - 1.0) <= self.c_over_e_unc[i] else "CHECK"
            lines.append(
                f"{rxn:<20} {self.measured[i]:>12.4e} {self.calculated[i]:>12.4e} "
                f"{self.c_over_e[i]:>8.3f} {status:>10}"
            )
        
        lines.append("-" * 70)
        lines.append(f"Mean C/E: {self.mean_c_over_e:.4f}, χ²: {self.chi2:.2f}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "reactions": self.reactions,
            "mean_c_over_e": self.mean_c_over_e,
            "chi2": self.chi2,
            "n_reactions": len(self.reactions),
            "data": [
                {
                    "reaction": rxn,
                    "measured": float(self.measured[i]),
                    "measured_unc": float(self.measured_unc[i]),
                    "calculated": float(self.calculated[i]),
                    "c_over_e": float(self.c_over_e[i]),
                    "c_over_e_unc": float(self.c_over_e_unc[i]),
                }
                for i, rxn in enumerate(self.reactions)
            ],
        }


def rebin_spectrum(
    spectrum: np.ndarray,
    energy_old: np.ndarray,
    energy_new: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    conserve_integral: bool = True,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Rebin spectrum to new energy grid.
    
    Parameters
    ----------
    spectrum : ndarray
        Flux values (n_groups)
    energy_old : ndarray
        Original energy bin boundaries (n_groups + 1)
    energy_new : ndarray
        New energy bin boundaries (n_new + 1)
    uncertainty : ndarray, optional
        Spectrum uncertainties
    conserve_integral : bool
        If True, conserve integral flux; if False, interpolate
        
    Returns
    -------
    tuple[ndarray, ndarray | None]
        Rebinned spectrum and uncertainty
    """
    n_old = len(spectrum)
    n_new = len(energy_new) - 1
    
    if len(energy_old) != n_old + 1:
        raise ValueError("energy_old must have n_groups + 1 boundaries")
    
    result = np.zeros(n_new)
    result_unc = np.zeros(n_new) if uncertainty is not None else None
    
    for i in range(n_new):
        E_lo_new = energy_new[i]
        E_hi_new = energy_new[i + 1]
        width_new = E_hi_new - E_lo_new
        
        # Find overlapping old bins
        total_flux = 0.0
        total_var = 0.0
        
        for j in range(n_old):
            E_lo_old = energy_old[j]
            E_hi_old = energy_old[j + 1]
            
            # Overlap range
            E_lo_overlap = max(E_lo_new, E_lo_old)
            E_hi_overlap = min(E_hi_new, E_hi_old)
            
            if E_hi_overlap > E_lo_overlap:
                # Fraction of old bin that overlaps
                width_old = E_hi_old - E_lo_old
                overlap_fraction = (E_hi_overlap - E_lo_overlap) / width_old
                
                if conserve_integral:
                    # Add flux contribution
                    total_flux += spectrum[j] * overlap_fraction * width_old / width_new
                else:
                    total_flux += spectrum[j] * overlap_fraction
                
                if uncertainty is not None:
                    total_var += (uncertainty[j] * overlap_fraction)**2
        
        result[i] = total_flux
        if result_unc is not None:
            result_unc[i] = np.sqrt(total_var)
    
    return result, result_unc


def compare_flux_spectrum(
    unfolded: np.ndarray,
    unfolded_unc: np.ndarray,
    unfolded_energy: np.ndarray,
    transport: np.ndarray,
    transport_unc: Optional[np.ndarray],
    transport_energy: np.ndarray,
    transport_code: TransportCode = TransportCode.OTHER,
    common_grid: Optional[np.ndarray] = None,
) -> SpectrumComparison:
    """Compare unfolded spectrum to transport calculation.
    
    Parameters
    ----------
    unfolded : ndarray
        Unfolded spectrum (from FluxForge)
    unfolded_unc : ndarray
        Unfolded spectrum uncertainties
    unfolded_energy : ndarray
        Unfolded energy bin boundaries
    transport : ndarray
        Transport calculation spectrum
    transport_unc : ndarray, optional
        Transport spectrum uncertainties
    transport_energy : ndarray
        Transport energy bin boundaries
    transport_code : TransportCode
        Source of transport calculation
    common_grid : ndarray, optional
        Common energy grid for comparison (default: use unfolded grid)
        
    Returns
    -------
    SpectrumComparison
        Complete comparison result
    """
    # Use unfolded grid if no common grid specified
    if common_grid is None:
        common_grid = unfolded_energy
    
    # Rebin both spectra to common grid
    if not np.allclose(unfolded_energy, common_grid):
        unfolded_rebinned, unfolded_unc_rebinned = rebin_spectrum(
            unfolded, unfolded_energy, common_grid, unfolded_unc
        )
    else:
        unfolded_rebinned = unfolded
        unfolded_unc_rebinned = unfolded_unc
    
    if not np.allclose(transport_energy, common_grid):
        transport_rebinned, transport_unc_rebinned = rebin_spectrum(
            transport, transport_energy, common_grid, transport_unc
        )
    else:
        transport_rebinned = transport
        transport_unc_rebinned = transport_unc
    
    return SpectrumComparison(
        energy_grid=common_grid,
        unfolded=unfolded_rebinned,
        transport=transport_rebinned,
        unfolded_unc=unfolded_unc_rebinned if unfolded_unc_rebinned is not None else np.array([]),
        transport_unc=transport_unc_rebinned if transport_unc_rebinned is not None else np.array([]),
        transport_code=transport_code,
    )


def compare_reaction_rates(
    reactions: list[str],
    measured: np.ndarray,
    measured_unc: np.ndarray,
    spectrum: np.ndarray,
    energy_grid: np.ndarray,
    cross_sections: dict[str, np.ndarray],
) -> ReactionRateComparison:
    """Compare measured reaction rates to spectrum-folded calculations.
    
    Parameters
    ----------
    reactions : list[str]
        Reaction identifiers
    measured : ndarray
        Measured reaction rates
    measured_unc : ndarray
        Measurement uncertainties
    spectrum : ndarray
        Flux spectrum for folding
    energy_grid : ndarray
        Energy bin boundaries
    cross_sections : dict
        Cross-sections for each reaction, keyed by reaction name
        
    Returns
    -------
    ReactionRateComparison
        Comparison result
    """
    n_reactions = len(reactions)
    calculated = np.zeros(n_reactions)
    calculated_unc = np.zeros(n_reactions)
    
    # Compute bin widths for integration
    widths = np.diff(energy_grid)
    
    for i, rxn in enumerate(reactions):
        if rxn in cross_sections:
            xs = cross_sections[rxn]
            # Fold cross-section with spectrum
            calculated[i] = np.sum(xs * spectrum * widths)
        else:
            calculated[i] = np.nan
    
    return ReactionRateComparison(
        reactions=reactions,
        measured=measured,
        measured_unc=measured_unc,
        calculated=calculated,
        calculated_unc=calculated_unc,
    )


@dataclass
class TransportComparisonBundle:
    """Complete comparison bundle for validation.
    
    Attributes
    ----------
    spectrum_comparison : SpectrumComparison, optional
        Flux spectrum comparison
    reaction_rate_comparison : ReactionRateComparison, optional
        Reaction rate comparison
    provenance : dict
        Provenance information
    metadata : dict
        Additional metadata
    passed : bool
        Whether validation criteria are met
    criteria_results : dict
        Results for individual criteria
    """
    
    spectrum_comparison: Optional[SpectrumComparison] = None
    reaction_rate_comparison: Optional[ReactionRateComparison] = None
    provenance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    passed: bool = False
    criteria_results: dict = field(default_factory=dict)
    
    def evaluate_criteria(
        self,
        max_chi2_reduced: float = 3.0,
        max_mean_ce_deviation: float = 0.10,
        min_fraction_within_1sigma: float = 0.68,
    ) -> bool:
        """Evaluate pass/fail criteria.
        
        Parameters
        ----------
        max_chi2_reduced : float
            Maximum acceptable reduced chi-squared
        max_mean_ce_deviation : float
            Maximum acceptable deviation of mean C/E from unity
        min_fraction_within_1sigma : float
            Minimum fraction of groups within 1σ
            
        Returns
        -------
        bool
            Whether all criteria pass
        """
        self.criteria_results = {}
        
        if self.spectrum_comparison is not None:
            sc = self.spectrum_comparison
            
            self.criteria_results["chi2_reduced"] = {
                "value": sc.chi2_reduced,
                "threshold": max_chi2_reduced,
                "passed": sc.chi2_reduced <= max_chi2_reduced,
            }
            
            mean_deviation = abs(sc.mean_c_over_e - 1.0)
            self.criteria_results["mean_ce_deviation"] = {
                "value": mean_deviation,
                "threshold": max_mean_ce_deviation,
                "passed": mean_deviation <= max_mean_ce_deviation,
            }
            
            frac_1sigma = float(np.sum(sc.within_uncertainty)) / sc.n_groups if sc.n_groups > 0 else 0
            self.criteria_results["fraction_within_1sigma"] = {
                "value": frac_1sigma,
                "threshold": min_fraction_within_1sigma,
                "passed": frac_1sigma >= min_fraction_within_1sigma,
            }
        
        if self.reaction_rate_comparison is not None:
            rr = self.reaction_rate_comparison
            
            mean_deviation = abs(rr.mean_c_over_e - 1.0)
            self.criteria_results["rxn_rate_ce_deviation"] = {
                "value": mean_deviation,
                "threshold": max_mean_ce_deviation,
                "passed": mean_deviation <= max_mean_ce_deviation,
            }
        
        self.passed = all(c["passed"] for c in self.criteria_results.values())
        return self.passed
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = ["=" * 70, "TRANSPORT COMPARISON VALIDATION REPORT", "=" * 70]
        
        if self.spectrum_comparison:
            lines.append("")
            lines.append(self.spectrum_comparison.summary())
        
        if self.reaction_rate_comparison:
            lines.append("")
            lines.append(self.reaction_rate_comparison.summary_table())
        
        lines.append("")
        lines.append("VALIDATION CRITERIA")
        lines.append("-" * 70)
        for name, result in self.criteria_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            lines.append(f"  {name}: {result['value']:.4f} vs {result['threshold']:.4f} [{status}]")
        
        lines.append("-" * 70)
        overall = "PASSED" if self.passed else "FAILED"
        lines.append(f"OVERALL: {overall}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "criteria_results": self.criteria_results,
            "spectrum_comparison": self.spectrum_comparison.to_dict() if self.spectrum_comparison else None,
            "reaction_rate_comparison": self.reaction_rate_comparison.to_dict() if self.reaction_rate_comparison else None,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }


__all__ = [
    "TransportCode",
    "SpectrumComparison",
    "ReactionRateComparison",
    "TransportComparisonBundle",
    "rebin_spectrum",
    "compare_flux_spectrum",
    "compare_reaction_rates",
]
