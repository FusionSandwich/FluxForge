"""
Validation Module for FluxForge.

Provides C/E (Calculated/Experimental) ratio tables, closure metrics,
and validation bundle structures for comparing unfolded spectra with
reference data or transport calculations.

Features:
- C/E ratio calculation with uncertainties
- Closure metrics (chi-square, residuals, pull distributions)
- Validation bundle artifact for reproducibility
- Statistical tests for goodness-of-fit

References:
- STAYSL PNNL validation methodology
- IAEA CRP on validation of dosimetry calculations

Author: FluxForge Development Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from scipy import stats


class ValidationStatus(Enum):
    """Overall validation status."""
    
    PASSED = "passed"  # All metrics within tolerance
    MARGINAL = "marginal"  # Some metrics borderline
    FAILED = "failed"  # Significant discrepancies
    INCOMPLETE = "incomplete"  # Missing required data


@dataclass
class CEEntry:
    """
    Single C/E (Calculated/Experimental) entry.
    
    Attributes:
        identifier: Reaction or group identifier
        calculated: Calculated value (C)
        experimental: Experimental/reference value (E)
        c_uncertainty: Uncertainty in C
        e_uncertainty: Uncertainty in E
        ce_ratio: C/E ratio
        ce_uncertainty: Propagated uncertainty in C/E
        pull: Standardized residual (C-E)/σ
        within_tolerance: Whether C/E is within acceptable range
    """
    
    identifier: str
    calculated: float
    experimental: float
    c_uncertainty: float = 0.0
    e_uncertainty: float = 0.0
    ce_ratio: float = 0.0
    ce_uncertainty: float = 0.0
    pull: float = 0.0
    within_tolerance: bool = True
    
    def __post_init__(self):
        """Calculate derived values."""
        if self.experimental > 0:
            self.ce_ratio = self.calculated / self.experimental
            
            # Propagate uncertainty
            rel_c = self.c_uncertainty / self.calculated if self.calculated > 0 else 0
            rel_e = self.e_uncertainty / self.experimental if self.experimental > 0 else 0
            self.ce_uncertainty = self.ce_ratio * np.sqrt(rel_c**2 + rel_e**2)
            
            # Calculate pull (standardized residual)
            combined_unc = np.sqrt(self.c_uncertainty**2 + self.e_uncertainty**2)
            if combined_unc > 0:
                self.pull = (self.calculated - self.experimental) / combined_unc
        
        # Check tolerance (default: within 2σ of C/E = 1)
        if self.ce_uncertainty > 0:
            self.within_tolerance = abs(self.ce_ratio - 1.0) < 2 * self.ce_uncertainty


@dataclass
class CETable:
    """
    Complete C/E ratio table for validation.
    
    Attributes:
        entries: List of C/E entries
        description: Description of comparison
        tolerance: Acceptable deviation from C/E = 1
        created_at: Timestamp of table creation
    """
    
    entries: List[CEEntry] = field(default_factory=list)
    description: str = ""
    tolerance: float = 0.1  # 10% default tolerance
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def n_entries(self) -> int:
        """Number of entries."""
        return len(self.entries)
    
    @property
    def mean_ce(self) -> float:
        """Mean C/E ratio."""
        if not self.entries:
            return 0.0
        return np.mean([e.ce_ratio for e in self.entries])
    
    @property
    def std_ce(self) -> float:
        """Standard deviation of C/E ratios."""
        if len(self.entries) < 2:
            return 0.0
        return np.std([e.ce_ratio for e in self.entries], ddof=1)
    
    @property
    def fraction_within_tolerance(self) -> float:
        """Fraction of entries within tolerance."""
        if not self.entries:
            return 0.0
        return sum(1 for e in self.entries if e.within_tolerance) / len(self.entries)
    
    def to_markdown(self) -> str:
        """Generate markdown table."""
        lines = [
            "| Reaction | C | E | C/E | σ(C/E) | Pull | Status |",
            "|----------|---|---|-----|--------|------|--------|",
        ]
        
        for e in self.entries:
            status = "✓" if e.within_tolerance else "✗"
            lines.append(
                f"| {e.identifier} | {e.calculated:.4e} | {e.experimental:.4e} | "
                f"{e.ce_ratio:.3f} | {e.ce_uncertainty:.3f} | {e.pull:+.2f} | {status} |"
            )
        
        lines.append("")
        lines.append(f"Mean C/E: {self.mean_ce:.3f} ± {self.std_ce:.3f}")
        lines.append(f"Within tolerance: {self.fraction_within_tolerance:.1%}")
        
        return "\n".join(lines)


@dataclass
class ClosureMetrics:
    """
    Statistical metrics for validation closure.
    
    Attributes:
        chi_square: Chi-square statistic
        dof: Degrees of freedom
        reduced_chi2: Reduced chi-square (χ²/ν)
        p_value: P-value from chi-square test
        rms_residual: RMS of residuals
        mean_residual: Mean residual (bias indicator)
        max_pull: Maximum absolute pull value
        ks_statistic: Kolmogorov-Smirnov statistic for pull distribution
        ks_pvalue: K-S p-value (should be > 0.05 for normal pulls)
    """
    
    chi_square: float = 0.0
    dof: int = 0
    reduced_chi2: float = 0.0
    p_value: float = 0.0
    rms_residual: float = 0.0
    mean_residual: float = 0.0
    max_pull: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0
    
    @property
    def chi2_acceptable(self) -> bool:
        """Check if chi-square is acceptable (p > 0.05)."""
        return self.p_value > 0.05
    
    @property
    def pulls_normal(self) -> bool:
        """Check if pulls are consistent with normal distribution."""
        return self.ks_pvalue > 0.05
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Chi-square: {self.chi_square:.2f} (dof={self.dof})",
            f"Reduced χ²: {self.reduced_chi2:.3f}",
            f"P-value: {self.p_value:.4f} {'✓' if self.chi2_acceptable else '✗'}",
            f"RMS residual: {self.rms_residual:.4e}",
            f"Max |pull|: {self.max_pull:.2f}",
            f"Pull normality (K-S p): {self.ks_pvalue:.4f} {'✓' if self.pulls_normal else '✗'}",
        ]
        return "\n".join(lines)


@dataclass
class ValidationBundle:
    """
    Complete validation bundle for spectrum comparison.
    
    Attributes:
        ce_table: C/E ratio table
        closure: Closure metrics
        reference_type: Type of reference data (e.g., "MCNP", "OpenMC", "measurement")
        reference_label: Label for reference
        test_label: Label for test/unfolded data
        energy_bins: Energy bin boundaries if applicable
        status: Overall validation status
        notes: Additional notes
        provenance: Provenance information
    """
    
    ce_table: CETable = field(default_factory=CETable)
    closure: ClosureMetrics = field(default_factory=ClosureMetrics)
    reference_type: str = "reference"
    reference_label: str = ""
    test_label: str = ""
    energy_bins: Optional[np.ndarray] = None
    status: ValidationStatus = ValidationStatus.INCOMPLETE
    notes: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def determine_status(self) -> ValidationStatus:
        """Determine overall validation status based on metrics."""
        if not self.ce_table.entries:
            return ValidationStatus.INCOMPLETE
        
        # Check multiple criteria
        chi2_ok = self.closure.chi2_acceptable
        fraction_ok = self.ce_table.fraction_within_tolerance > 0.9
        pulls_ok = self.closure.max_pull < 3.0
        
        if chi2_ok and fraction_ok and pulls_ok:
            return ValidationStatus.PASSED
        elif chi2_ok or fraction_ok:
            return ValidationStatus.MARGINAL
        else:
            return ValidationStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'schema': 'fluxforge://validation_bundle/v1',
            'reference_type': self.reference_type,
            'reference_label': self.reference_label,
            'test_label': self.test_label,
            'status': self.status.value,
            'metrics': {
                'chi_square': self.closure.chi_square,
                'reduced_chi2': self.closure.reduced_chi2,
                'p_value': self.closure.p_value,
                'mean_ce': self.ce_table.mean_ce,
                'std_ce': self.ce_table.std_ce,
                'fraction_within_tolerance': self.ce_table.fraction_within_tolerance,
            },
            'ce_entries': [
                {
                    'identifier': e.identifier,
                    'calculated': e.calculated,
                    'experimental': e.experimental,
                    'ce_ratio': e.ce_ratio,
                    'pull': e.pull,
                }
                for e in self.ce_table.entries
            ],
            'notes': self.notes,
            'provenance': self.provenance,
        }
    
    def full_report(self) -> str:
        """Generate full validation report."""
        lines = [
            "=" * 70,
            "VALIDATION REPORT",
            "=" * 70,
            f"Reference: {self.reference_label} ({self.reference_type})",
            f"Test:      {self.test_label}",
            f"Status:    {self.status.value.upper()}",
            "",
            "--- C/E Table ---",
            self.ce_table.to_markdown(),
            "",
            "--- Closure Metrics ---",
            self.closure.summary(),
            "",
        ]
        
        if self.notes:
            lines.extend(["--- Notes ---", self.notes, ""])
        
        lines.append("=" * 70)
        return "\n".join(lines)


def calculate_ce_table(
    calculated: np.ndarray,
    experimental: np.ndarray,
    c_uncertainties: Optional[np.ndarray] = None,
    e_uncertainties: Optional[np.ndarray] = None,
    identifiers: Optional[List[str]] = None,
    tolerance: float = 0.1,
) -> CETable:
    """
    Calculate C/E table from arrays.
    
    Parameters
    ----------
    calculated : np.ndarray
        Calculated/predicted values.
    experimental : np.ndarray
        Experimental/reference values.
    c_uncertainties : np.ndarray, optional
        Uncertainties in calculated values.
    e_uncertainties : np.ndarray, optional
        Uncertainties in experimental values.
    identifiers : list of str, optional
        Identifiers for each entry (default: numbered).
    tolerance : float
        Acceptable C/E deviation from 1.0.
        
    Returns
    -------
    CETable
        Populated C/E table.
    """
    n = len(calculated)
    
    if c_uncertainties is None:
        c_uncertainties = np.zeros(n)
    if e_uncertainties is None:
        e_uncertainties = np.zeros(n)
    if identifiers is None:
        identifiers = [f"Entry_{i+1}" for i in range(n)]
    
    entries = []
    for i in range(n):
        entry = CEEntry(
            identifier=identifiers[i],
            calculated=float(calculated[i]),
            experimental=float(experimental[i]),
            c_uncertainty=float(c_uncertainties[i]),
            e_uncertainty=float(e_uncertainties[i]),
        )
        # Check tolerance
        entry.within_tolerance = abs(entry.ce_ratio - 1.0) < tolerance
        entries.append(entry)
    
    return CETable(entries=entries, tolerance=tolerance)


def calculate_closure_metrics(
    calculated: np.ndarray,
    experimental: np.ndarray,
    covariance: Optional[np.ndarray] = None,
    c_uncertainties: Optional[np.ndarray] = None,
    e_uncertainties: Optional[np.ndarray] = None,
) -> ClosureMetrics:
    """
    Calculate closure metrics for validation.
    
    Parameters
    ----------
    calculated : np.ndarray
        Calculated/predicted values.
    experimental : np.ndarray
        Experimental/reference values.
    covariance : np.ndarray, optional
        Full covariance matrix (preferred).
    c_uncertainties : np.ndarray, optional
        Uncertainties in calculated (if no covariance).
    e_uncertainties : np.ndarray, optional
        Uncertainties in experimental (if no covariance).
        
    Returns
    -------
    ClosureMetrics
        Statistical closure metrics.
    """
    residuals = calculated - experimental
    n = len(residuals)
    
    # Build covariance if not provided
    if covariance is None:
        if c_uncertainties is None:
            c_uncertainties = np.zeros(n)
        if e_uncertainties is None:
            e_uncertainties = np.zeros(n)
        # Assume independent
        variances = c_uncertainties**2 + e_uncertainties**2
        covariance = np.diag(variances)
    
    # Handle near-zero variances
    diag = np.diag(covariance).copy()
    diag[diag < 1e-30] = 1e-30
    
    # Chi-square
    try:
        cov_inv = np.linalg.pinv(covariance)
        chi2 = float(residuals @ cov_inv @ residuals)
    except np.linalg.LinAlgError:
        chi2 = 0.0
    
    dof = max(1, n)
    reduced_chi2 = chi2 / dof if dof > 0 else 0
    
    # P-value
    p_value = 1.0 - stats.chi2.cdf(chi2, dof) if chi2 > 0 else 1.0
    
    # Residual statistics
    rms = float(np.sqrt(np.mean(residuals**2)))
    mean_res = float(np.mean(residuals))
    
    # Pull values
    pulls = residuals / np.sqrt(diag)
    max_pull = float(np.max(np.abs(pulls)))
    
    # K-S test for normality of pulls
    try:
        ks_stat, ks_p = stats.kstest(pulls, 'norm')
    except:
        ks_stat, ks_p = 0.0, 0.0
    
    return ClosureMetrics(
        chi_square=chi2,
        dof=dof,
        reduced_chi2=reduced_chi2,
        p_value=p_value,
        rms_residual=rms,
        mean_residual=mean_res,
        max_pull=max_pull,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
    )


def create_validation_bundle(
    calculated: np.ndarray,
    experimental: np.ndarray,
    c_uncertainties: Optional[np.ndarray] = None,
    e_uncertainties: Optional[np.ndarray] = None,
    covariance: Optional[np.ndarray] = None,
    identifiers: Optional[List[str]] = None,
    reference_type: str = "reference",
    reference_label: str = "",
    test_label: str = "",
    tolerance: float = 0.1,
) -> ValidationBundle:
    """
    Create complete validation bundle.
    
    Parameters
    ----------
    calculated : np.ndarray
        Calculated/predicted values.
    experimental : np.ndarray
        Experimental/reference values.
    c_uncertainties : np.ndarray, optional
        Uncertainties in calculated.
    e_uncertainties : np.ndarray, optional
        Uncertainties in experimental.
    covariance : np.ndarray, optional
        Full covariance matrix.
    identifiers : list of str, optional
        Identifiers for each entry.
    reference_type : str
        Type of reference data.
    reference_label : str
        Label for reference.
    test_label : str
        Label for test data.
    tolerance : float
        C/E tolerance.
        
    Returns
    -------
    ValidationBundle
        Complete validation bundle.
        
    Examples
    --------
    >>> calc = np.array([1.02, 0.98, 1.05, 0.95])
    >>> expt = np.array([1.00, 1.00, 1.00, 1.00])
    >>> bundle = create_validation_bundle(calc, expt, 
    ...     reference_label="MCNP flux", test_label="Unfolded flux")
    >>> print(bundle.status)
    """
    ce_table = calculate_ce_table(
        calculated, experimental,
        c_uncertainties, e_uncertainties,
        identifiers, tolerance
    )
    
    closure = calculate_closure_metrics(
        calculated, experimental,
        covariance, c_uncertainties, e_uncertainties
    )
    
    bundle = ValidationBundle(
        ce_table=ce_table,
        closure=closure,
        reference_type=reference_type,
        reference_label=reference_label,
        test_label=test_label,
    )
    
    bundle.status = bundle.determine_status()
    bundle.provenance = {
        'created_at': datetime.now().isoformat(),
        'n_entries': len(calculated),
        'tolerance': tolerance,
    }
    
    return bundle
