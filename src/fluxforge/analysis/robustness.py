"""
Wire Set Robustness Diagnostics Module.

Provides stability analysis for activation monitor wire sets used in 
neutron spectrum unfolding. Includes condition number analysis,
redundancy metrics, and sensitivity diagnostics.

Features:
- Response matrix conditioning analysis
- Wire set information content metrics
- Singular value decomposition diagnostics
- Energy coverage and gap detection
- Leave-one-out stability analysis
- Optimal wire set recommendations

References:
- STAYSL PNNL documentation
- "Neutron Spectrum Unfolding" IAEA Technical Reports

Author: FluxForge Development Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Sequence

import numpy as np
from scipy import linalg


class RobustnessLevel(Enum):
    """Qualitative robustness classification."""
    
    EXCELLENT = "excellent"  # Well-conditioned, redundant
    GOOD = "good"  # Acceptable conditioning
    MARGINAL = "marginal"  # May have stability issues
    POOR = "poor"  # Likely unstable
    CRITICAL = "critical"  # Severely ill-conditioned


@dataclass
class ConditioningMetrics:
    """
    Condition number and related metrics for a response matrix.
    
    Attributes:
        condition_number: Standard condition number (σ_max/σ_min)
        log_condition: log10 of condition number
        effective_rank: Number of significant singular values
        truncation_threshold: Threshold used for rank determination
        singular_values: Array of singular values
        explained_variance: Cumulative variance explained by each SV
    """
    
    condition_number: float
    log_condition: float
    effective_rank: int
    truncation_threshold: float
    singular_values: np.ndarray
    explained_variance: np.ndarray
    
    @property
    def robustness_level(self) -> RobustnessLevel:
        """Classify robustness based on condition number."""
        if self.log_condition < 2:
            return RobustnessLevel.EXCELLENT
        elif self.log_condition < 4:
            return RobustnessLevel.GOOD
        elif self.log_condition < 6:
            return RobustnessLevel.MARGINAL
        elif self.log_condition < 8:
            return RobustnessLevel.POOR
        else:
            return RobustnessLevel.CRITICAL


@dataclass
class EnergyCoverage:
    """
    Energy coverage analysis for a wire set.
    
    Attributes:
        energy_bins: Energy bin boundaries
        coverage_per_group: Fraction of information per energy group
        gaps: List of (start_eV, end_eV) tuples for poorly covered regions
        peak_sensitivity_groups: Groups with highest sensitivity
        thermal_coverage: Coverage in thermal region (<0.5 eV)
        epithermal_coverage: Coverage in epithermal (0.5 eV - 100 keV)
        fast_coverage: Coverage in fast region (>100 keV)
    """
    
    energy_bins: np.ndarray
    coverage_per_group: np.ndarray
    gaps: List[Tuple[float, float]]
    peak_sensitivity_groups: List[int]
    thermal_coverage: float
    epithermal_coverage: float
    fast_coverage: float


@dataclass 
class LeaveOneOutResult:
    """
    Result of leave-one-out stability analysis.
    
    Attributes:
        wire_index: Index of removed wire
        wire_name: Name/identifier of removed wire
        condition_change: Change in condition number
        rank_change: Change in effective rank
        critical: Whether removal critically destabilizes
        importance_score: Importance of this wire (0-1)
    """
    
    wire_index: int
    wire_name: str
    condition_change: float
    rank_change: int
    critical: bool
    importance_score: float


@dataclass
class WireSetDiagnostics:
    """
    Complete diagnostics for a wire set / response matrix.
    
    Attributes:
        n_wires: Number of activation monitors
        n_groups: Number of energy groups
        conditioning: Condition number metrics
        coverage: Energy coverage analysis
        leave_one_out: Leave-one-out stability results
        redundancy_factor: Overall redundancy (n_wires / effective_rank)
        recommendations: List of improvement recommendations
        overall_robustness: Overall robustness classification
    """
    
    n_wires: int
    n_groups: int
    conditioning: ConditioningMetrics
    coverage: EnergyCoverage
    leave_one_out: List[LeaveOneOutResult]
    redundancy_factor: float
    recommendations: List[str]
    overall_robustness: RobustnessLevel
    
    def summary(self) -> str:
        """Generate text summary of diagnostics."""
        lines = [
            "=" * 60,
            "WIRE SET ROBUSTNESS DIAGNOSTICS",
            "=" * 60,
            f"Wires: {self.n_wires}  |  Energy Groups: {self.n_groups}",
            f"Condition Number: {self.conditioning.condition_number:.2e} "
            f"(log10: {self.conditioning.log_condition:.1f})",
            f"Effective Rank: {self.conditioning.effective_rank}",
            f"Redundancy Factor: {self.redundancy_factor:.2f}",
            f"Overall Robustness: {self.overall_robustness.value.upper()}",
            "",
            "Energy Coverage:",
            f"  Thermal (<0.5 eV):      {self.coverage.thermal_coverage:.1%}",
            f"  Epithermal (0.5eV-100keV): {self.coverage.epithermal_coverage:.1%}",
            f"  Fast (>100 keV):        {self.coverage.fast_coverage:.1%}",
        ]
        
        if self.coverage.gaps:
            lines.append(f"  Coverage Gaps: {len(self.coverage.gaps)}")
        
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def calculate_condition_metrics(
    R: np.ndarray,
    truncation_ratio: float = 1e-10,
) -> ConditioningMetrics:
    """
    Calculate condition number and SVD-based metrics for response matrix.
    
    Parameters
    ----------
    R : np.ndarray
        Response matrix with shape (n_wires, n_groups).
    truncation_ratio : float
        Threshold for determining effective rank (σ_i/σ_max).
        
    Returns
    -------
    ConditioningMetrics
        Comprehensive conditioning analysis.
    """
    # Compute SVD
    U, s, Vh = linalg.svd(R, full_matrices=False)
    
    # Condition number
    sigma_max = s[0]
    sigma_min = s[-1] if s[-1] > 0 else 1e-300
    cond = sigma_max / sigma_min
    log_cond = np.log10(cond) if cond > 0 else 0.0
    
    # Effective rank
    threshold = sigma_max * truncation_ratio
    effective_rank = np.sum(s > threshold)
    
    # Explained variance
    total_var = np.sum(s ** 2)
    cumulative_var = np.cumsum(s ** 2) / total_var if total_var > 0 else s
    
    return ConditioningMetrics(
        condition_number=cond,
        log_condition=log_cond,
        effective_rank=effective_rank,
        truncation_threshold=threshold,
        singular_values=s,
        explained_variance=cumulative_var,
    )


def analyze_energy_coverage(
    R: np.ndarray,
    energy_bins: np.ndarray,
    coverage_threshold: float = 0.01,
) -> EnergyCoverage:
    """
    Analyze energy coverage of wire set response.
    
    Parameters
    ----------
    R : np.ndarray
        Response matrix with shape (n_wires, n_groups).
    energy_bins : np.ndarray
        Energy bin boundaries in eV (n_groups + 1 values).
    coverage_threshold : float
        Minimum normalized response to consider "covered".
        
    Returns
    -------
    EnergyCoverage
        Energy coverage analysis results.
    """
    n_wires, n_groups = R.shape
    
    # Normalize responses per wire for coverage analysis
    R_norm = R / (np.max(R, axis=1, keepdims=True) + 1e-30)
    
    # Coverage per group: max response across all wires
    coverage = np.max(R_norm, axis=0)
    
    # Identify gaps (poorly covered regions)
    gaps = []
    in_gap = False
    gap_start = 0
    
    for g in range(n_groups):
        if coverage[g] < coverage_threshold:
            if not in_gap:
                in_gap = True
                gap_start = energy_bins[g]
        else:
            if in_gap:
                gaps.append((gap_start, energy_bins[g]))
                in_gap = False
    
    if in_gap:
        gaps.append((gap_start, energy_bins[-1]))
    
    # Peak sensitivity groups
    peak_groups = np.argsort(coverage)[-5:][::-1].tolist()
    
    # Regional coverage
    thermal_mask = energy_bins[:-1] < 0.5
    epithermal_mask = (energy_bins[:-1] >= 0.5) & (energy_bins[:-1] < 1e5)
    fast_mask = energy_bins[:-1] >= 1e5
    
    def region_coverage(mask: np.ndarray) -> float:
        if np.sum(mask) == 0:
            return 0.0
        return np.mean(coverage[mask])
    
    return EnergyCoverage(
        energy_bins=energy_bins,
        coverage_per_group=coverage,
        gaps=gaps,
        peak_sensitivity_groups=peak_groups,
        thermal_coverage=region_coverage(thermal_mask),
        epithermal_coverage=region_coverage(epithermal_mask),
        fast_coverage=region_coverage(fast_mask),
    )


def leave_one_out_analysis(
    R: np.ndarray,
    wire_names: Optional[List[str]] = None,
    critical_threshold: float = 2.0,
) -> List[LeaveOneOutResult]:
    """
    Perform leave-one-out stability analysis.
    
    For each wire, compute how the condition number and rank change
    when that wire is removed. Identifies critical wires.
    
    Parameters
    ----------
    R : np.ndarray
        Response matrix with shape (n_wires, n_groups).
    wire_names : list of str, optional
        Names for each wire (default: Wire_0, Wire_1, ...).
    critical_threshold : float
        Factor increase in condition number to consider critical.
        
    Returns
    -------
    list of LeaveOneOutResult
        Analysis result for each wire.
    """
    n_wires, n_groups = R.shape
    
    if wire_names is None:
        wire_names = [f"Wire_{i}" for i in range(n_wires)]
    
    # Baseline metrics
    baseline = calculate_condition_metrics(R)
    baseline_cond = baseline.log_condition
    baseline_rank = baseline.effective_rank
    
    results = []
    
    for i in range(n_wires):
        # Remove wire i
        R_reduced = np.delete(R, i, axis=0)
        
        # Calculate reduced metrics
        if R_reduced.shape[0] > 0:
            reduced = calculate_condition_metrics(R_reduced)
            cond_change = reduced.log_condition - baseline_cond
            rank_change = reduced.effective_rank - baseline_rank
        else:
            cond_change = 10.0  # Very bad
            rank_change = -baseline_rank
        
        # Determine criticality
        critical = cond_change > np.log10(critical_threshold)
        
        # Importance score (normalized to 0-1)
        importance = min(1.0, max(0.0, cond_change / 2.0))
        
        results.append(LeaveOneOutResult(
            wire_index=i,
            wire_name=wire_names[i],
            condition_change=cond_change,
            rank_change=rank_change,
            critical=critical,
            importance_score=importance,
        ))
    
    return results


def generate_recommendations(
    conditioning: ConditioningMetrics,
    coverage: EnergyCoverage,
    leave_one_out: List[LeaveOneOutResult],
    n_wires: int,
) -> List[str]:
    """Generate improvement recommendations based on diagnostics."""
    recommendations = []
    
    # Conditioning-based recommendations
    if conditioning.log_condition > 8:
        recommendations.append(
            "CRITICAL: Response matrix is severely ill-conditioned. "
            "Results may be unreliable."
        )
    elif conditioning.log_condition > 6:
        recommendations.append(
            "Consider adding more wires to improve conditioning."
        )
    
    # Rank-based recommendations
    if conditioning.effective_rank < n_wires * 0.5:
        recommendations.append(
            f"Only {conditioning.effective_rank}/{n_wires} wires provide "
            "independent information. Some wires may be redundant."
        )
    
    # Coverage-based recommendations
    if coverage.thermal_coverage < 0.3:
        recommendations.append(
            "Poor thermal coverage. Consider adding Au-197(n,g) or similar."
        )
    if coverage.epithermal_coverage < 0.3:
        recommendations.append(
            "Poor epithermal coverage. Consider adding In-115(n,g) or similar."
        )
    if coverage.fast_coverage < 0.3:
        recommendations.append(
            "Poor fast coverage. Consider adding threshold reactions "
            "(e.g., Ni-58(n,p), Ti-46(n,p))."
        )
    
    if coverage.gaps:
        gap_str = ", ".join([f"{g[0]:.2e}-{g[1]:.2e} eV" for g in coverage.gaps[:3]])
        recommendations.append(f"Coverage gaps detected: {gap_str}")
    
    # Critical wire warnings
    critical_wires = [r for r in leave_one_out if r.critical]
    if critical_wires:
        names = ", ".join([w.wire_name for w in critical_wires])
        recommendations.append(
            f"Critical wires (removal destabilizes): {names}"
        )
    
    if not recommendations:
        recommendations.append("Wire set appears well-conditioned and balanced.")
    
    return recommendations


def diagnose_wire_set(
    R: np.ndarray,
    energy_bins: np.ndarray,
    wire_names: Optional[List[str]] = None,
    truncation_ratio: float = 1e-10,
) -> WireSetDiagnostics:
    """
    Perform comprehensive robustness diagnostics on a wire set.
    
    Parameters
    ----------
    R : np.ndarray
        Response matrix with shape (n_wires, n_groups).
    energy_bins : np.ndarray
        Energy bin boundaries in eV (n_groups + 1 values).
    wire_names : list of str, optional
        Names for each wire.
    truncation_ratio : float
        SVD truncation threshold for rank determination.
        
    Returns
    -------
    WireSetDiagnostics
        Complete diagnostic results.
        
    Examples
    --------
    >>> R = np.random.rand(10, 100)  # 10 wires, 100 energy groups
    >>> energy_bins = np.logspace(-4, 1, 101) * 1e6  # eV
    >>> diag = diagnose_wire_set(R, energy_bins)
    >>> print(diag.summary())
    """
    n_wires, n_groups = R.shape
    
    if wire_names is None:
        wire_names = [f"Wire_{i}" for i in range(n_wires)]
    
    # Calculate all metrics
    conditioning = calculate_condition_metrics(R, truncation_ratio)
    coverage = analyze_energy_coverage(R, energy_bins)
    loo_results = leave_one_out_analysis(R, wire_names)
    
    # Redundancy factor
    redundancy = n_wires / max(1, conditioning.effective_rank)
    
    # Generate recommendations
    recommendations = generate_recommendations(
        conditioning, coverage, loo_results, n_wires
    )
    
    # Overall robustness
    # Combine condition number, coverage, and redundancy
    cond_score = min(10, conditioning.log_condition) / 10  # 0-1, lower is better
    coverage_score = 1 - (coverage.thermal_coverage + 
                          coverage.epithermal_coverage + 
                          coverage.fast_coverage) / 3
    redundancy_score = 1 - min(1, redundancy / 3)  # Want redundancy >= 2
    
    combined = (cond_score + coverage_score + redundancy_score) / 3
    
    if combined < 0.2:
        overall = RobustnessLevel.EXCELLENT
    elif combined < 0.35:
        overall = RobustnessLevel.GOOD
    elif combined < 0.5:
        overall = RobustnessLevel.MARGINAL
    elif combined < 0.7:
        overall = RobustnessLevel.POOR
    else:
        overall = RobustnessLevel.CRITICAL
    
    return WireSetDiagnostics(
        n_wires=n_wires,
        n_groups=n_groups,
        conditioning=conditioning,
        coverage=coverage,
        leave_one_out=loo_results,
        redundancy_factor=redundancy,
        recommendations=recommendations,
        overall_robustness=overall,
    )


def quick_condition_check(R: np.ndarray) -> Tuple[float, str]:
    """
    Quick condition number check with qualitative assessment.
    
    Parameters
    ----------
    R : np.ndarray
        Response matrix.
        
    Returns
    -------
    tuple
        (log10_condition_number, assessment_string)
    """
    metrics = calculate_condition_metrics(R)
    level = metrics.robustness_level
    
    assessments = {
        RobustnessLevel.EXCELLENT: "Excellent - well-conditioned",
        RobustnessLevel.GOOD: "Good - acceptable conditioning",
        RobustnessLevel.MARGINAL: "Marginal - may have stability issues",
        RobustnessLevel.POOR: "Poor - likely unstable",
        RobustnessLevel.CRITICAL: "Critical - severely ill-conditioned",
    }
    
    return metrics.log_condition, assessments[level]


def estimate_optimal_wire_count(
    n_groups: int,
    target_redundancy: float = 2.0,
    desired_condition: float = 1e4,
) -> int:
    """
    Estimate optimal number of wires for given energy structure.
    
    Parameters
    ----------
    n_groups : int
        Number of energy groups.
    target_redundancy : float
        Desired redundancy factor.
    desired_condition : float
        Target condition number.
        
    Returns
    -------
    int
        Recommended number of wires.
    """
    # Rule of thumb: need at least log2(n_groups) truly independent measurements
    # Plus redundancy for stability
    min_independent = max(3, int(np.log2(n_groups)))
    recommended = int(min_independent * target_redundancy)
    
    # Cap at reasonable values
    return max(5, min(recommended, 30))
