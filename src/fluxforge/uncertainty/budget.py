"""Uncertainty budget decomposition and tracking.

This module provides tools for decomposing total uncertainty into
component sources, following STAYSL PNNL convention for reactor
dosimetry applications.

Uncertainty Categories (STAYSL convention):
1. Counting statistics (Poisson)
2. Efficiency calibration (detector response)
3. Timing (irradiation/cooling/counting)
4. Geometry (sample-detector distance)
5. Cross-section (nuclear data)
6. Decay data (half-life, branching ratios)
7. Self-shielding corrections
8. Prior spectrum uncertainty

References:
    PNNL-22253 (STAYSL PNNL User Manual)
    GUM (Guide to the Expression of Uncertainty in Measurement)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class UncertaintyCategory(Enum):
    """Standard uncertainty categories for activation analysis."""
    
    # Experimental uncertainties
    COUNTING_STATISTICS = "counting"
    EFFICIENCY = "efficiency"
    TIMING = "timing"
    GEOMETRY = "geometry"
    DEAD_TIME = "dead_time"
    PILE_UP = "pile_up"
    
    # Nuclear data uncertainties
    CROSS_SECTION = "cross_section"
    DECAY_DATA = "decay_data"
    BRANCHING_RATIO = "branching_ratio"
    HALF_LIFE = "half_life"
    
    # Correction uncertainties
    SELF_SHIELDING = "self_shielding"
    COINCIDENCE_SUMMING = "coincidence_summing"
    BACKGROUND = "background"
    
    # Spectral uncertainties
    PRIOR_SPECTRUM = "prior_spectrum"
    RESPONSE_MATRIX = "response_matrix"
    ENERGY_CALIBRATION = "energy_calibration"
    
    # k0-NAA specific
    K0_FACTOR = "k0_factor"
    Q0_RATIO = "q0_ratio"
    ALPHA_PARAMETER = "alpha"
    F_FLUX_RATIO = "f_flux_ratio"
    
    # Other
    OTHER = "other"
    TOTAL = "total"


@dataclass
class UncertaintyComponent:
    """Single uncertainty component.
    
    Attributes
    ----------
    category : UncertaintyCategory
        Type of uncertainty source
    value : float
        Absolute uncertainty value (same units as measurement)
    relative : float
        Relative uncertainty (fractional, not percent)
    description : str
        Human-readable description
    is_correlated : bool
        Whether this component is correlated across measurements
    correlation_group : str, optional
        Identifier for correlation group (if correlated)
    """
    
    category: UncertaintyCategory
    value: float
    relative: float = 0.0
    description: str = ""
    is_correlated: bool = False
    correlation_group: Optional[str] = None
    
    @classmethod
    def from_relative(
        cls,
        category: UncertaintyCategory,
        relative: float,
        measurement: float,
        description: str = "",
    ) -> UncertaintyComponent:
        """Create component from relative uncertainty.
        
        Parameters
        ----------
        category : UncertaintyCategory
            Uncertainty type
        relative : float
            Relative uncertainty (fractional)
        measurement : float
            Measured value for computing absolute uncertainty
        description : str
            Description
            
        Returns
        -------
        UncertaintyComponent
        """
        return cls(
            category=category,
            value=relative * abs(measurement),
            relative=relative,
            description=description,
        )
    
    @classmethod
    def from_absolute(
        cls,
        category: UncertaintyCategory,
        value: float,
        measurement: float,
        description: str = "",
    ) -> UncertaintyComponent:
        """Create component from absolute uncertainty.
        
        Parameters
        ----------
        category : UncertaintyCategory
            Uncertainty type
        value : float
            Absolute uncertainty
        measurement : float
            Measured value for computing relative uncertainty
        description : str
            Description
            
        Returns
        -------
        UncertaintyComponent
        """
        relative = value / abs(measurement) if measurement != 0 else 0.0
        return cls(
            category=category,
            value=value,
            relative=relative,
            description=description,
        )


@dataclass
class UncertaintyBudget:
    """Complete uncertainty budget with component breakdown.
    
    Follows GUM methodology for combining uncertainties from
    multiple independent sources.
    
    Attributes
    ----------
    measurement : float
        Central value of the measurement
    total_uncertainty : float
        Combined standard uncertainty
    components : list[UncertaintyComponent]
        Individual uncertainty sources
    coverage_factor : float
        Coverage factor k for expanded uncertainty (default 1.0)
    confidence_level : float
        Confidence level for expanded uncertainty (95% for k=2)
    units : str
        Units of the measurement
    name : str
        Identifier for this budget
    """
    
    measurement: float
    total_uncertainty: float = 0.0
    components: list[UncertaintyComponent] = field(default_factory=list)
    coverage_factor: float = 1.0
    confidence_level: float = 0.68
    units: str = ""
    name: str = ""
    
    def add_component(self, component: UncertaintyComponent) -> None:
        """Add an uncertainty component."""
        self.components.append(component)
    
    def add_relative(
        self,
        category: UncertaintyCategory,
        relative: float,
        description: str = "",
    ) -> None:
        """Add component from relative uncertainty.
        
        Parameters
        ----------
        category : UncertaintyCategory
            Uncertainty type
        relative : float
            Relative uncertainty (fractional)
        description : str
            Description
        """
        self.components.append(
            UncertaintyComponent.from_relative(
                category, relative, self.measurement, description
            )
        )
    
    def add_absolute(
        self,
        category: UncertaintyCategory,
        value: float,
        description: str = "",
    ) -> None:
        """Add component from absolute uncertainty.
        
        Parameters
        ----------
        category : UncertaintyCategory
            Uncertainty type
        value : float
            Absolute uncertainty
        description : str
            Description
        """
        self.components.append(
            UncertaintyComponent.from_absolute(
                category, value, self.measurement, description
            )
        )
    
    def compute_total(self, method: str = "quadrature") -> float:
        """Compute total uncertainty from components.
        
        Parameters
        ----------
        method : str
            Combination method: "quadrature" (RSS) or "linear"
            
        Returns
        -------
        float
            Combined uncertainty
        """
        if not self.components:
            return self.total_uncertainty
        
        values = np.array([c.value for c in self.components])
        
        if method == "quadrature":
            # Root-sum-of-squares for uncorrelated uncertainties
            self.total_uncertainty = float(np.sqrt(np.sum(values**2)))
        elif method == "linear":
            # Linear sum (conservative, for correlated)
            self.total_uncertainty = float(np.sum(np.abs(values)))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.total_uncertainty
    
    @property
    def relative_total(self) -> float:
        """Total relative uncertainty."""
        if self.measurement == 0:
            return 0.0
        return self.total_uncertainty / abs(self.measurement)
    
    @property
    def expanded_uncertainty(self) -> float:
        """Expanded uncertainty (k × standard uncertainty)."""
        return self.coverage_factor * self.total_uncertainty
    
    def dominant_component(self) -> Optional[UncertaintyComponent]:
        """Get the largest uncertainty component."""
        if not self.components:
            return None
        return max(self.components, key=lambda c: abs(c.value))
    
    def fraction_by_category(self, category: UncertaintyCategory) -> float:
        """Get fraction of variance from a category.
        
        Parameters
        ----------
        category : UncertaintyCategory
            Category to check
            
        Returns
        -------
        float
            Fraction of total variance (0-1)
        """
        if self.total_uncertainty == 0:
            return 0.0
        
        category_var = sum(
            c.value**2 for c in self.components if c.category == category
        )
        total_var = self.total_uncertainty**2
        
        return category_var / total_var if total_var > 0 else 0.0
    
    def summary_table(self) -> str:
        """Generate text summary table.
        
        Returns
        -------
        str
            Formatted table
        """
        lines = [
            f"Uncertainty Budget: {self.name}",
            f"Measurement: {self.measurement:.6g} {self.units}",
            f"Total Uncertainty: ±{self.total_uncertainty:.6g} ({100*self.relative_total:.2f}%)",
            "",
            "Component Breakdown:",
            "-" * 70,
            f"{'Category':<25} {'Absolute':>12} {'Relative':>10} {'Variance %':>10}",
            "-" * 70,
        ]
        
        total_var = self.total_uncertainty**2
        
        for c in sorted(self.components, key=lambda x: -x.value**2):
            var_frac = (c.value**2 / total_var * 100) if total_var > 0 else 0
            lines.append(
                f"{c.category.value:<25} {c.value:>12.4g} {100*c.relative:>9.2f}% {var_frac:>9.1f}%"
            )
        
        lines.append("-" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "measurement": self.measurement,
            "total_uncertainty": self.total_uncertainty,
            "relative_total": self.relative_total,
            "units": self.units,
            "name": self.name,
            "coverage_factor": self.coverage_factor,
            "components": [
                {
                    "category": c.category.value,
                    "value": c.value,
                    "relative": c.relative,
                    "description": c.description,
                    "variance_fraction": self.fraction_by_category(c.category),
                }
                for c in self.components
            ],
        }


def combine_budgets(
    budgets: list[UncertaintyBudget],
    correlation_matrix: Optional[np.ndarray] = None,
) -> UncertaintyBudget:
    """Combine multiple uncertainty budgets.
    
    Parameters
    ----------
    budgets : list[UncertaintyBudget]
        Individual budgets to combine
    correlation_matrix : ndarray, optional
        Correlation matrix between budgets (default: identity)
        
    Returns
    -------
    UncertaintyBudget
        Combined budget
    """
    n = len(budgets)
    if n == 0:
        return UncertaintyBudget(measurement=0.0)
    
    if n == 1:
        return budgets[0]
    
    # Default to uncorrelated
    if correlation_matrix is None:
        correlation_matrix = np.eye(n)
    
    measurements = np.array([b.measurement for b in budgets])
    uncertainties = np.array([b.total_uncertainty for b in budgets])
    
    # Mean measurement
    mean_measurement = float(np.mean(measurements))
    
    # Combined uncertainty accounting for correlations
    cov = np.outer(uncertainties, uncertainties) * correlation_matrix
    combined_var = np.sum(cov) / n**2
    combined_unc = float(np.sqrt(combined_var))
    
    # Aggregate components by category
    category_totals: dict[UncertaintyCategory, float] = {}
    for budget in budgets:
        for c in budget.components:
            if c.category not in category_totals:
                category_totals[c.category] = 0.0
            category_totals[c.category] += c.value**2
    
    components = [
        UncertaintyComponent(
            category=cat,
            value=float(np.sqrt(var_sum)) / n,
            relative=float(np.sqrt(var_sum)) / n / abs(mean_measurement) if mean_measurement != 0 else 0.0,
        )
        for cat, var_sum in category_totals.items()
    ]
    
    return UncertaintyBudget(
        measurement=mean_measurement,
        total_uncertainty=combined_unc,
        components=components,
        name="combined",
    )


def create_activation_budget(
    activity: float,
    counting_rel: float = 0.01,
    efficiency_rel: float = 0.02,
    timing_rel: float = 0.005,
    geometry_rel: float = 0.01,
    cross_section_rel: float = 0.03,
    decay_data_rel: float = 0.005,
    self_shielding_rel: float = 0.01,
    coincidence_rel: float = 0.01,
    units: str = "Bq",
) -> UncertaintyBudget:
    """Create standard activation analysis uncertainty budget.
    
    Parameters
    ----------
    activity : float
        Measured activity
    counting_rel : float
        Counting statistics relative uncertainty (default 1%)
    efficiency_rel : float
        Efficiency calibration uncertainty (default 2%)
    timing_rel : float
        Timing uncertainty (default 0.5%)
    geometry_rel : float
        Geometry uncertainty (default 1%)
    cross_section_rel : float
        Cross-section uncertainty (default 3%)
    decay_data_rel : float
        Decay data uncertainty (default 0.5%)
    self_shielding_rel : float
        Self-shielding correction uncertainty (default 1%)
    coincidence_rel : float
        Coincidence summing uncertainty (default 1%)
    units : str
        Activity units
        
    Returns
    -------
    UncertaintyBudget
        Complete budget with all components
    """
    budget = UncertaintyBudget(
        measurement=activity,
        units=units,
        name="activation_analysis",
    )
    
    budget.add_relative(
        UncertaintyCategory.COUNTING_STATISTICS,
        counting_rel,
        "Poisson counting statistics",
    )
    budget.add_relative(
        UncertaintyCategory.EFFICIENCY,
        efficiency_rel,
        "Detector efficiency calibration",
    )
    budget.add_relative(
        UncertaintyCategory.TIMING,
        timing_rel,
        "Irradiation/cooling/counting timing",
    )
    budget.add_relative(
        UncertaintyCategory.GEOMETRY,
        geometry_rel,
        "Sample-detector geometry",
    )
    budget.add_relative(
        UncertaintyCategory.CROSS_SECTION,
        cross_section_rel,
        "Activation cross-section",
    )
    budget.add_relative(
        UncertaintyCategory.DECAY_DATA,
        decay_data_rel,
        "Half-life and branching ratios",
    )
    budget.add_relative(
        UncertaintyCategory.SELF_SHIELDING,
        self_shielding_rel,
        "Neutron self-shielding correction",
    )
    budget.add_relative(
        UncertaintyCategory.COINCIDENCE_SUMMING,
        coincidence_rel,
        "True coincidence summing correction",
    )
    
    budget.compute_total()
    
    return budget


def create_k0_naa_budget(
    concentration: float,
    counting_rel: float = 0.01,
    efficiency_rel: float = 0.02,
    k0_rel: float = 0.015,
    q0_rel: float = 0.05,
    f_rel: float = 0.03,
    alpha_abs: float = 0.01,
    timing_rel: float = 0.005,
    coincidence_rel: float = 0.01,
    units: str = "mg/kg",
) -> UncertaintyBudget:
    """Create k₀-NAA uncertainty budget.
    
    Parameters
    ----------
    concentration : float
        Measured concentration
    counting_rel : float
        Counting statistics (default 1%)
    efficiency_rel : float
        Efficiency ratio uncertainty (default 2%)
    k0_rel : float
        k₀ factor uncertainty (default 1.5%)
    q0_rel : float
        Q₀ ratio uncertainty (default 5%)
    f_rel : float
        Thermal/epithermal flux ratio (default 3%)
    alpha_abs : float
        α parameter absolute uncertainty (default 0.01)
    timing_rel : float
        Timing uncertainty (default 0.5%)
    coincidence_rel : float
        Coincidence summing (default 1%)
    units : str
        Concentration units
        
    Returns
    -------
    UncertaintyBudget
        Complete k₀-NAA budget
    """
    budget = UncertaintyBudget(
        measurement=concentration,
        units=units,
        name="k0_naa",
    )
    
    budget.add_relative(
        UncertaintyCategory.COUNTING_STATISTICS,
        counting_rel,
        "Peak area counting statistics",
    )
    budget.add_relative(
        UncertaintyCategory.EFFICIENCY,
        efficiency_rel,
        "Efficiency ratio ε_a/ε_*",
    )
    budget.add_relative(
        UncertaintyCategory.K0_FACTOR,
        k0_rel,
        "k₀ nuclear constant",
    )
    budget.add_relative(
        UncertaintyCategory.Q0_RATIO,
        q0_rel,
        "Q₀ resonance integral ratio",
    )
    budget.add_relative(
        UncertaintyCategory.F_FLUX_RATIO,
        f_rel,
        "Thermal/epithermal flux ratio f",
    )
    budget.add_relative(
        UncertaintyCategory.ALPHA_PARAMETER,
        alpha_abs / abs(concentration) if concentration != 0 else 0,
        "Epithermal flux shape parameter α",
    )
    budget.add_relative(
        UncertaintyCategory.TIMING,
        timing_rel,
        "Decay time corrections",
    )
    budget.add_relative(
        UncertaintyCategory.COINCIDENCE_SUMMING,
        coincidence_rel,
        "True coincidence summing",
    )
    
    budget.compute_total()
    
    return budget


def create_flux_unfolding_budget(
    flux_value: float,
    prior_rel: float = 0.10,
    response_rel: float = 0.03,
    measurement_rel: float = 0.02,
    cross_section_rel: float = 0.05,
    self_shielding_rel: float = 0.01,
    units: str = "n/cm²/s",
) -> UncertaintyBudget:
    """Create flux unfolding uncertainty budget.
    
    Parameters
    ----------
    flux_value : float
        Unfolded flux value (single group or integral)
    prior_rel : float
        Prior spectrum uncertainty (default 10%)
    response_rel : float
        Response matrix uncertainty (default 3%)
    measurement_rel : float
        Reaction rate measurement (default 2%)
    cross_section_rel : float
        Dosimetry cross-section (default 5%)
    self_shielding_rel : float
        Self-shielding correction (default 1%)
    units : str
        Flux units
        
    Returns
    -------
    UncertaintyBudget
        Complete flux unfolding budget
    """
    budget = UncertaintyBudget(
        measurement=flux_value,
        units=units,
        name="flux_unfolding",
    )
    
    budget.add_relative(
        UncertaintyCategory.PRIOR_SPECTRUM,
        prior_rel,
        "Prior/guess spectrum uncertainty",
    )
    budget.add_relative(
        UncertaintyCategory.RESPONSE_MATRIX,
        response_rel,
        "Response matrix (folded XS)",
    )
    budget.add_relative(
        UncertaintyCategory.COUNTING_STATISTICS,
        measurement_rel,
        "Reaction rate measurement",
    )
    budget.add_relative(
        UncertaintyCategory.CROSS_SECTION,
        cross_section_rel,
        "Dosimetry cross-sections",
    )
    budget.add_relative(
        UncertaintyCategory.SELF_SHIELDING,
        self_shielding_rel,
        "Self-shielding corrections",
    )
    
    budget.compute_total()
    
    return budget


__all__ = [
    "UncertaintyCategory",
    "UncertaintyComponent",
    "UncertaintyBudget",
    "combine_budgets",
    "create_activation_budget",
    "create_k0_naa_budget",
    "create_flux_unfolding_budget",
]
