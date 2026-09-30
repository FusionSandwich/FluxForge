"""Reaction-rate uncertainty budgets with shared-component covariance.

Each observation carries named relative 1-sigma components. A component with a
``correlation_group`` is fully correlated between all rows that share the same
group key (for example one detector efficiency calibration, one irradiation
history or one nuclide's half-life); a component with no group is independent.

    C_ij = R_i R_j * sum_k r_ik r_jk [group_ik == group_jk != None or i == j]

Components that a complete budget needs but that were not supplied are listed
in ``missing`` so they are never mistaken for zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Components expected for an activation reaction rate (ASTM E261 terms).
REQUIRED_COMPONENTS = (
    "activity",
    "detector_efficiency",
    "gamma_yield",
    "half_life",
    "target_mass",
    "isotopic_abundance",
    "irradiation_history",
)


@dataclass(frozen=True)
class UncertaintyComponent:
    name: str
    relative: float
    correlation_group: Optional[str] = None
    source: str = ""
    covers: tuple[str, ...] = ()
    sensitivity_sign: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative) or self.relative < 0:
            raise ValueError(
                f"Component {self.name} needs a finite non-negative relative value"
            )
        object.__setattr__(self, "covers", tuple(self.covers))
        if not self.name or self.sensitivity_sign not in (-1, 1):
            raise ValueError("Component needs a name and sensitivity sign +/-1")
        if len(set(self.covers)) != len(self.covers) or self.name in self.covers:
            raise ValueError(
                "Component coverage must not repeat its own or another name"
            )

    @classmethod
    def from_input(
        cls,
        name,
        standard_uncertainty,
        log_sensitivity,
        *,
        source,
        correlation_group=None,
        covers=(),
    ):
        """Propagate u(x) * d(log rate)/dx, with units declared by the source.

        The caller supplies a measured uncertainty and signed sensitivity, not
        an assumed missing component. Retain sign for cross-row covariance.
        """
        if (
            not math.isfinite(standard_uncertainty)
            or standard_uncertainty < 0
            or not math.isfinite(log_sensitivity)
        ):
            raise ValueError(
                "Input uncertainty/sensitivity must be finite; uncertainty nonnegative"
            )
        effect = standard_uncertainty * log_sensitivity
        return cls(
            name,
            abs(effect),
            correlation_group,
            source,
            tuple(covers),
            -1 if effect < 0 else 1,
        )


@dataclass
class RateUncertaintyBudget:
    row_id: str
    rate: float
    components: List[UncertaintyComponent] = field(default_factory=list)
    required: Sequence[str] = REQUIRED_COMPONENTS

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate) or self.rate < 0:
            raise ValueError("Reaction rate must be finite and non-negative")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Uncertainty component names must be unique within a row")
        accounted = set()
        for component in self.components:
            component.__post_init__()
            coverage = {component.name, *component.covers}
            if accounted.intersection(coverage):
                raise ValueError(
                    "Uncertainty component coverage overlaps; avoid double counting"
                )
            accounted.update(coverage)

    @property
    def missing(self) -> List[str]:
        names = {
            name
            for component in self.components
            for name in (component.name, *component.covers)
        }
        return [name for name in self.required if name not in names]

    @property
    def complete(self) -> bool:
        names = {name for c in self.components for name in (c.name, *c.covers)}
        return set(REQUIRED_COMPONENTS).union(self.required).issubset(names) and all(
            c.source.strip() for c in self.components
        )

    def require_complete(self) -> None:
        self.__post_init__()
        if not self.complete:
            raise ValueError(
                f"{self.row_id}: incomplete source/component uncertainty budget"
            )

    @property
    def total_relative(self) -> float:
        return math.sqrt(sum(c.relative**2 for c in self.components))

    @property
    def total_absolute(self) -> float:
        return abs(self.rate) * self.total_relative

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "row_id": self.row_id,
            "rate": self.rate,
            "total_relative": self.total_relative,
            "missing_components": ";".join(self.missing),
            "component_coverage_complete": self.complete,
            "scientific_admission": False,
        }
        for component in self.components:
            row[f"{component.name}_relative"] = component.relative
            row[f"{component.name}_group"] = (
                component.correlation_group or "independent"
            )
            row[f"{component.name}_source"] = component.source
            row[f"{component.name}_covers"] = ";".join(component.covers)
            row[f"{component.name}_sensitivity_sign"] = component.sensitivity_sign
        return row


def floor_as_component(
    name: str, base_relative: float, floor_relative: float, source: str
) -> Optional[UncertaintyComponent]:
    """Express ``max(total, floor)`` as an explicit additive component."""
    if floor_relative <= base_relative:
        return None
    return UncertaintyComponent(
        name, math.sqrt(floor_relative**2 - base_relative**2), None, source
    )


def rate_covariance(
    budgets: Sequence[RateUncertaintyBudget], *, require_complete: bool = False
) -> np.ndarray:
    """Absolute covariance matrix of the rates in ``budgets``."""
    n = len(budgets)
    if len({b.row_id for b in budgets}) != n:
        raise ValueError("Rate budget observation identities must be unique")
    for budget in budgets:
        budget.__post_init__()
        if require_complete:
            budget.require_complete()
    cov = np.zeros((n, n))
    for i, bi in enumerate(budgets):
        for j, bj in enumerate(budgets):
            total = 0.0
            for ci in bi.components:
                for cj in bj.components:
                    if ci.name != cj.name:
                        continue
                    same = i == j or (
                        ci.correlation_group is not None
                        and ci.correlation_group == cj.correlation_group
                    )
                    if same:
                        total += (
                            ci.relative
                            * cj.relative
                            * ci.sensitivity_sign
                            * cj.sensitivity_sign
                        )
            cov[i, j] = total * bi.rate * bj.rate
    return cov


def budget_table(budgets: Iterable[RateUncertaintyBudget]) -> List[Dict[str, object]]:
    return [budget.as_row() for budget in budgets]


__all__ = [
    "REQUIRED_COMPONENTS",
    "RateUncertaintyBudget",
    "UncertaintyComponent",
    "budget_table",
    "floor_as_component",
    "rate_covariance",
]
