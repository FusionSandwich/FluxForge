"""Reaction-rate uncertainty budgets with shared-component covariance.

Each observation carries named relative 1-sigma components. Scalar terms sharing
name/group are fully correlated. Full input-covariance terms sharing a source
group propagate signed row Jacobians and can be partially correlated. A scalar
component with no group is independent.

    C_rate = sum_source (diag(R) J L) (diag(R) J L)^T, where L L^T = C_source

Components that a complete budget needs but that were not supplied are listed
in ``missing`` so they are never mistaken for zero.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from fluxforge.uncertainty.covariance import covariance_matrix, covariance_factor

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
    input_covariance: tuple[tuple[float, ...], ...] = ()
    log_sensitivities: tuple[float, ...] = ()
    input_names: tuple[str, ...] = ()
    input_units: tuple[str, ...] = ()
    assumed: bool = False
    uncertainty_scope: str = "unspecified"

    def __post_init__(self) -> None:
        if self.uncertainty_scope not in {
            "unspecified",
            "reported_total_unknown",
            "itemized",
            "conditional",
            "marginalized",
        }:
            raise ValueError(
                "Declare itemized/conditional/marginalized or unknown uncertainty scope"
            )
        if type(self.assumed) is not bool:
            raise ValueError("Component assumed flag must be boolean")
        if len(self.input_covariance):
            size = len(self.input_covariance)
            raw_cov = np.asarray(self.input_covariance, dtype=float)
            covariance_matrix(raw_cov, size, "input covariance")
            sensitivities = np.asarray(self.log_sensitivities, dtype=float)
            if (
                sensitivities.shape != (size,)
                or not np.all(np.isfinite(sensitivities))
                or len(self.input_names) != size
                or len(set(self.input_names)) != size
                or len(self.input_units) != size
                or any(
                    not isinstance(x, str) or not x.strip()
                    for x in (*self.input_names, *self.input_units)
                )
                or not self.correlation_group
            ):
                raise ValueError(
                    "Covariance needs aligned named/unit inputs, finite sensitivities and source group"
                )
            expected = math.hypot(*(sensitivities @ covariance_factor(raw_cov)))
            if not math.isclose(self.relative, expected, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(
                    "Component relative uncertainty disagrees with input covariance"
                )
            object.__setattr__(
                self,
                "input_covariance",
                tuple(tuple(float(v) for v in row) for row in raw_cov),
            )
            object.__setattr__(
                self, "log_sensitivities", tuple(float(v) for v in sensitivities)
            )
            object.__setattr__(self, "input_names", tuple(self.input_names))
            object.__setattr__(self, "input_units", tuple(self.input_units))
        elif self.log_sensitivities or self.input_names or self.input_units:
            raise ValueError("Covariance input metadata requires an input covariance")
        if not math.isfinite(self.relative) or self.relative < 0:
            raise ValueError(
                f"Component {self.name} needs a finite non-negative relative value"
            )
        object.__setattr__(self, "covers", tuple(self.covers))
        if self.source is None:
            object.__setattr__(self, "source", "")
        elif not isinstance(self.source, str):
            raise ValueError("Component source must be a string or absent")
        if self.correlation_group is not None and (
            not isinstance(self.correlation_group, str)
            or not self.correlation_group.strip()
        ):
            raise ValueError("Correlation group must be a nonempty string or None")
        if any(not isinstance(name, str) or not name.strip() for name in self.covers):
            raise ValueError("Component coverage needs nonempty names")
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.sensitivity_sign not in (-1, 1)
        ):
            raise ValueError("Component needs a name and sensitivity sign +/-1")
        if len(set(self.covers)) != len(self.covers) or self.name in self.covers:
            raise ValueError(
                "Component coverage must not repeat its own or another name"
            )

    @property
    def covariance_binding(self) -> str:
        if not self.input_covariance:
            return ""
        return hashlib.sha256(
            json.dumps(
                [
                    self.input_names,
                    self.input_units,
                    self.input_covariance,
                    self.uncertainty_scope,
                ],
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    @classmethod
    def from_covariance(
        cls,
        name,
        input_covariance,
        log_sensitivities,
        *,
        input_names,
        input_units,
        source,
        correlation_group,
        covers=(),
        assumed=False,
        uncertainty_scope="unspecified",
    ):
        """Propagate a full source covariance with signed row-specific Jacobian.

        Inputs retain their declared units. For rate R, J = d(log R)/dx;
        C_rate[i,j] = R_i R_j J_i C_source J_j^T. No covariance is invented.
        """
        if len(input_names) == 0:
            raise ValueError("Covariance requires named inputs")
        covariance_matrix(input_covariance, len(input_names), "input covariance")
        sensitivity = np.asarray(log_sensitivities, dtype=float)
        if sensitivity.shape != (len(input_names),) or not np.all(
            np.isfinite(sensitivity)
        ):
            raise ValueError("Covariance sensitivity must align with named inputs")
        relative = float(
            math.hypot(*(sensitivity @ covariance_factor(input_covariance)))
        )
        return cls(
            name,
            relative,
            correlation_group,
            source,
            tuple(covers),
            1,
            tuple(tuple(row) for row in input_covariance),
            tuple(sensitivity),
            tuple(input_names),
            tuple(input_units),
            assumed,
            uncertainty_scope,
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
        assumed=False,
        uncertainty_scope="unspecified",
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
            assumed=assumed,
            uncertainty_scope=uncertainty_scope,
        )


@dataclass
class RateUncertaintyBudget:
    row_id: str
    rate: float
    components: List[UncertaintyComponent] = field(default_factory=list)
    required: Sequence[str] = REQUIRED_COMPONENTS
    diagnostic_assumptions: List[str] = field(default_factory=list)
    irradiation_log_binding: Optional[Dict[str, object]] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate) or self.rate < 0:
            raise ValueError("Reaction rate must be finite and non-negative")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Uncertainty component names must be unique within a row")
        accounted = set()
        covariance_groups = set()
        for component in self.components:
            component.__post_init__()
            coverage = {component.name, *component.covers}
            if accounted.intersection(coverage):
                raise ValueError(
                    "Uncertainty component coverage overlaps; avoid double counting"
                )
            accounted.update(coverage)
            if component.input_covariance:
                if component.correlation_group in covariance_groups:
                    raise ValueError(
                        "Represent each shared input covariance once per row, with joint coverage"
                    )
                covariance_groups.add(component.correlation_group)

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
        return math.hypot(*(c.relative for c in self.components))

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
            "diagnostic_assumptions": ";".join(self.diagnostic_assumptions),
        }
        for component in self.components:
            row[f"{component.name}_relative"] = component.relative
            row[f"{component.name}_group"] = (
                component.correlation_group or "independent"
            )
            row[f"{component.name}_source"] = component.source
            row[f"{component.name}_covers"] = ";".join(component.covers)
            row[f"{component.name}_sensitivity_sign"] = component.sensitivity_sign
            row[f"{component.name}_assumed"] = component.assumed
            row[f"{component.name}_uncertainty_scope"] = component.uncertainty_scope
            row[f"{component.name}_covariance_binding"] = component.covariance_binding
            row[f"{component.name}_input_names"] = ";".join(component.input_names)
            row[f"{component.name}_input_units"] = ";".join(component.input_units)
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
    groups = {}
    group_kinds = {}
    for budget in budgets:
        for component in budget.components:
            if component.correlation_group is None:
                continue
            kind = bool(component.input_covariance)
            group = component.correlation_group
            if group in group_kinds and group_kinds[group] != kind:
                raise ValueError(
                    "A shared source cannot mix scalar and full covariance declarations; use a joint input block"
                )
            group_kinds[group] = kind
            key = (
                "input covariance" if component.input_covariance else component.name,
                component.correlation_group,
            )
            binding = component.covariance_binding
            if key in groups and groups[key] != binding:
                raise ValueError(
                    "Shared source covariance/parameter binding disagrees between rows"
                )
            groups[key] = binding
    # Source factors form a Gram matrix, preserving PSD for singular blocks and
    # cancellation without adding a variance floor or diagonal jitter.
    source_effects = {}
    for i, budget in enumerate(budgets):
        for component in budget.components:
            if component.input_covariance:
                key = ("vector", component.correlation_group)
                effect = np.asarray(component.log_sensitivities) @ covariance_factor(
                    component.input_covariance
                )
            else:
                key = (
                    ("scalar", component.name, component.correlation_group)
                    if component.correlation_group is not None
                    else ("independent", i, component.name)
                )
                effect = np.array([component.relative * component.sensitivity_sign])
            if key not in source_effects:
                source_effects[key] = np.zeros((n, len(effect)))
            source_effects[key][i] += budget.rate * effect
    cov = np.zeros((n, n))
    for effects in source_effects.values():
        cov += effects @ effects.T
    if not np.all(np.isfinite(cov)):
        raise ValueError("Propagated rate covariance must be finite")
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
