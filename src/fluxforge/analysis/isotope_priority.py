"""Isotope-priority ranking from activity-review gamma-spectrum artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_EPSILON = 1.0e-12


@dataclass(frozen=True)
class IsotopePriorityWeights:
    """Weights for isotope-priority score components."""

    activity: float = 0.35
    detectability: float = 0.25
    confidence: float = 0.20
    line_support: float = 0.10
    dose: float = 0.10


@dataclass(frozen=True)
class IsotopePriorityScore:
    """One ranked isotope with component diagnostics."""

    nuclide: str
    priority_score: float
    irradiation_time_activity_bq: float
    irradiation_time_uncertainty_bq: float
    relative_uncertainty: float
    total_net_counts: float
    line_count: int
    dose_rate_uSv_h: float
    activity_component: float
    detectability_component: float
    confidence_component: float
    line_support_component: float
    dose_component: float

    def to_row(self, rank: int) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "nuclide": self.nuclide,
            "priority_score": float(self.priority_score),
            "irradiation_time_activity_Bq": float(self.irradiation_time_activity_bq),
            "irradiation_time_activity_unc_Bq": float(self.irradiation_time_uncertainty_bq),
            "relative_uncertainty": float(self.relative_uncertainty),
            "total_net_counts": float(self.total_net_counts),
            "line_count": int(self.line_count),
            "dose_rate_uSv_h": float(self.dose_rate_uSv_h),
            "activity_component": float(self.activity_component),
            "detectability_component": float(self.detectability_component),
            "confidence_component": float(self.confidence_component),
            "line_support_component": float(self.line_support_component),
            "dose_component": float(self.dose_component),
        }


def _normalize_nuclide_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _parse_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _isotope_rows_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows_raw = payload.get("isotope_summaries")
    if not isinstance(rows_raw, list) or len(rows_raw) == 0:
        raise ValueError(
            "Activity-review payload requires a non-empty 'isotope_summaries' list."
        )

    parsed: list[dict[str, Any]] = []
    for item in rows_raw:
        if not isinstance(item, Mapping):
            continue
        nuclide = str(item.get("nuclide") or item.get("isotope") or "").strip()
        if not nuclide:
            continue
        activity_bq = _parse_float(
            item.get("irradiation_time_activity_Bq", item.get("activity_Bq", 0.0)),
            default=0.0,
        )
        unc_bq = _parse_float(
            item.get(
                "irradiation_time_activity_unc_Bq",
                item.get("activity_unc_Bq", item.get("count_time_activity_unc_Bq", 0.0)),
            ),
            default=0.0,
        )
        net_counts = _parse_float(item.get("total_net_counts", 0.0), default=0.0)
        line_count = _parse_int(item.get("line_count", 1), default=1)
        dose_rate = _parse_float(item.get("dose_rate_uSv_h", 0.0), default=0.0)

        parsed.append(
            {
                "nuclide": nuclide,
                "activity_bq": max(activity_bq, 0.0),
                "unc_bq": max(unc_bq, 0.0),
                "net_counts": max(net_counts, 0.0),
                "line_count": max(line_count, 0),
                "dose_rate_uSv_h": max(dose_rate, 0.0),
            }
        )

    if len(parsed) == 0:
        raise ValueError("No valid isotope summaries found in activity-review payload.")
    return parsed


def rank_isotopes_from_activity_review_payload(
    payload: Mapping[str, Any],
    *,
    weights: IsotopePriorityWeights | None = None,
    isotopes_of_interest: Sequence[str] | None = None,
    top_n: int | None = None,
) -> tuple[IsotopePriorityScore, ...]:
    """Rank isotopes by a composite importance score from activity-review outputs."""

    rows = _isotope_rows_from_payload(payload)
    selected = tuple(
        str(item).strip() for item in (isotopes_of_interest or ()) if str(item).strip()
    )
    if len(selected) > 0:
        allowed = {_normalize_nuclide_label(item) for item in selected}
        rows = [
            row
            for row in rows
            if _normalize_nuclide_label(str(row["nuclide"])) in allowed
        ]
        if len(rows) == 0:
            raise ValueError(
                "No isotopes remain after applying isotopes-of-interest filtering."
            )

    resolved_weights = weights or IsotopePriorityWeights()
    max_activity = max(float(row["activity_bq"]) for row in rows)
    max_counts = max(float(row["net_counts"]) for row in rows)
    max_line_count = max(int(row["line_count"]) for row in rows)
    max_dose = max(float(row["dose_rate_uSv_h"]) for row in rows)

    ranked: list[IsotopePriorityScore] = []
    for row in rows:
        activity = float(row["activity_bq"])
        uncertainty = float(row["unc_bq"])
        rel_unc = uncertainty / max(activity, _EPSILON)

        activity_component = activity / max(max_activity, _EPSILON)
        detectability_component = float(row["net_counts"]) / max(max_counts, _EPSILON)
        line_support_component = float(row["line_count"]) / max(float(max_line_count), 1.0)
        dose_component = (
            float(row["dose_rate_uSv_h"]) / max(max_dose, _EPSILON)
            if max_dose > 0.0
            else 0.0
        )
        confidence_component = max(0.0, 1.0 - min(rel_unc, 2.0) / 2.0)

        priority_score = (
            float(resolved_weights.activity) * activity_component
            + float(resolved_weights.detectability) * detectability_component
            + float(resolved_weights.confidence) * confidence_component
            + float(resolved_weights.line_support) * line_support_component
            + float(resolved_weights.dose) * dose_component
        )

        ranked.append(
            IsotopePriorityScore(
                nuclide=str(row["nuclide"]),
                priority_score=float(priority_score),
                irradiation_time_activity_bq=float(activity),
                irradiation_time_uncertainty_bq=float(uncertainty),
                relative_uncertainty=float(rel_unc),
                total_net_counts=float(row["net_counts"]),
                line_count=int(row["line_count"]),
                dose_rate_uSv_h=float(row["dose_rate_uSv_h"]),
                activity_component=float(activity_component),
                detectability_component=float(detectability_component),
                confidence_component=float(confidence_component),
                line_support_component=float(line_support_component),
                dose_component=float(dose_component),
            )
        )

    ranked.sort(key=lambda item: item.priority_score, reverse=True)
    if top_n is not None and int(top_n) > 0:
        return tuple(ranked[: int(top_n)])
    return tuple(ranked)


def serialize_isotope_priority_ranking(
    ranked: Sequence[IsotopePriorityScore],
    *,
    source_path: str,
    isotopes_of_interest: Sequence[str],
    weights: IsotopePriorityWeights,
) -> dict[str, Any]:
    """Serialize isotope-priority ranking to a CLI artifact shape."""

    rows = [item.to_row(index) for index, item in enumerate(ranked, start=1)]
    return {
        "schema": "fluxforge.isotope_priority.v1",
        "source_activity_review": str(source_path),
        "isotopes_of_interest": [str(item) for item in isotopes_of_interest],
        "weights": {
            "activity": float(weights.activity),
            "detectability": float(weights.detectability),
            "confidence": float(weights.confidence),
            "line_support": float(weights.line_support),
            "dose": float(weights.dose),
        },
        "ranked_isotopes": rows,
    }
