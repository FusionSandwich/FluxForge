"""Masking-analysis utilities from activity-review spectrum artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fluxforge.core.planning_models import LineMaskingResult

_EPSILON = 1.0e-12


@dataclass(frozen=True)
class AlternateLineRecommendation:
    """Recommended line choice for reducing masking risk per isotope."""

    nuclide: str
    preferred_line_energy_keV: float
    preferred_masking_score: float
    strongest_line_energy_keV: float
    strongest_line_net_counts: float
    guidance: str

    def to_row(self, rank: int) -> dict[str, Any]:
        return {
            "rank": int(rank),
            "nuclide": self.nuclide,
            "preferred_line_energy_keV": float(self.preferred_line_energy_keV),
            "preferred_masking_score": float(self.preferred_masking_score),
            "strongest_line_energy_keV": float(self.strongest_line_energy_keV),
            "strongest_line_net_counts": float(self.strongest_line_net_counts),
            "guidance": self.guidance,
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_nuclide_label(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _action_for_masking_score(score: float) -> str:
    if score >= 1.0:
        return "avoid"
    if score >= 0.35:
        return "measure_later"
    if score >= 0.15:
        return "alternate_line"
    return "use"


def _line_rows_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows_raw = payload.get("line_results")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("Activity-review payload requires a non-empty 'line_results' list.")

    parsed: list[dict[str, Any]] = []
    for item in rows_raw:
        if not isinstance(item, Mapping):
            continue
        nuclide = str(item.get("nuclide") or item.get("isotope") or "").strip()
        if not nuclide:
            continue
        energy_keV = _safe_float(
            item.get("matched_line_energy_keV", item.get("line_energy_keV", 0.0))
        )
        net_counts = max(_safe_float(item.get("net_counts"), 0.0), 0.0)
        net_counts_unc = max(_safe_float(item.get("net_counts_uncertainty"), 0.0), 0.0)
        if energy_keV <= 0.0 or net_counts <= 0.0:
            continue
        # Back out a conservative background proxy from counting variance.
        background_proxy = max(net_counts_unc * net_counts_unc - net_counts, 0.0)
        parsed.append(
            {
                "nuclide": nuclide,
                "line_energy_keV": float(energy_keV),
                "net_counts": float(net_counts),
                "background_counts": float(background_proxy),
            }
        )

    if not parsed:
        raise ValueError("No valid line rows found in activity-review payload.")
    return parsed


def rank_line_masking_from_activity_review_payload(
    payload: Mapping[str, Any],
    *,
    energy_window_keV: float = 3.0,
    isotopes_of_interest: Sequence[str] | None = None,
    top_n: int | None = None,
) -> tuple[LineMaskingResult, ...]:
    """Rank likely line-masking interactions from activity-review outputs."""

    rows = _line_rows_from_payload(payload)
    selected = tuple(
        str(item).strip() for item in (isotopes_of_interest or ()) if str(item).strip()
    )
    if selected:
        allowed = {_normalize_nuclide_label(item) for item in selected}
        rows = [
            row
            for row in rows
            if _normalize_nuclide_label(str(row["nuclide"])) in allowed
        ]
        if not rows:
            raise ValueError("No line rows remain after isotopes-of-interest filtering.")

    window = max(float(energy_window_keV), 0.1)
    sigma = max(window / 2.355, 1.0e-6)

    results: list[LineMaskingResult] = []
    for target in rows:
        for masker in rows:
            if target is masker:
                continue
            if str(target["nuclide"]) == str(masker["nuclide"]):
                continue

            delta = abs(float(target["line_energy_keV"]) - float(masker["line_energy_keV"]))
            if delta > window:
                continue

            overlap_weight = math.exp(-0.5 * (delta / sigma) ** 2)
            interference_counts = max(float(masker["net_counts"]) * overlap_weight, 0.0)
            continuum_counts = max(
                float(masker["net_counts"]) * 0.15 * max(1.0 - delta / window, 0.0),
                0.0,
            )
            background_counts = max(float(target["background_counts"]), 0.0)
            burial_ratio = (
                interference_counts + continuum_counts + 0.25 * background_counts
            ) / max(float(target["net_counts"]), _EPSILON)
            masking_score = burial_ratio * (1.0 + 0.1 * (window - delta) / window)

            results.append(
                LineMaskingResult(
                    target_nuclide=str(target["nuclide"]),
                    target_line_energy_keV=float(target["line_energy_keV"]),
                    masking_nuclide=str(masker["nuclide"]),
                    masking_line_energy_keV=float(masker["line_energy_keV"]),
                    energy_delta_keV=float(delta),
                    target_signal_counts=float(target["net_counts"]),
                    masking_signal_counts=float(masker["net_counts"]),
                    background_counts=float(background_counts),
                    interference_counts=float(interference_counts),
                    continuum_counts=float(continuum_counts),
                    masking_score=float(masking_score),
                    burial_ratio=float(burial_ratio),
                    recommended_action=_action_for_masking_score(masking_score),
                )
            )

    results.sort(key=lambda item: item.masking_score, reverse=True)
    if top_n is not None and int(top_n) > 0:
        return tuple(results[: int(top_n)])
    return tuple(results)


def summarize_masking_isotopes(
    ranked: Sequence[LineMaskingResult],
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate line-level masking results by masking isotope."""

    totals: dict[str, dict[str, Any]] = {}
    for item in ranked:
        row = totals.setdefault(
            item.masking_nuclide,
            {
                "masking_nuclide": item.masking_nuclide,
                "total_masking_score": 0.0,
                "total_interference_counts": 0.0,
                "total_continuum_counts": 0.0,
                "target_line_count": 0,
                "dominant_target_nuclide": item.target_nuclide,
            },
        )
        row["total_masking_score"] += float(item.masking_score)
        row["total_interference_counts"] += float(item.interference_counts)
        row["total_continuum_counts"] += float(item.continuum_counts)
        row["target_line_count"] += 1

    ranked_rows = list(totals.values())
    ranked_rows.sort(
        key=lambda item: (
            -_safe_float(item.get("total_masking_score")),
            str(item.get("masking_nuclide")),
        )
    )
    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index
    if top_n is not None and int(top_n) > 0:
        return ranked_rows[: int(top_n)]
    return ranked_rows


def recommend_alternate_lines(
    payload: Mapping[str, Any],
    ranked: Sequence[LineMaskingResult],
    *,
    top_n: int | None = None,
) -> tuple[AlternateLineRecommendation, ...]:
    """Pick lower-masking line alternatives per isotope where available."""

    rows = _line_rows_from_payload(payload)
    score_by_target_line: dict[tuple[str, float], float] = {}
    for item in ranked:
        key = (item.target_nuclide, round(float(item.target_line_energy_keV), 6))
        score_by_target_line[key] = score_by_target_line.get(key, 0.0) + float(
            item.masking_score
        )

    by_nuclide: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_nuclide.setdefault(str(row["nuclide"]), []).append(row)

    recommendations: list[AlternateLineRecommendation] = []
    for nuclide, line_rows in by_nuclide.items():
        strongest = max(line_rows, key=lambda item: _safe_float(item.get("net_counts")))

        preferred = min(
            line_rows,
            key=lambda item: score_by_target_line.get(
                (str(item["nuclide"]), round(float(item["line_energy_keV"]), 6)),
                0.0,
            ),
        )
        preferred_score = score_by_target_line.get(
            (str(preferred["nuclide"]), round(float(preferred["line_energy_keV"]), 6)),
            0.0,
        )

        if abs(float(preferred["line_energy_keV"]) - float(strongest["line_energy_keV"])) > 1.0e-9:
            guidance = "switch_to_preferred_line"
        elif preferred_score >= 0.35:
            guidance = "defer_to_later_cooldown"
        else:
            guidance = "keep_current_line"

        recommendations.append(
            AlternateLineRecommendation(
                nuclide=str(nuclide),
                preferred_line_energy_keV=float(preferred["line_energy_keV"]),
                preferred_masking_score=float(preferred_score),
                strongest_line_energy_keV=float(strongest["line_energy_keV"]),
                strongest_line_net_counts=float(strongest["net_counts"]),
                guidance=guidance,
            )
        )

    recommendations.sort(
        key=lambda item: (
            -float(item.preferred_masking_score),
            str(item.nuclide),
        )
    )
    if top_n is not None and int(top_n) > 0:
        return tuple(recommendations[: int(top_n)])
    return tuple(recommendations)


def serialize_masking_review(
    ranked: Sequence[LineMaskingResult],
    *,
    source_path: str,
    isotopes_of_interest: Sequence[str],
    energy_window_keV: float,
    isotope_summary_rows: Sequence[Mapping[str, Any]],
    recommendations: Sequence[AlternateLineRecommendation],
) -> dict[str, Any]:
    """Serialize masking analysis outputs to an artifact payload."""

    line_rows = [item.to_row(index) for index, item in enumerate(ranked, start=1)]
    return {
        "schema": "fluxforge.masking_review.v1",
        "source_activity_review": str(source_path),
        "isotopes_of_interest": [str(item) for item in isotopes_of_interest],
        "energy_window_keV": float(energy_window_keV),
        "line_masking_results": line_rows,
        "masking_isotope_ranking": [dict(row) for row in isotope_summary_rows],
        "alternate_line_recommendations": [
            item.to_row(index) for index, item in enumerate(recommendations, start=1)
        ],
    }
