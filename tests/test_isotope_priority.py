from __future__ import annotations

import pytest

from fluxforge.analysis.isotope_priority import (
    IsotopePriorityWeights,
    rank_isotopes_from_activity_review_payload,
)


def _sample_activity_review_payload() -> dict:
    return {
        "schema": "fluxforge.activity_review.v1",
        "isotope_summaries": [
            {
                "nuclide": "Mo-99",
                "line_count": 2,
                "total_net_counts": 9000.0,
                "irradiation_time_activity_Bq": 1200.0,
                "irradiation_time_activity_unc_Bq": 90.0,
                "dose_rate_uSv_h": 55.0,
            },
            {
                "nuclide": "Co-60",
                "line_count": 1,
                "total_net_counts": 12000.0,
                "irradiation_time_activity_Bq": 950.0,
                "irradiation_time_activity_unc_Bq": 220.0,
                "dose_rate_uSv_h": 40.0,
            },
            {
                "nuclide": "Sc-46",
                "line_count": 3,
                "total_net_counts": 5000.0,
                "irradiation_time_activity_Bq": 700.0,
                "irradiation_time_activity_unc_Bq": 45.0,
                "dose_rate_uSv_h": 25.0,
            },
        ],
    }


def test_rank_isotopes_from_activity_review_payload_orders_descending() -> None:
    ranked = rank_isotopes_from_activity_review_payload(_sample_activity_review_payload())

    assert len(ranked) == 3
    assert ranked[0].priority_score >= ranked[1].priority_score >= ranked[2].priority_score
    assert {row.nuclide for row in ranked} == {"Mo-99", "Co-60", "Sc-46"}


def test_rank_isotopes_from_activity_review_payload_respects_isotopes_of_interest() -> None:
    ranked = rank_isotopes_from_activity_review_payload(
        _sample_activity_review_payload(),
        isotopes_of_interest=("sc46", "MO99"),
    )

    assert [row.nuclide for row in ranked] == ["Mo-99", "Sc-46"]


def test_rank_isotopes_from_activity_review_payload_honors_weight_shift() -> None:
    ranked = rank_isotopes_from_activity_review_payload(
        _sample_activity_review_payload(),
        weights=IsotopePriorityWeights(
            activity=0.0,
            detectability=0.7,
            confidence=0.0,
            line_support=0.3,
            dose=0.0,
        ),
    )

    assert ranked[0].nuclide == "Co-60"
    assert ranked[0].priority_score > ranked[-1].priority_score


def test_rank_isotopes_from_activity_review_payload_raises_when_subset_empty() -> None:
    with pytest.raises(ValueError, match="No isotopes remain"):
        rank_isotopes_from_activity_review_payload(
            _sample_activity_review_payload(),
            isotopes_of_interest=("Xe-135",),
        )
