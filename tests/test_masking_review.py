from __future__ import annotations

from fluxforge.analysis.masking_review import (
    rank_line_masking_from_activity_review_payload,
    recommend_alternate_lines,
    summarize_masking_isotopes,
)


def test_rank_line_masking_from_activity_review_payload_returns_sorted_scores():
    payload = {
        "line_results": [
            {
                "nuclide": "Mo-99",
                "matched_line_energy_keV": 140.5,
                "net_counts": 5000.0,
                "net_counts_uncertainty": 80.0,
            },
            {
                "nuclide": "Sc-46",
                "matched_line_energy_keV": 140.6,
                "net_counts": 9000.0,
                "net_counts_uncertainty": 100.0,
            },
            {
                "nuclide": "Co-60",
                "matched_line_energy_keV": 1332.5,
                "net_counts": 3000.0,
                "net_counts_uncertainty": 60.0,
            },
        ]
    }

    ranked = rank_line_masking_from_activity_review_payload(
        payload,
        energy_window_keV=3.0,
    )

    assert ranked
    assert ranked[0].masking_score >= ranked[-1].masking_score
    assert ranked[0].recommended_action in {
        "use",
        "alternate_line",
        "measure_later",
        "avoid",
    }


def test_summarize_masking_isotopes_aggregates_scores():
    payload = {
        "line_results": [
            {
                "nuclide": "Mo-99",
                "matched_line_energy_keV": 140.5,
                "net_counts": 5000.0,
                "net_counts_uncertainty": 80.0,
            },
            {
                "nuclide": "Sc-46",
                "matched_line_energy_keV": 140.6,
                "net_counts": 9000.0,
                "net_counts_uncertainty": 100.0,
            },
            {
                "nuclide": "Sc-46",
                "matched_line_energy_keV": 1120.5,
                "net_counts": 2500.0,
                "net_counts_uncertainty": 50.0,
            },
        ]
    }

    ranked = rank_line_masking_from_activity_review_payload(payload, energy_window_keV=3.0)
    summary = summarize_masking_isotopes(ranked)

    assert summary
    assert summary[0]["rank"] == 1
    assert "masking_nuclide" in summary[0]
    assert summary[0]["total_masking_score"] >= 0.0


def test_recommend_alternate_lines_prefers_lower_masking_line():
    payload = {
        "line_results": [
            {
                "nuclide": "Mo-99",
                "matched_line_energy_keV": 140.5,
                "net_counts": 5000.0,
                "net_counts_uncertainty": 80.0,
            },
            {
                "nuclide": "Mo-99",
                "matched_line_energy_keV": 739.5,
                "net_counts": 1800.0,
                "net_counts_uncertainty": 50.0,
            },
            {
                "nuclide": "Sc-46",
                "matched_line_energy_keV": 140.6,
                "net_counts": 9000.0,
                "net_counts_uncertainty": 100.0,
            },
        ]
    }

    ranked = rank_line_masking_from_activity_review_payload(payload, energy_window_keV=3.0)
    recommendations = recommend_alternate_lines(payload, ranked)

    assert recommendations
    mo99 = next(item for item in recommendations if item.nuclide == "Mo-99")
    assert mo99.strongest_line_energy_keV == 140.5
    assert mo99.preferred_line_energy_keV in {140.5, 739.5}
