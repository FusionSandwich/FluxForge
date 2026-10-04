"""Reference reproduction and unavailable evidence must not pass raw validation."""

import pytest

from fluxforge.examples.rafm_workflow import (
    build_summary_markdown,
    build_validation_flags,
    enforce_raw_comparison,
    summarize_validation_artifacts,
)


def flags(*, copied=False, peak_rows=None, isotope_rows=None):
    if isotope_rows is None and peak_rows:
        isotope_rows = [
            {"matched": True, "relative_activity_error": 0.0, "activity_en_score": 0.0}
        ]
    return build_validation_flags(
        "RAFM3",
        peak_rows or [],
        isotope_rows or [],
        [],
        [],
        [],
        [],
        {},
        reference_used_for_analysis=copied,
    )


def passing_rows():
    return [
        {
            "matched": True,
            "isotope_match": True,
            "relative_count_error": 0.0,
            "count_en_score": 0.0,
        }
    ]


def test_identical_copied_counts_do_not_validate_raw_recovery():
    result = flags(copied=True, peak_rows=passing_rows())
    assert result["comparison_passed"] is True
    assert result["passed"] is None
    assert result["comparison_basis"] == "reference_reproduction"
    summary = summarize_validation_artifacts(
        [{"sample_id": "copied", "validation": result}]
    )
    assert summary["overall_passed"] is None
    assert summary["unvalidated_samples"] == ["copied"]
    assert summary["reference_reproduction_samples"] == ["copied"]
    assert summary["failing_samples"] == []


def test_independent_raw_estimates_can_pass_or_fail_numeric_thresholds():
    rows = passing_rows()
    assert flags(peak_rows=rows)["passed"] is True
    rows[0]["relative_count_error"] = 2.0
    assert flags(peak_rows=rows)["passed"] is False
    assert flags(copied=True, peak_rows=rows)["comparison_passed"] is False
    assert flags(copied=True, peak_rows=rows)["passed"] is None


def test_absent_reference_or_no_samples_is_not_a_pass():
    result = flags()
    assert result["passed"] is None
    assert result["comparison_basis"] == "not_evaluated"
    assert summarize_validation_artifacts([])["overall_passed"] is None


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf"), "bad"])
def test_missing_or_invalid_metrics_are_not_zero_error(metric):
    rows = passing_rows()
    rows[0]["count_en_score"] = metric
    result = flags(peak_rows=rows)
    assert result["passed"] is None
    assert result["incomplete_comparison_rows"] == 1


def test_failed_raw_sample_takes_precedence_over_unchecked_samples():
    summary = summarize_validation_artifacts(
        [
            {"sample_id": "raw_bad", "validation": {"passed": False}},
            {"sample_id": "unchecked", "validation": {"passed": None}},
        ]
    )
    assert summary["overall_passed"] is False
    assert summary["failing_samples"] == ["raw_bad"]
    assert summary["unvalidated_samples"] == ["unchecked"]


def test_summary_does_not_label_reference_reproduction_overall_pass(tmp_path):
    summary = summarize_validation_artifacts(
        [
            {
                "sample_id": "copied",
                "validation": flags(copied=True, peak_rows=passing_rows()),
            }
        ]
    )
    summary.update(
        n_raw_analyzed=1, n_matched_pairs=1, unmatched_raw=[], unmatched_qg=[]
    )
    target = tmp_path / "summary.md"
    build_summary_markdown(summary, target)
    text = target.read_text()
    assert "Overall raw comparison: not established" in text
    assert "Reference reproduction samples: 1" in text
    assert "Overall pass: yes" not in text


def test_invalid_thresholds_fail():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_validation_flags(
            "RAFM3", passing_rows(), [], [], [], [], [], {"max_en_score": float("nan")}
        )


@pytest.mark.parametrize("unsupported", ["true", 0, 1, None])
def test_unsupported_summary_states_are_unknown(unsupported):
    summary = summarize_validation_artifacts(
        [
            {"sample_id": "unsupported", "validation": {"passed": unsupported}},
        ]
    )
    assert summary["overall_passed"] is None
    assert summary["unvalidated_samples"] == ["unsupported"]
    with pytest.raises(RuntimeError, match="not established"):
        enforce_raw_comparison(summary)


def test_enforced_raw_gate_accepts_only_explicit_success():
    summary = summarize_validation_artifacts(
        [
            {"sample_id": "raw", "validation": flags(peak_rows=passing_rows())},
        ]
    )
    enforce_raw_comparison(summary)
    summary = summarize_validation_artifacts(
        [
            {
                "sample_id": "copied",
                "validation": flags(copied=True, peak_rows=passing_rows()),
            },
        ]
    )
    with pytest.raises(RuntimeError, match="copied"):
        enforce_raw_comparison(summary)


@pytest.mark.parametrize(
    "peak_rows,isotope_rows,missing_domain",
    [
        ([{"matched": False}], [], "counts"),
        (passing_rows(), [], "activities"),
        (
            [],
            [
                {
                    "matched": True,
                    "relative_activity_error": 0.0,
                    "activity_en_score": 0.0,
                }
            ],
            "counts",
        ),
    ],
)
def test_required_comparison_domains_cannot_be_omitted(
    peak_rows, isotope_rows, missing_domain
):
    result = build_validation_flags(
        "RAFM3", peak_rows, isotope_rows, [], [], [], [], {}
    )
    assert result["passed"] is None
    assert missing_domain in result["missing_comparison_domains"]


def test_reference_without_raw_spectrum_prevents_complete_pass():
    summary = summarize_validation_artifacts(
        [
            {"sample_id": "raw", "validation": flags(peak_rows=passing_rows())},
        ],
        unmatched_qg=["unpaired-QG.txt"],
    )
    assert summary["overall_passed"] is None
    assert summary["unvalidated_reference_files"] == ["unpaired-QG.txt"]
    with pytest.raises(RuntimeError, match="unpaired-QG"):
        enforce_raw_comparison(summary)
