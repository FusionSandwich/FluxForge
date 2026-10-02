"""Separate peak area from conversion without claiming independent EOI truth."""

from fluxforge.examples.rafm_workflow import build_measurement_time_audit


def row(count_ratio, activity_ratio, bucket="count_parity_failure"):
    return {
        "reference_isotope": "A",
        "reference_energy_keV": 100.0,
        "raw_net_counts": 1000 * count_ratio,
        "reference_net_counts": 1000,
        "raw_line_activity_bq": 100 * activity_ratio,
        "reference_line_activity_bq": 100,
        "isotope_match": True,
        "diagnostic_bucket": bucket,
    }


def test_area_error_alone_does_not_become_efficiency_error():
    result = build_measurement_time_audit([row(2, 2)], [{"matched": True}], [])
    assert result["categories"] == ["peak_area_limited"]
    assert (
        result["conversion_diagnostics"][0]["activity_conversion_ratio_raw_over_report"]
        == 1
    )
    assert result["accuracy_qualified"] is False
    assert result["eoi_parity"].startswith("not_evaluated")


def test_conversion_and_area_discrepancies_can_coexist():
    result = build_measurement_time_audit([row(2, 4)], [{"matched": True}], [])
    assert set(result["categories"]) == {
        "peak_area_limited",
        "activity_conversion_or_efficiency_limited",
    }
    assert (
        result["conversion_diagnostics"][0]["activity_conversion_ratio_raw_over_report"]
        == 2
    )


def test_report_inconsistency_and_ambiguity_are_not_efficiency_measurements():
    line = row(2, 4, "assignment_ambiguous")
    line["assignment_ambiguous"] = True
    result = build_measurement_time_audit(
        [line], [{"matched": True}], [{"flag_internal_inconsistency": True}]
    )
    assert set(result["categories"]) == {
        "library_or_assignment_limited",
        "export_data_limited",
    }
    assert result["conversion_diagnostics"] == []


def test_summary_exposes_comparison_basis_and_measurement_stage(tmp_path):
    from fluxforge.examples.rafm_workflow import build_summary_markdown

    audit = build_measurement_time_audit([row(2, 2)], [{"matched": True}], [])
    audit["comparison_basis"] = "raw_estimate_vs_report"
    summary = {
        "overall_passed": False,
        "n_raw_analyzed": 1,
        "n_matched_pairs": 1,
        "unmatched_raw": [],
        "unmatched_qg": [],
        "failing_samples": ["B24h"],
        "measurement_time_audits": {"B24h": audit},
    }
    output = tmp_path / "summary.md"
    build_summary_markdown(summary, output)
    text = output.read_text()
    assert "basis=raw_estimate_vs_report" in text
    assert "stage=measurement_time" in text
    assert "peak_area_limited" in text
    assert "Independent EOI parity is not evaluated" in text


def test_malformed_conversion_is_unknown_not_a_crash_or_pass():
    line = row(1, 1, "matched")
    line["raw_line_activity_bq"] = "unavailable"
    audit = build_measurement_time_audit([line], [{"matched": True}], [])
    assert "activity_evidence_incomplete" in audit["categories"]
    assert audit["conversion_diagnostics"] == []
