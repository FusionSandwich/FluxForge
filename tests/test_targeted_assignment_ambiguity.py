"""Extra isotope hypotheses must not duplicate or erase a measured peak."""

import numpy as np
import pytest

from fluxforge.analysis.flux_wire_analysis import (
    GammaLine,
    IdentifiedPeak,
    analyze_raw_spectrum_targeted,
    combine_peak_activities,
)
from fluxforge.examples.rafm_workflow import (
    merge_detected_and_targeted_peaks,
    build_validation_flags,
    match_peak,
    peak_to_dict,
    TimingInfo,
)
from fluxforge.io.flux_wire import FluxWireData
from fluxforge.io.spe import GammaSpectrum


def specimen():
    channels = np.arange(201)
    counts = 40.0 + 300.0 * np.exp(-0.5 * ((channels - 100.0) / 1.5) ** 2)
    return FluxWireData(
        energy_calibration=[0.0, 1.0],
        resolution=[3.5325, 0.0],
        spectrum=GammaSpectrum(
            channels=channels, counts=counts, live_time=100.0, real_time=100.0
        ),
    )


def extract(lines):
    return analyze_raw_spectrum_targeted(
        specimen(),
        lines,
        background_subtract=False,
        peak_threshold=3.0,
        counting_method="iec_tiered",
    )


@pytest.mark.parametrize("other_isotope", ["B", "A"])
def test_nearby_hypotheses_preserve_one_area_and_withhold_activity(other_isotope):
    alone = extract([GammaLine(100.0, 0.9, "A")])[0]
    crowded = extract(
        [
            GammaLine(99.0, 0.002, other_isotope),
            GammaLine(100.0, 0.9, "A"),
            GammaLine(101.0, 0.004, "C"),
        ]
    )
    assert len(crowded) == 1
    peak = crowded[0]
    assert peak.net_counts == pytest.approx(alone.net_counts, rel=1e-5)
    assert peak.net_counts_unc == pytest.approx(alone.net_counts_unc, rel=1e-5)
    assert peak.assignment_ambiguous
    assert len(peak.assignment_candidates) == 3
    assert peak.isotope is None
    assert peak.activity_estimation_state == "withheld_ambiguous_assignment"
    assert peak.to_dict()["activity_bq"] is None
    assert combine_peak_activities([peak]) == {}


def test_unresolved_same_isotope_transitions_also_withhold_activity():
    peak = extract([GammaLine(100.0, 0.9, "A"), GammaLine(101.0, 0.004, "A")])[0]
    assert peak.assignment_ambiguous
    assert peak.activity_estimation_state == "withheld_ambiguous_assignment"


def test_equal_energy_known_and_unknown_labels_are_preserved_without_sort_error():
    peak = extract([GammaLine(100.0, 0.9, "A"), GammaLine(100.0, 0.002, None)])[0]
    assert peak.assignment_ambiguous
    assert {line.isotope for line in peak.assignment_candidates} == {"A", None}


def test_single_unknown_label_cannot_estimate_isotope_activity():
    peak = extract([GammaLine(100.0, 0.9, None)])[0]
    assert peak.assignment_ambiguous
    assert peak.to_dict()["activity_bq"] is None


def test_ambiguity_cannot_become_activity_or_eoi_by_stale_numeric_fields():
    peak = extract([GammaLine(100.0, 0.9, "A"), GammaLine(101.0, 0.004, "B")])[0]
    peak.isotope = "A"
    peak.activity_bq, peak.activity_unc_bq = 100.0, 1.0
    timing = TimingInfo("RAFM3", True, None, None, None, 100.0, None, "100s", None)
    payload = peak_to_dict(peak, {"A": 1000.0}, timing, 100.0)
    assert payload["activity_bq"] is None
    assert payload["eoi_activity_bq"] is None
    assert combine_peak_activities([peak]) == {}


def test_merge_cannot_replace_ambiguity_with_exploratory_label():
    detected = IdentifiedPeak(100, 100.0, 1200, 30, 1500, 40, 3.5, 40, isotope="A")
    targeted = IdentifiedPeak(100, 100.1, 1100, 50, 1500, 40, 3.5, 22)
    targeted.assignment_ambiguous = True
    targeted.assignment_candidates = [
        GammaLine(99.0, 0.002, "B"),
        GammaLine(100.0, 0.9, "A"),
    ]
    targeted.activity_estimation_state = "withheld_ambiguous_assignment"
    merged = merge_detected_and_targeted_peaks([detected], [targeted], {})
    assert merged == [targeted]
    assert merged[0].assignment_ambiguous


def test_ambiguous_reference_assignment_cannot_pass_even_with_identical_counts():
    peak = extract([GammaLine(100.0, 0.9, "A"), GammaLine(101.0, 0.004, "B")])[0]
    matched, isotope_match = match_peak(
        {"energy_keV": 100.0, "isotope": "A"}, [peak], {}
    )
    assert matched is peak
    assert not isotope_match
    rows = [
        {
            "matched": True,
            "isotope_match": True,
            "assignment_ambiguous": True,
            "relative_count_error": 0.0,
            "count_en_score": 0.0,
        }
    ]
    result = build_validation_flags(
        "RAFM3",
        rows,
        [{"matched": True, "relative_activity_error": 0.0, "activity_en_score": 0.0}],
        [],
        [],
        [],
        [],
        {},
    )
    assert result["passed"] is False
    assert result["ambiguous_reference_assignments"] == 1


def test_reference_match_cannot_bypass_ambiguity_with_nearby_exploratory_label():
    targeted = extract([GammaLine(100.0, 0.9, "A"), GammaLine(101.0, 0.004, "B")])[0]
    detected = IdentifiedPeak(102, 102.2, 1200, 30, 1500, 40, 3.5, 40, isotope="A")
    merged = merge_detected_and_targeted_peaks([detected], [targeted], {})
    assert detected in merged
    best, identity_verified = match_peak(
        {"energy_keV": 100.0, "isotope": "A"}, merged, {}
    )
    assert best is targeted
    assert not identity_verified


def test_failed_joint_fit_preserves_window_diagnostic_and_cannot_pass(monkeypatch):
    from fluxforge.analysis import flux_wire_analysis as analysis
    from fluxforge.analysis.peakfit import PeakFitResult, GaussianPeak

    monkeypatch.setattr(
        analysis,
        "fit_multiple_peaks",
        lambda **kwargs: [
            PeakFitResult(
                peak=GaussianPeak(centroid=ch, amplitude=0, sigma=1),
                background=np.zeros(1),
                success=False,
                message="forced failure",
            )
            for ch in kwargs["peak_channels"]
        ],
    )
    data = specimen()
    data.spectrum.counts += 200 * np.exp(
        -0.5 * ((data.spectrum.channels - 108) / 1.5) ** 2
    )
    diagnostics = []
    assert (
        analyze_raw_spectrum_targeted(
            data,
            [GammaLine(100, 0.9, "A"), GammaLine(108, 0.5, "B")],
            background_subtract=False,
            counting_method="iec_tiered",
            fit_diagnostics=diagnostics,
        )
        == []
    )
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["state"] == "unidentifiable_joint_fit"
    lo, hi = diagnostic["channel_window"]
    assert diagnostic["observed_signed_window_counts"] == pytest.approx(
        np.sum(data.spectrum.counts[lo : hi + 1])
    )
    assert not diagnostic["net_peak_area_estimated"]
    assert not diagnostic["isotope_activity_estimated"]
    result = build_validation_flags(
        "RAFM3",
        [
            {
                "matched": True,
                "isotope_match": True,
                "relative_count_error": 0,
                "count_en_score": 0,
            }
        ],
        [{"matched": True, "relative_activity_error": 0, "activity_en_score": 0}],
        [],
        [],
        [],
        [],
        {},
        fit_diagnostics=diagnostics,
    )
    assert result["passed"] is False
    assert result["unidentifiable_fit_groups"] == 1


def test_failed_joint_window_cannot_export_exploratory_activity():
    detected = IdentifiedPeak(
        100,
        100,
        1000,
        20,
        1400,
        400,
        3.5,
        50,
        isotope="A",
        activity_bq=100,
        activity_unc_bq=5,
    )
    outside = IdentifiedPeak(
        200,
        200,
        500,
        30,
        800,
        300,
        3.5,
        16,
        isotope="B",
        activity_bq=50,
        activity_unc_bq=3,
    )
    diagnostics = [{"state": "unidentifiable_joint_fit", "channel_window": [80, 120]}]
    merged = merge_detected_and_targeted_peaks([detected, outside], [], {}, diagnostics)
    assert merged == [outside]
    assert "A" not in combine_peak_activities(merged)
    assert "B" in combine_peak_activities(merged)
