"""Independent analytic truths and counterexamples for issue #241."""

import math
import hashlib
import importlib.util
import json
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from fluxforge.analysis.repeated_count_validation import (
    CountObservation,
    SIMPLE_DECAY,
    UNIFORM_ACCEPTANCE,
    compare_repeated_counts,
    normalize_activity,
)

START = datetime(2025, 8, 12, 10, tzinfo=timezone.utc)


def truth(delay=0, dead=0.2, real=3600, half=3257.4, name=None):
    """Oracle integrates actual emission, without using the production helper."""
    lam = math.log(2) / half
    activity = 1000 * math.exp(-lam * delay)
    counts = 0.05 * 0.8 * (1 - dead) * activity * -math.expm1(-lam * real) / lam
    return CountObservation(
        measurement_id=name or str(delay),
        specimen_id="wire-1",
        nuclide="In116m",
        channel_id="1293.6keV",
        calibration_id="calibration-source-hash",
        background_id="explicit-none-continuum-only",
        count_basis="physical",
        source_sha256=hashlib.sha256(str(delay).encode()).hexdigest(),
        identity_basis="source roster and count receipt",
        half_life_s=half,
        value=counts,
        input_kind="net_counts",
        count_start=START + timedelta(seconds=delay),
        real_time_s=real,
        live_time_s=real * (1 - dead),
        efficiency=0.05,
        gamma_intensity=0.8,
        acceptance_assumption=UNIFORM_ACCEPTANCE,
        eoi=START - timedelta(seconds=900),
    )


def compare(*rows, **kwargs):
    return compare_repeated_counts(
        rows, relative_tolerance=0.01, decay_assumption=SIMPLE_DECAY, **kwargs
    )


@pytest.mark.parametrize("dead", [0.1, 0.2, 0.3])
@pytest.mark.parametrize(
    "reference,delta",
    [
        ("count_start", 0),
        ("count_end", 3600),
        ("end_of_irradiation", -900),
        ("timestamp", 700),
    ],
)
def test_known_truth_reference_and_dead_acceptance_once(dead, reference, delta):
    row = replace(truth(dead=dead), reference_timestamp=START + timedelta(seconds=700))
    result = normalize_activity(row, target_reference=reference)
    assert result["status"] == "AVAILABLE"
    assert result["activity_value"] == pytest.approx(
        1000 * math.exp(-math.log(2) * delta / row.half_life_s), rel=1e-12
    )
    assert result["count_window_corrections"] == 1
    assert result["scientific_admission"] is False


@pytest.mark.parametrize(
    "reference,delta",
    [("count_start", 0), ("count_end", 3600), ("end_of_irradiation", -900)],
)
def test_processed_point_activity_is_not_corrected_twice(reference, delta):
    row = replace(
        truth(),
        input_kind="reference_activity",
        activity_reference=reference,
        includes_count_decay=True,
        value=1000 * math.exp(-math.log(2) * delta / 3257.4),
    )
    result = normalize_activity(row)
    assert result["activity_value"] == pytest.approx(1000)
    assert result["count_window_corrections"] == 0


def test_explicit_count_average_corrected_once():
    raw = truth()
    average = raw.value / (raw.live_time_s * raw.efficiency * raw.gamma_intensity)
    row = replace(
        raw, value=average, input_kind="count_average", includes_count_decay=False
    )
    assert normalize_activity(row)["activity_value"] == pytest.approx(1000)


@pytest.mark.parametrize("n", [2, 3])
def test_repeated_counts_vary_duration_and_dead_fraction(n):
    rows = [
        truth(0, dead=0.1),
        truth(5000, dead=0.3, real=1800),
        truth(9000, dead=0.2, real=7200),
    ][:n]
    result = compare(*rows)
    assert result["status"] == "CONSISTENT"
    assert len(result["pairs"]) == n * (n - 1) / 2
    assert all(abs(p["log_residual"]) < 1e-12 for p in result["pairs"])


def test_wrong_timestamp_or_processed_reference_is_detected():
    a, b = truth(), truth(5000)
    assert (
        compare(a, replace(b, count_start=b.count_start + timedelta(hours=1)))["status"]
        == "DISCREPANCY"
    )
    # A genuine count-end activity incorrectly labelled count-start.
    wrong = replace(
        b,
        input_kind="reference_activity",
        includes_count_decay=True,
        activity_reference="count_start",
        value=1000 * math.exp(-math.log(2) * 8600 / 3257.4),
    )
    assert compare(a, wrong)["status"] == "DISCREPANCY"


def test_long_half_life_short_count_limit():
    row = truth(real=0.01, half=1e25)
    result = normalize_activity(row)
    no_decay = row.value / (row.efficiency * row.gamma_intensity * row.live_time_s)
    assert result["activity_value"] == pytest.approx(no_decay, rel=1e-14)


@pytest.mark.parametrize(
    "field", ["count_start", "real_time_s", "half_life_s", "value"]
)
def test_unknown_acquisition_or_decay_input_is_unavailable(field):
    bad = replace(truth(), **{field: None})
    normalized = normalize_activity(bad)
    assert normalized["status"] == "UNAVAILABLE"
    assert normalized["activity_value"] is None
    result = compare(bad, truth(5000))
    assert result["status"] == "UNAVAILABLE"
    assert result["pairs"][0]["log_residual"] is None
    assert "reaction_rate" not in repr(result)


def test_timezone_unknown_default_and_explicit_naive_scenario():
    a, b = [
        replace(
            truth(t), count_start=truth(t).count_start.replace(tzinfo=None), eoi=None
        )
        for t in (0, 5000)
    ]
    assert compare(a, b)["status"] == "UNAVAILABLE"
    result = compare(
        a, b, common_naive_clock="explicit shared local clock; zone unknown"
    )
    assert result["status"] == "CONSISTENT"
    assert result["scientific_admission"] is False
    assert (
        compare(a, truth(5000), common_naive_clock="shared clock")["status"]
        == "UNAVAILABLE"
    )


def test_same_zone_dst_fold_uses_elapsed_utc():
    zone = ZoneInfo("America/New_York")
    a = replace(
        truth(), count_start=datetime(2025, 11, 2, 1, 15, tzinfo=zone, fold=0), eoi=None
    )
    b = replace(
        truth(3600),
        count_start=datetime(2025, 11, 2, 1, 15, tzinfo=zone, fold=1),
        eoi=None,
    )
    assert compare(a, b)["status"] == "CONSISTENT"
    assert normalize_activity(a, target_reference="count_end")[
        "activity_value"
    ] == pytest.approx(1000 * math.exp(-math.log(2) * 3600 / 3257.4))


def test_nonexistent_dst_time_and_negative_short_count_end_are_unavailable():
    impossible = datetime(2025, 3, 9, 2, 30, tzinfo=ZoneInfo("America/New_York"))
    assert (
        normalize_activity(replace(truth(), count_start=impossible, eoi=None))["status"]
        == "UNAVAILABLE"
    )
    row = replace(
        truth(),
        real_time_s=0.0001,
        live_time_s=0.00008,
        count_end=START - timedelta(microseconds=100),
    )
    assert normalize_activity(row)["status"] == "UNAVAILABLE"
    assert (
        normalize_activity(replace(row, count_end=START + timedelta(microseconds=200)))[
            "status"
        ]
        == "UNAVAILABLE"
    )


@pytest.mark.parametrize(
    "field",
    [
        "specimen_id",
        "nuclide",
        "channel_id",
        "calibration_id",
        "background_id",
        "count_basis",
        "activity_unit",
        "half_life_s",
    ],
)
def test_incompatible_scenarios_are_separate_from_decay_failure(field):
    value = 4000 if field == "half_life_s" else "different"
    result = compare(truth(), replace(truth(5000), **{field: value}))
    pair = result["pairs"][0]
    assert pair["status"] == "INCOMPATIBLE"
    assert "incompatible_" + field in pair["reasons"]
    assert pair["log_residual"] is None


@pytest.mark.parametrize(
    "field",
    [
        "specimen_id",
        "calibration_id",
        "background_id",
        "count_basis",
        "identity_basis",
        "source_sha256",
    ],
)
def test_unknown_identity_is_not_pass(field):
    assert (
        compare(truth(), replace(truth(5000), **{field: None}))["status"]
        == "UNAVAILABLE"
    )


def test_duplicate_single_count_is_not_repeated_count():
    assert compare(truth(), truth())["pairs"][0]["reasons"] == [
        "duplicate_measurement_identity",
        "duplicate_source_sha256",
    ]


def test_alias_of_same_file_and_overlapping_windows_are_not_independent():
    a, b = truth(), truth(5000)
    assert (
        compare(a, replace(b, source_sha256=a.source_sha256.upper()))["pairs"][0][
            "status"
        ]
        == "INCOMPATIBLE"
    )
    assert compare(a, truth(1000))["pairs"][0]["reasons"] == [
        "overlapping_count_windows"
    ]
    assert compare(truth(1000), a)["pairs"][0]["reasons"] == [
        "overlapping_count_windows"
    ]


def test_tiny_half_life_does_not_raise_or_emit_infinite_activity():
    for kind in ("net_counts", "count_average"):
        row = replace(
            truth(), half_life_s=5e-324, input_kind=kind, includes_count_decay=False
        )
        assert normalize_activity(row)["status"] == "UNAVAILABLE"


def test_conflicting_correction_conventions_are_unavailable():
    row = truth()
    # Simulate count-decay-normalized counts submitted with their true flag.
    x = math.log(2) * row.real_time_s / row.half_life_s
    corrected = replace(
        row, includes_count_decay=True, value=row.value * x / -math.expm1(-x)
    )
    assert normalize_activity(corrected)["status"] == "UNAVAILABLE"
    assert (
        normalize_activity(replace(row, activity_reference="count_end"))["status"]
        == "UNAVAILABLE"
    )
    average = replace(
        row,
        input_kind="count_average",
        includes_count_decay=False,
        activity_reference="count_end",
    )
    assert normalize_activity(average)["status"] == "UNAVAILABLE"


def test_known_differing_eois_contradict_simple_decay_history():
    a, b = truth(), truth(5000)
    result = compare(a, replace(b, eoi=START + timedelta(seconds=1800)))
    assert result["pairs"][0]["status"] == "INCOMPATIBLE"
    assert result["reasons"] == ["incompatible_eoi"]


def test_unknown_history_and_eoi_never_assume_saturation():
    a = replace(truth(), eoi=None)
    assert normalize_activity(a, target_reference="end_of_irradiation")["reasons"] == [
        "unknown_eoi"
    ]
    result = compare_repeated_counts(
        [a, truth(5000)], relative_tolerance=0.01, decay_assumption=None
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["pairs"][0]["reasons"] == ["unknown_or_unsupported_decay_history"]


@pytest.mark.parametrize("declared", [None, False, "true", 1])
def test_processed_convention_requires_actual_boolean(declared):
    row = replace(
        truth(),
        input_kind="reference_activity",
        includes_count_decay=declared,
        activity_reference="count_start",
        value=1000,
    )
    assert normalize_activity(row)["status"] == "UNAVAILABLE"


@pytest.mark.parametrize(
    "updates",
    [
        {"real_time_s": 0},
        {"real_time_s": float("nan")},
        {"half_life_s": float("inf")},
        {"live_time_s": 4000},
        {"live_time_s": None},
        {"efficiency": None},
        {"value": float("inf")},
        {"acceptance_assumption": None},
        {"count_end": START + timedelta(seconds=3599)},
        {"eoi": START + timedelta(hours=1)},
    ],
)
def test_invalid_clocks_response_and_acceptance(updates):
    assert normalize_activity(replace(truth(), **updates))["status"] == "UNAVAILABLE"


def test_zero_excluded_and_partial_results_stay_qualified():
    assert compare(replace(truth(), value=0), truth(5000))["status"] == "UNAVAILABLE"
    excluded = replace(truth(9000), exclusion_reason="Sc48 inconsistent report yields")
    result = compare(truth(), truth(5000), excluded)
    assert result["status"] == "PARTIAL"
    assert [p["status"] for p in result["pairs"]] == [
        "CONSISTENT",
        "EXCLUDED",
        "EXCLUDED",
    ]
    assert compare(truth())["status"] == "UNAVAILABLE"


def test_numeric_overflow_and_underflow_are_unavailable():
    for offset in (-1e8, 1e8):
        row = replace(truth(), reference_timestamp=START + timedelta(seconds=offset))
        result = normalize_activity(row, target_reference="timestamp")
        assert result["status"] == "UNAVAILABLE"
        assert result["activity_value"] is None


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf"), True, "0.01"])
def test_bad_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        compare_repeated_counts(
            [truth(), truth(5000)],
            relative_tolerance=tolerance,
            decay_assumption=SIMPLE_DECAY,
        )


def example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/RAFM_irradiation/repeated_count_validation/run_example.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bounded_repeated_count_example", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_ti_example_preserves_unknowns_and_excludes_sc48():
    result = example_module().run_example()
    assert result["physical_counts"] == 3
    assert result["comparison_scenario_counts"] == 3
    assert result["scientific_admission"] is False
    conditional = result["conditional_same_wire_same_clock_line_diagnostics"]
    assert conditional["Sc47@159.4"]["status"] == "CONSISTENT"
    assert conditional["Sc46@1120.5"]["status"] == "CONSISTENT"
    assert conditional["Sc46@889.3"]["status"] == "DISCREPANCY"
    assert all(
        v["status"] == "EXCLUDED"
        for k, v in conditional.items()
        if k.startswith("Sc48")
    )
    assert all(
        v["status"] != "CONSISTENT" for v in result["strict_source_validation"].values()
    )
    assert all(
        v["status"] != "CONSISTENT"
        for v in result["processed_reference_validation"].values()
    )
    assert all(
        a["activity_value"] is None
        for v in result["processed_reference_validation"].values()
        for a in v["activities"]
    )


def test_source_report_byte_change_is_rejected_before_analysis(tmp_path):
    module = example_module()
    fixture = json.loads(
        (module.HERE / "fixtures.json").read_text(encoding="utf-8-sig")
    )["reports"][0]
    original = (module.ROOT / fixture["local_path"]).read_bytes()
    changed = tmp_path / "changed.txt"
    changed.write_bytes(original.replace(b"36,293", b"36,294"))
    assert changed.stat().st_size == len(original)
    with pytest.raises(ValueError, match="hash/size mismatch"):
        module.read_bound_report(changed, fixture)


@pytest.mark.parametrize(
    "field",
    [
        "value",
        "half_life_s",
        "real_time_s",
        "live_time_s",
        "efficiency",
        "gamma_intensity",
    ],
)
def test_boolean_is_not_a_physical_numeric_input(field):
    result = normalize_activity(replace(truth(), **{field: True}))
    assert result["status"] == "UNAVAILABLE"
    assert result["activity_value"] is None


@pytest.mark.parametrize(
    "field",
    [
        "source_sha256",
        "identity_basis",
        "measurement_id",
        "calibration_id",
        "background_id",
        "specimen_id",
    ],
)
def test_malformed_identity_types_are_unavailable(field):
    a, b = [replace(truth(t), **{field: 123}) for t in (0, 5000)]
    result = compare(a, b)
    assert result["status"] == "UNAVAILABLE"
    assert result["pairs"][0]["log_residual"] is None


def test_malformed_exclusion_reason_does_not_fake_excluded_or_raise():
    bad = replace(truth(), exclusion_reason=123)
    assert normalize_activity(bad)["status"] == "UNAVAILABLE"
    assert compare(bad, truth(5000))["status"] == "UNAVAILABLE"
