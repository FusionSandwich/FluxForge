"""Counterexamples for the bounded historical protocol; no vendor parity claim."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

from fluxforge.analysis.qg_protocol import (
    HistoricalProtocol,
    LAYOUT_ID,
    ProtocolField,
    parse_saved_state,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples/qg_protocol/fixtures"


def synthetic_header():
    data = bytearray(1548 + 8192 * 4)
    for offset, fmt, value in [
        (0, "<h", 4),
        (96, "<d", 101),
        (104, "<d", 100),
        (860, "<H", 1),
        (852, "<f", 4),
        (846, "<h", 1),
        (1022, "<h", 8191),
    ]:
        struct.pack_into(fmt, data, offset, value)
    data[1294:1306] = b"GammaLib.mdb"
    return bytes(data)


def parse(data):
    return parse_saved_state(
        data, expected_sha256=hashlib.sha256(data).hexdigest(), layout_id=LAYOUT_ID
    )


def test_bounded_first_saved_controls():
    state = parse(synthetic_header())
    assert state.to_dict()["saved_state"] == {
        "ambient_enabled": False,
        "continuum_enabled": True,
    }
    assert state.report_confirmed == {}
    assert "SAVED_HEADER_ONLY" in state.to_dict()["qualification"]


@pytest.mark.parametrize(
    "offset,fmt,value,match",
    [(0, "<h", 3, "revision"), (1022, "<h", 4095, "bounds"), (1026, "<h", 1, "length")],
)
def test_wrong_layout_anchors(offset, fmt, value, match):
    data = bytearray(synthetic_header())
    struct.pack_into(fmt, data, offset, value)
    with pytest.raises(ValueError, match=match):
        parse(bytes(data))


def test_wrong_source_hash():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_saved_state(
            synthetic_header(), expected_sha256="0" * 64, layout_id=LAYOUT_ID
        )


@pytest.mark.parametrize(
    "offset,fmt,value,match",
    [
        (0, "<h", 5, "revision"),
        (1020, "<h", 1, "bounds"),
        (1026, "<h", -1, "length"),
        (104, "<d", float("nan"), "timing"),
        (96, "<d", float("inf"), "timing"),
        (104, "<d", 102, "timing"),
        (860, "<H", 4, "control"),
        (1292, "<h", 7, "boolean"),
        (852, "<f", float("nan"), "ROI"),
        (852, "<f", 0, "ROI"),
        (848, "<f", -1, "ROI"),
        (846, "<h", 0, "ROI"),
    ],
)
def test_independent_structural_anchors(offset, fmt, value, match):
    data = bytearray(synthetic_header())
    struct.pack_into(fmt, data, offset, value)
    with pytest.raises(ValueError, match=match):
        parse(bytes(data))


@pytest.mark.parametrize(
    "data",
    [b"", synthetic_header()[:-1], synthetic_header() + b"x"],
    ids=["empty", "truncated", "extra_byte"],
)
def test_truncation_and_extra_bytes(data):
    with pytest.raises(ValueError, match="Truncated|length"):
        parse(data)


def test_explicit_layout_and_library_anchor():
    data = synthetic_header()
    with pytest.raises(ValueError, match="layout"):
        parse_saved_state(
            data,
            expected_sha256=hashlib.sha256(data).hexdigest(),
            layout_id="manual-appendix-c",
        )
    changed = bytearray(data)
    changed[1294] = 255
    with pytest.raises(ValueError, match="descriptor"):
        parse(bytes(changed))


def test_all_32_original_hashes_and_saved_headers():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert len(manifest["rows"]) == 32
    assert len({r["sha256"] for r in manifest["rows"]}) == 32
    for row in manifest["rows"]:
        data = (FIXTURES / row["source"]).read_bytes()
        report = (
            (FIXTURES / row["report_source"]).read_bytes()
            if row["report_sha256"]
            else None
        )
        state = parse_saved_state(
            data,
            expected_sha256=row["sha256"],
            layout_id=LAYOUT_ID,
            report=report,
            expected_report_sha256=row["report_sha256"],
        )
        assert state.values == row["expected_values"]
        assert state.byte_length == row["bytes"]
        assert state.to_dict()["saved_state"] == {
            "ambient_enabled": False,
            "continuum_enabled": True,
        }
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        if report:
            assert state.report_confirmed["use_library_efficiencies"] is False
            assert state.report_confirmed["activity_reference"] == "measurement_date"
        else:
            assert state.report_confirmed == {}


def test_ambient_bit_changes_only_saved_scenario():
    data = bytearray(synthetic_header())
    before = parse(bytes(data))
    struct.pack_into("<H", data, 860, 0)
    after = parse(bytes(data))
    assert before.source_sha256 != after.source_sha256
    assert {k: v for k, v in before.values.items() if k != "analysis_ctrl"} == {
        k: v for k, v in after.values.items() if k != "analysis_ctrl"
    }
    assert (
        before.to_dict()["saved_state"]["continuum_enabled"]
        == after.to_dict()["saved_state"]["continuum_enabled"]
    )
    assert after.to_dict()["saved_state"]["ambient_enabled"] is True
    with pytest.raises(ValueError, match="mismatch"):
        parse_saved_state(
            bytes(data), expected_sha256=before.source_sha256, layout_id=LAYOUT_ID
        )


def report_state(report, **kwargs):
    data = synthetic_header()
    return parse_saved_state(
        data,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        layout_id=LAYOUT_ID,
        report=report,
        expected_report_sha256=hashlib.sha256(report).hexdigest(),
        **kwargs,
    )


GOOD_REPORT = b"Library: GammaLib.mdb\nLT: 100.00 RT: 101.00\nLibrary efficiencies were ignored\nActivities reported as of Measurement Date."


def test_report_hash_and_contradictions():
    assert (
        report_state(GOOD_REPORT).report_confirmed["use_library_efficiencies"] is False
    )
    for changed in (
        GOOD_REPORT.replace(b"GammaLib", b"OtherLib"),
        GOOD_REPORT.replace(b"100.00", b"99.00"),
        b"Library: GammaLib.mdb",
    ):
        with pytest.raises(ValueError, match="report|Report"):
            report_state(changed)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_saved_state(
            synthetic_header(),
            expected_sha256=hashlib.sha256(synthetic_header()).hexdigest(),
            layout_id=LAYOUT_ID,
            report=GOOD_REPORT,
            expected_report_sha256="0" * 64,
        )
    data = bytearray(synthetic_header())
    struct.pack_into("<h", data, 1292, 1)
    with pytest.raises(ValueError, match="contradicts"):
        parse_saved_state(
            bytes(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
            layout_id=LAYOUT_ID,
            report=GOOD_REPORT,
            expected_report_sha256=hashlib.sha256(GOOD_REPORT).hexdigest(),
        )
    # Absence of an explicit report statement is unknown, not confirmation.
    minimal = b"Library: GammaLib.mdb\nLT: 100.00 RT: 101.00"
    assert set(report_state(minimal).report_confirmed) == {"gamma_library_name"}


def test_protocol_roundtrip_statuses_and_independent_assumptions():
    state = report_state(GOOD_REPORT)
    protocol = HistoricalProtocol.from_saved_state(state, scenario_name="observed")
    assert (
        HistoricalProtocol.from_dict(json.loads(protocol.to_json())).to_dict()
        == protocol.to_dict()
    )
    for name in (
        "gamma_intensities",
        "gamma_corrections",
        "gamma_library_sha256",
        "aggregation",
        "efficiency_identity",
        "efficiency_sha256",
        "activity_reference_timestamp",
        "activity_reference_timezone",
    ):
        assert protocol.fields[name].status == "unknown"
        assert protocol.fields[name].value is None
    assert protocol.fields["ambient_enabled"].status == "saved_state"
    assert protocol.fields["use_library_efficiencies"].status == "report_confirmed"
    changed = protocol.with_assumptions(
        scenario_name="counterfactual",
        rationale="test declared ambient on",
        ambient_enabled=True,
    )
    assert changed.saved_header.to_dict() == protocol.saved_header.to_dict()
    assert changed.fields["ambient_enabled"].status == "assumed"
    assert {k: v for k, v in changed.fields.items() if k != "ambient_enabled"} == {
        k: v for k, v in protocol.fields.items() if k != "ambient_enabled"
    }
    assert protocol.fields["ambient_enabled"].value is False
    with pytest.raises(TypeError):
        protocol.fields["ambient_enabled"] = ProtocolField(True, "assumed", "mutable")


@pytest.mark.parametrize(
    "choices",
    [
        {"ambient_enabled": "false"},
        {"roi_width_fwhm": True},
        {"roi_width_fwhm": 0},
        {"background_gap_fwhm": -1},
        {"background_width_channels": 1.2},
        {"gamma_library_sha256": "missing"},
        {"continuum_enabled": None},
        {"ambient_enabled": float("nan")},
        {"bogus": 3},
    ],
)
def test_invalid_protocol_choices_fail(choices):
    protocol = HistoricalProtocol.from_saved_state(
        parse(synthetic_header()), scenario_name="test"
    )
    with pytest.raises(ValueError):
        protocol.with_assumptions(scenario_name="invalid", rationale="test", **choices)


def test_unknown_and_unsupported_method_not_executed():
    protocol = HistoricalProtocol.from_saved_state(
        parse(synthetic_header()), scenario_name="test"
    )
    with pytest.raises(ValueError, match="continuum method"):
        protocol.roi_parameters()
    fields = dict(protocol.fields)
    fields["continuum_method"] = ProtocolField(
        "vendor_unknown_algorithm", "unsupported", "Unknown installed version"
    )
    unsupported = HistoricalProtocol("unsupported", fields, protocol.saved_header)
    restored = HistoricalProtocol.from_dict(json.loads(unsupported.to_json()))
    assert restored.fields["continuum_method"].status == "unsupported"
    with pytest.raises(ValueError, match="continuum method"):
        restored.roi_parameters()


def load_example():
    spec = importlib.util.spec_from_file_location(
        "qg_focused_example", ROOT / "examples/qg_protocol/run_example.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


def test_focused_example_separates_bases_and_controls():
    example = load_example()
    saved = example.build_evidence()
    on = example.build_evidence(ambient="on")
    gross = example.build_evidence(continuum="off")
    assert len(saved["saved_headers"]) == 32
    assert len(saved["line_evidence"]) == 2
    assert saved["protocol"]["fields"]["ambient_enabled"]["value"] is False
    assert saved["protocol"]["fields"]["continuum_enabled"]["value"] is True
    for alternative in (on, gross):
        assert alternative["physical_count_basis"] == saved["physical_count_basis"]
        assert [r["physical_counts"] for r in alternative["line_evidence"]] == [
            r["physical_counts"] for r in saved["line_evidence"]
        ]
        assert (
            alternative["protocol"]["saved_header"] == saved["protocol"]["saved_header"]
        )
    for line in saved["line_evidence"]:
        comparison = line["comparison_counts"]
        assert comparison["local_continuum_counts"] > 0
        assert comparison["net_counts"] == pytest.approx(
            comparison["gross_counts"] - comparison["local_continuum_counts"]
        )
        assert comparison["net_counts"] != line["physical_counts"]["net_counts"]
    for line in gross["line_evidence"]:
        assert (
            line["comparison_counts"]["net_counts"]
            == line["comparison_counts"]["gross_counts"]
        )
        assert line["comparison_counts"]["local_continuum_counts"] == 0
    assert saved["exact_vendor_parity"] is False
    assert saved["physical_defaults_changed"] is False
    assert saved["protocol"]["fields"]["gamma_corrections"]["value"] is None


def test_serialized_header_spoof_needs_source_rebinding():
    protocol = HistoricalProtocol.from_saved_state(
        parse(synthetic_header()), scenario_name="test"
    )
    payload = json.loads(protocol.to_json())
    payload["saved_header"]["values"]["revision"] = 3
    with pytest.raises(ValueError, match="revision"):
        HistoricalProtocol.from_dict(payload)
    payload = json.loads(protocol.to_json())
    payload["fields"]["ambient_enabled"]["value"] = True
    with pytest.raises(ValueError, match="saved-state"):
        HistoricalProtocol.from_dict(payload)
    payload = json.loads(protocol.to_json())
    payload["saved_header"]["values"]["analysis_ctrl"] = 0
    payload["saved_header"]["saved_state"]["ambient_enabled"] = True
    payload["fields"]["ambient_enabled"]["value"] = True
    # Self-consistent JSON is not proof of source bytes.
    restored = HistoricalProtocol.from_dict(payload)
    with pytest.raises(ValueError, match="differs from source bytes"):
        restored.verify_sources(synthetic_header())
    assert (
        protocol.verify_sources(synthetic_header()).source_sha256
        == protocol.saved_header.source_sha256
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("gamma_intensities", "unknown_gamma_yield"),
        ("aggregation", "activity_over_sigma"),
        ("ambient_enabled", False),
    ],
)
def test_false_report_confirmation_rejected(field, value):
    protocol = HistoricalProtocol.from_saved_state(
        report_state(GOOD_REPORT), scenario_name="test"
    )
    fields = dict(protocol.fields)
    fields[field] = ProtocolField(
        value,
        "report_confirmed",
        f"Report SHA256:{protocol.saved_header.report_sha256}",
    )
    with pytest.raises(ValueError, match="report confirmation"):
        HistoricalProtocol("false_confirmation", fields, protocol.saved_header)


def test_rebound_report_and_immutable_header():
    state = report_state(GOOD_REPORT)
    protocol = HistoricalProtocol.from_saved_state(state, scenario_name="test")
    restored = HistoricalProtocol.from_dict(json.loads(protocol.to_json()))
    assert (
        restored.verify_sources(synthetic_header(), report=GOOD_REPORT).report_confirmed
        == state.report_confirmed
    )
    with pytest.raises(ValueError, match="without report"):
        restored.verify_sources(synthetic_header())
    with pytest.raises(ValueError, match="mismatch"):
        restored.verify_sources(synthetic_header(), report=GOOD_REPORT + b"changed")
    with pytest.raises(TypeError):
        state.values["analysis_ctrl"] = 0
    with pytest.raises(TypeError):
        state.report_confirmed["activity_reference"] = "EOI"


def test_portable_raw_count_oracle():
    """Direct raw bin sums challenge an incorrectly shared physical count basis."""
    import numpy as np
    from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc

    example = load_example()
    evidence = example.build_evidence()
    report = read_processed_txt(FIXTURES / "QG_report/Co-Cd-RAFM-1.txt")
    sample = read_raw_asc(
        FIXTURES / "Co-Cd-RAFM-1.ASC",
        energy_calibration_override=report.energy_calibration,
    ).spectrum
    for line in evidence["line_evidence"]:
        row = line["comparison_counts"]
        lo, hi = row["roi_channels"]
        gross = float(np.sum(sample.counts[lo : hi + 1]))
        # These observed saved settings use one immediately adjacent channel
        # per side. This is an independent count and variance oracle.
        local = float(
            (hi - lo + 1) * (sample.counts[lo - 1] + sample.counts[hi + 1]) / 2
        )
        variance = (
            gross
            + (hi - lo + 1) ** 2 * (sample.counts[lo - 1] + sample.counts[hi + 1]) / 4
        )
        assert row["gross_counts"] == gross
        assert row["local_continuum_counts"] == local
        assert row["net_counts"] == gross - local
        assert row["net_counts_unc"] ** 2 == pytest.approx(variance)


def test_cli_no_overwrite_and_source_set_gate(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "evidence.json"
    output.write_text("preserve existing artifact")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/qg_protocol/run_example.py"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert output.read_text() == "preserve existing artifact"
    example = load_example()
    with pytest.raises(ValueError, match="32-file"):
        example.audit_originals(
            tmp_path, json.loads((FIXTURES / "manifest.json").read_text())
        )
    with pytest.raises(ValueError, match="on/off"):
        example.build_evidence(ambient="unknown")


def test_serialized_schema_and_missing_fields_rejected():
    protocol = HistoricalProtocol.from_saved_state(
        parse(synthetic_header()), scenario_name="test"
    )
    for key, value in [
        ("schema_version", True),
        ("schema_version", 2),
        ("purpose", "physical"),
        ("physical_defaults_changed", True),
        ("exact_vendor_parity", True),
    ]:
        payload = json.loads(protocol.to_json())
        payload[key] = value
        with pytest.raises(ValueError):
            HistoricalProtocol.from_dict(payload)
    payload = json.loads(protocol.to_json())
    del payload["fields"]["gamma_intensities"]
    with pytest.raises(ValueError, match="fields"):
        HistoricalProtocol.from_dict(payload)
