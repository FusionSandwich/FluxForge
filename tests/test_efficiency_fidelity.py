import math
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import struct

import numpy as np
import pytest

from fluxforge.physics.efficiency_fidelity import (
    AuditCurve,
    CurveIdentity,
    Geometry,
    GammaYield,
    ReferencePoint,
    Thickness,
    audit_header_export,
    check_references,
    coefficient_round_trip,
    compare_efficiencies,
    compare_gamma_yields,
    efficiency_fraction,
    existing_efficiency_curve,
    known_value_control,
    logarithmic_residual,
    read_study_efficiency_header,
    report_effective_curve,
    report_effective_point,
    source_table_curve,
    xcom_pgt_alternative,
)
from fluxforge.analysis.qg_calibration import QGEfficiencyTable
from fluxforge.data.efficiency import EfficiencyCurve, calculate_efficiency_from_source

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "examples/efficiency_audit/fixtures"
CALIBRATION = ROOT / "examples/RAFM_irradiation/calibration"
ANS_SHA = "d04a65ed6052a08b831788a63e89f2eb291b5de4905357de2227e7a845f90ce7"
GEO = Geometry("South HPGe", 25.0, "small_vial_0.5mL")


@pytest.fixture
def table():
    return QGEfficiencyTable(
        CALIBRATION / "South Small Vial 25cm.csv",
        CALIBRATION / "South Small Vial 25cm.provenance.json",
    )


def identity(method="test_model", *, kind="model", energy_range=(40, 1999)):
    return CurveIdentity(
        method,
        kind,
        ("synthetic analytical control",),
        GEO,
        energy_range,
        "unvalidated_model" if kind == "model" else "conditional_report_reconstruction",
        "synthetic; no attenuation",
        "constant",
        "none",
        "historical_report_net" if kind == "report_effective" else "not_applicable",
    )


def test_independent_unit_controls_catch_plausible_factor_100():
    # Known measurement: 5000 counts / (100 s * 100000 Bq * 0.5) = 0.001.
    expected = 0.001
    measured, _ = calculate_efficiency_from_source(5000, 100, 100000, 0.5)
    assert measured == expected
    assert efficiency_fraction(0.1, "percent") == expected
    assert known_value_control(0.001, expected, relative_tolerance=1e-12)["passed"]
    wrong = known_value_control(0.1, expected, relative_tolerance=0.01)
    assert not wrong["passed"]
    assert wrong["suspected_percent_fraction_error"]
    assert Thickness(1000, "um").cm() == 0.1
    assert Thickness(700, "um").cm() == pytest.approx(0.07)
    assert Thickness(1, "mm").areal_density(2.7) == pytest.approx(0.27)
    assert math.exp(-Thickness(1000, "um").cm() * 10) == pytest.approx(math.exp(-1))
    with pytest.raises(ValueError):
        Thickness(0.27, "g/cm2").cm()


def test_polynomial_is_multiplicative_and_log_base_is_explicit():
    np.testing.assert_allclose(
        logarithmic_residual([1, math.e], [2, 3, 0, 0], log_base="ln"), [2, 5]
    )
    assert logarithmic_residual(10, [2, 3, 0, 0], log_base="log10") == 5
    with pytest.raises(ValueError):
        logarithmic_residual(100, [2, 3, 0, 0], log_base="unknown")


def test_round_trip_rejects_reversed_or_scaled_coefficients():
    assert coefficient_round_trip(-20.25685691833496, "-20.2569")[
        "within_printed_rounding"
    ]
    assert not coefficient_round_trip(10.2871723175, "-20.2569")[
        "within_printed_rounding"
    ]
    assert not coefficient_round_trip(-0.202568569, "-20.2569")[
        "within_printed_rounding"
    ]


@pytest.mark.parametrize(
    "value,unit",
    [
        (1.1, "fraction"),
        (101, "percent"),
        (0, "percent"),
        (-1, "fraction"),
        (math.nan, "percent"),
        (1, "unknown"),
    ],
)
def test_unit_range_errors_are_rejected(value, unit):
    with pytest.raises(ValueError):
        efficiency_fraction(value, unit)


def test_source_header_export_report_round_trips_and_unresolved_slots(table):
    header = read_study_efficiency_header(
        FIXTURES / "Co-Cd-RAFM-1.ANS", expected_sha256=ANS_SHA
    )
    audit = audit_header_export(
        header, table, (FIXTURES / "Co-Cd-RAFM-1.txt").read_text(encoding="cp1252")
    )
    assert audit["coefficient_round_trips_pass"]
    assert audit["report_anchor_status"] == "corroborated_at_printed_precision"
    assert audit["export_DI_raw"] == 1.39
    assert audit["physical_detector_thickness_cm"] == pytest.approx(6.45)
    assert audit["DI_mapping_status"].startswith("unsupported")
    assert audit["Error_saved_raw"] == pytest.approx(4.5893419e-5)
    assert audit["Error_export_raw"] == 0.00458934
    assert header["values"]["dead_layer_observed_um"] == 700
    # A wrong coefficient can still give a plausible curve: reject the identity.
    header["values"]["C1"] *= 0.01
    assert not audit_header_export(
        header, table, (FIXTURES / "Co-Cd-RAFM-1.txt").read_text(encoding="cp1252")
    )["coefficient_round_trips_pass"]


def test_thickness_unit_mistake_and_missing_report_anchors(table):
    header = read_study_efficiency_header(
        FIXTURES / "Co-Cd-RAFM-1.ANS", expected_sha256=ANS_SHA
    )
    report = (FIXTURES / "Co-Cd-RAFM-1.txt").read_text(encoding="cp1252")
    wrong = report.replace("1000.000 um", "1000.000 cm")
    assert (
        audit_header_export(header, table, wrong)["report_anchor_status"]
        == "contradiction"
    )
    missing = report.replace("Al Window (T1):", "Unidentified window:")
    audit = audit_header_export(header, table, missing)
    assert audit["report_anchor_status"] == "unknown"
    assert audit["thickness_geometry_anchors"]["window_um"]["matches"] is None


def test_actual_saved_20cm_vs_report_25cm_conflict_preserves_both(table):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    header = read_study_efficiency_header(
        FIXTURES / "RAFM-A-300s.ANS",
        expected_sha256=manifest["sha256"]["RAFM-A-300s.ANS"],
    )
    report_bytes = (FIXTURES / "RAFM-A-300s.txt").read_bytes()
    assert (
        hashlib.sha256(report_bytes).hexdigest()
        == manifest["sha256"]["RAFM-A-300s.txt"]
    )
    audit = audit_header_export(header, table, report_bytes.decode("cp1252"))
    assert audit["report_anchor_status"] == "contradiction"
    assert not audit["coefficient_round_trips_pass"]
    anchors = audit["thickness_geometry_anchors"]["source_distance_cm"]
    assert anchors["saved_cm"] == 20
    assert anchors["report_cm"] == 25


def test_changed_original_hash_and_wrong_revision_channels_length_fail(tmp_path):
    data = (FIXTURES / "Co-Cd-RAFM-1.ANS").read_bytes()
    for offset, fmt, value in [(0, "<h", 3), (1022, "<h", 8190), (1026, "<h", 100)]:
        changed = bytearray(data)
        struct.pack_into(fmt, changed, offset, value)
        path = tmp_path / "changed.ANS"
        path.write_bytes(changed)
        with pytest.raises(ValueError, match="hash"):
            read_study_efficiency_header(path, expected_sha256=ANS_SHA)
        with pytest.raises(ValueError):
            read_study_efficiency_header(
                path, expected_sha256=hashlib.sha256(changed).hexdigest()
            )


def test_source_reproduction_reference_knots_invalid_brackets_and_range(table):
    curve = source_table_curve(table, unit_assumption="percent")
    # Withheld literal source values, not computed from QG target activities.
    for energy, expected in [
        (100, 0.00100230),
        (500, 0.000821244),
        (1000, 0.000444251),
        (1999, 0.00029946),
    ]:
        assert curve.at(energy, GEO)["efficiency_fraction"] == pytest.approx(
            expected, rel=1e-12
        )
    assert curve.at(100.5, GEO)["efficiency_fraction"] == pytest.approx(
        (0.10023 + 0.102312) / 200
    )
    for energy in [40, 56, 56.5]:
        row = curve.at(energy, GEO)
        assert row["status"] == "excluded_nonpositive_efficiency"
        assert row["efficiency_fraction"] is None
    for energy in [39, 2000]:
        assert curve.at(energy, GEO)["status"] == "excluded_outside_range"
    assert curve.at(57, GEO)["admissible_for_comparison"]
    assert table.rows[0]["efficiency_reported"] == -0.0151349


def test_energy_deltas_explicit_selection_and_fe_cd_rejection(table):
    percent = source_table_curve(table, unit_assumption="percent")
    fraction = source_table_curve(table, unit_assumption="fraction")
    output = compare_efficiencies(
        percent,
        [fraction],
        [100, 40, 2000],
        geometry=GEO,
        selected_method=fraction.identity.method_id,
    )
    wrong = output["rows"][1]
    assert wrong["delta_relative_to_reference"] == pytest.approx(99)
    assert wrong["selected"] and wrong["scientific_admission"] is False
    assert output["rows"][3]["delta_fraction"] is None
    json.dumps(output, allow_nan=False)
    near = Geometry("South HPGe", 0, "wire", True)
    assert percent.at(100, near)["status"] == "excluded_near_contact_25cm_transfer"
    assert (
        percent.at(100, replace(GEO, distance_cm=None))["status"] == "unknown_geometry"
    )
    assert (
        percent.at(100, replace(GEO, distance_cm=20))["status"]
        == "excluded_geometry_mismatch"
    )
    with pytest.raises(ValueError):
        compare_efficiencies(
            percent, [fraction], [100], geometry=GEO, selected_method="automatic"
        )


def test_existing_model_is_snapshotted_and_no_extrapolation_or_invalid_clipping():
    curve = EfficiencyCurve.from_polynomial([math.log(0.001)], energy_range=(40, 1999))
    adapter = existing_efficiency_curve(curve, identity())
    curve.parameters["coefficients"][0] = math.log(0.1)
    assert adapter.at(100, GEO)["efficiency_fraction"] == pytest.approx(0.001)
    assert adapter.at(2000, GEO)["efficiency_fraction"] is None
    bad = AuditCurve(identity(), lambda e: 2)
    assert bad.at(100, GEO)["status"] == "excluded_invalid_model_value"
    unsupported = AuditCurve(
        identity(), None, "proprietary McMaster routines unavailable"
    )
    assert unsupported.at(100, GEO)["status"] == "unsupported_model"
    with pytest.raises(ValueError):
        replace(identity(), kind="source_calibration_export")


def test_density_aware_model_independent_known_table_value_and_units():
    # Literal embedded table knots at 100 keV: Al/Ge 0.03709/0.1038 cm²/g.
    # This tests algebra and units; it does not independently qualify these tables.
    # Densities 2.699/5.323 g/cm³: density cannot be omitted from the exponent.
    model = xcom_pgt_alternative(
        geometry=GEO,
        coefficients=[1, 0, 0, 0],
        geometry_factor=0.01,
        window=Thickness(1000, "um"),
        dead_layer=Thickness(700, "um"),
        detector=Thickness(6.45, "cm"),
        angle_deg=0,
        log_base="ln",
        energy_range_keV=(40, 1999),
        source_refs=("synthetic known-value test",),
    )
    expected = (
        0.01
        * math.exp(-(0.03709 * 2.699 * 0.1 + 0.1038 * 5.323 * 0.07))
        * (1 - math.exp(-0.1038 * 5.323 * 6.45))
    )
    assert model.at(100, GEO)["efficiency_fraction"] == pytest.approx(
        expected, rel=1e-12
    )
    assert "not vendor McMaster" in model.identity.attenuation_data
    assert model.at(2000, GEO)["efficiency_fraction"] is None


@pytest.mark.parametrize("angle", [90, -90, 180, math.nan])
def test_angle_failure_does_not_become_normal_incidence(angle):
    with pytest.raises(ValueError):
        xcom_pgt_alternative(
            geometry=GEO,
            coefficients=[1, 0, 0, 0],
            geometry_factor=0.01,
            window=Thickness(1000, "um"),
            dead_layer=Thickness(700, "um"),
            detector=Thickness(6.45, "cm"),
            angle_deg=angle,
            log_base="ln",
            energy_range_keV=(40, 1999),
            source_refs=("test",),
        )


def test_historical_yields_kept_separate_and_effective_curve_never_independent():
    historical = GammaYield(1, "percent", "QG:original:line", "historical_report")
    modern = GammaYield(
        1, "fraction", "synthetic evaluated-yield scenario", "modern_evaluated"
    )
    report = compare_gamma_yields(historical, modern)
    assert report["historical"]["value"] == report["modern"]["value"] == 1
    assert report["historical"]["fraction"] == 0.01
    assert report["modern"]["fraction"] == 1
    assert compare_gamma_yields(historical, None)["modern_status"] == "unavailable"
    assert GammaYield(
        179.8, "percent", "annihilation control", "historical_report"
    ).fraction() == pytest.approx(1.798)
    point = report_effective_point(
        net_counts=100,
        activity_bq=1000,
        live_time_s=100,
        historical_yield=historical,
        report_ref="QG:source",
    )
    assert point["efficiency_fraction"] == 0.1
    assert point["count_basis"] == "historical_report_net"
    curve = report_effective_curve(
        [100, 200], [point, point], geometry=GEO, source_refs=("QG",)
    )
    assert curve.identity.kind == "report_effective"
    assert curve.at(150, GEO)["efficiency_fraction"] == 0.1
    assert not curve.at(150, GEO)["scientific_admission"]


def test_independent_validation_requires_true_holdout_not_target_activity():
    curve = AuditCurve(identity(), lambda e: 0.001)
    refs = [
        ReferencePoint(str(i), 100, 0.001, "source-" + str(i), origin, GEO, 0.01)
        for i, origin in enumerate(
            [
                "implementation_control",
                "source_export_holdout",
                "quantumgold_target_activity",
                "report_effective",
                "independent_calibration",
                "independent_calibration",
            ]
        )
    ]
    refs[4] = replace(
        refs[4],
        source_id="certificate:sha256:" + "a" * 64,
        certificate_ref="certificate:sha256:" + "a" * 64,
    )
    refs[5] = replace(
        refs[5],
        source_id="certificate:sha256:" + "b" * 64,
        certificate_ref="certificate:sha256:" + "b" * 64,
    )
    rows = check_references(curve, refs, fitted_source_ids=["b" * 64])
    assert all(row["control"]["passed"] for row in rows)
    assert [r["independent_validation_pass"] for r in rows] == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert rows[5]["fitted_source_reused"]


@pytest.mark.parametrize(
    "alias",
    ["{sha}", "changed-name:SHA256:{sha}", "new-label/{sha}/reference", "{upper}"],
)
def test_exact_source_or_sha_alias_cannot_be_independent(table, alias):
    curve = source_table_curve(table, unit_assumption="percent")
    source_id = alias.format(sha=table.sha256, upper=table.sha256.upper())
    ref = ReferencePoint(
        "false-independent",
        100,
        0.0010023,
        source_id,
        "independent_calibration",
        GEO,
        1e-12,
        certificate_ref="claimed-cert:sha256:" + table.sha256,
    )
    result = check_references(curve, [ref], fitted_source_ids=[])[0]
    assert result["control"]["passed"]
    assert result["curve_source_reused"]
    assert not result["independent_absolute_validation"]
    assert not result["independent_validation_pass"]


def test_missing_certificate_does_not_establish_independence():
    curve = AuditCurve(identity(), lambda e: 0.001)
    ref = ReferencePoint(
        "missing-cert",
        100,
        0.001,
        "external-source",
        "independent_calibration",
        GEO,
        0.01,
    )
    result = check_references(curve, [ref], fitted_source_ids=[])[0]
    assert result["control"]["passed"]
    assert result["independence_status"] == "missing_or_unbound_certificate_identity"
    assert not result["independent_validation_pass"]


def test_caller_owned_identity_containers_cannot_change_provenance_or_range():
    refs = ["original-source"]
    bounds = [100, 200]
    declared = replace(identity(), source_refs=refs, energy_range_keV=bounds)
    curve = AuditCurve(declared, lambda e: 0.001)
    refs[0] = "changed-source"
    bounds[1] = 1000
    assert declared.source_refs == ("original-source",)
    assert curve.at(300, GEO)["status"] == "excluded_outside_range"
    assert declared.energy_range_keV == (100, 200)


def test_report_curve_copies_mutable_energies_and_values():
    historical = GammaYield(50, "percent", "original-yield", "historical_report")
    points = [
        report_effective_point(
            net_counts=n,
            activity_bq=100,
            live_time_s=100,
            historical_yield=historical,
            report_ref="original-report",
        )
        for n in [500, 1000]
    ]
    energies = np.array([100.0, 200.0])
    curve = report_effective_curve(
        energies, points, geometry=GEO, source_refs=("original",)
    )
    assert curve.at(150, GEO)["efficiency_fraction"] == pytest.approx(0.15)
    energies[1] = 10000
    points[1]["efficiency_fraction"] = 0.9
    assert curve.at(150, GEO)["efficiency_fraction"] == pytest.approx(0.15)


def test_missing_reference_and_bad_source_data_never_zero_or_pass():
    curve = AuditCurve(identity(), None, "missing tables")
    ref = ReferencePoint(
        "independent", 100, 0.001, "certificate", "independent_calibration", GEO, 0.01
    )
    row = check_references(curve, [ref], fitted_source_ids=[])[0]
    assert not row["independent_validation_pass"]
    assert row["evaluation"]["efficiency_fraction"] is None
    with pytest.raises(ValueError):
        replace(identity(), energy_range_keV=(100, math.inf))


@pytest.mark.parametrize("energy", [math.nan, math.inf, 0, -1])
def test_invalid_energy_export_retains_unknown_without_nan_json(energy):
    curve = AuditCurve(identity(), lambda e: 0.001)
    row = curve.at(energy, GEO)
    assert row["status"] == "invalid_energy"
    assert row["efficiency_fraction"] is None
    json.dumps(row, allow_nan=False)
