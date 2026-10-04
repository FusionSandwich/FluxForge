"""Bounded efficiency audit; originals, defaults and activities unchanged."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402 (repository path bootstrap for direct execution)
import scipy  # noqa: E402

from fluxforge.analysis.qg_calibration import QGEfficiencyTable  # noqa: E402
from fluxforge.validation.example_identity import source_identity  # noqa: E402
from fluxforge.analysis.qg_report_qc import qg_yield_diagnostic  # noqa: E402
from fluxforge.io.flux_wire import (  # noqa: E402
    EfficiencyCalibration,
    read_processed_txt,
)  # noqa: E402
from fluxforge.physics.efficiency_fidelity import (  # noqa: E402
    AuditCurve,
    CurveIdentity,
    GammaYield,
    Geometry,
    ReferencePoint,
    Thickness,
    audit_header_export,
    check_references,
    compare_efficiencies,
    compare_gamma_yields,
    read_study_efficiency_header,
    report_effective_curve,
    report_effective_point,
    source_table_curve,
    xcom_pgt_alternative,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, selected_method: str) -> dict:
    if output.exists():
        raise FileExistsError(f"Audit output must be a new directory: {output}")
    here = Path(__file__).resolve().parent
    manifest_path = here / "fixtures/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {"fixtures/manifest.json": sha(manifest_path)}
    for name, expected in manifest["sha256"].items():
        actual = sha(here / "fixtures" / name)
        if actual != expected:
            raise ValueError(f"Original fixture changed: {name}")
        inputs[f"fixtures/{name}"] = actual
    calibration = ROOT / "examples/RAFM_irradiation/calibration"
    table = QGEfficiencyTable(
        calibration / "South Small Vial 25cm.csv",
        calibration / "South Small Vial 25cm.provenance.json",
    )
    for path in [table.path, calibration / "South Small Vial 25cm.provenance.json"]:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    source = source_table_curve(table, unit_assumption="percent")
    geo = source.identity.geometry
    header = read_study_efficiency_header(
        here / "fixtures/Co-Cd-RAFM-1.ANS",
        expected_sha256=manifest["sha256"]["Co-Cd-RAFM-1.ANS"],
    )
    report_path = here / "fixtures/Co-Cd-RAFM-1.txt"
    report = read_processed_txt(report_path)
    header_audit = audit_header_export(
        header, table, report_path.read_text(encoding="cp1252")
    )
    if (
        not header_audit["coefficient_round_trips_pass"]
        or header_audit["report_anchor_status"] != "corroborated_at_printed_precision"
    ):
        raise ValueError("Source coefficient/thickness report anchors failed")
    conflicting_header = read_study_efficiency_header(
        here / "fixtures/RAFM-A-300s.ANS",
        expected_sha256=manifest["sha256"]["RAFM-A-300s.ANS"],
    )
    conflict = audit_header_export(
        conflicting_header,
        table,
        (here / "fixtures/RAFM-A-300s.txt").read_text(encoding="cp1252"),
    )
    if (
        conflict["report_anchor_status"] != "contradiction"
        or conflict["coefficient_round_trips_pass"]
    ):
        raise ValueError("Actual saved-vs-report state conflict was not detected")
    refs = (
        f"Co-Cd-RAFM-1.ANS:sha256:{inputs['fixtures/Co-Cd-RAFM-1.ANS']}",
        f"Co-Cd-RAFM-1.txt:sha256:{inputs['fixtures/Co-Cd-RAFM-1.txt']}",
    )
    values = header["values"]
    coeffs = [values[f"C{i}"] for i in range(1, 5)]
    density_model = xcom_pgt_alternative(
        geometry=geo,
        coefficients=coeffs,
        geometry_factor=values["A"],
        window=Thickness(values["window_um"], "um"),
        dead_layer=Thickness(values["dead_layer_observed_um"], "um"),
        detector=Thickness(values["detector_thickness_cm"], "cm"),
        angle_deg=values["angle_deg"],
        log_base="ln",
        energy_range_keV=source.identity.energy_range_keV,
        source_refs=refs,
    )
    # Existing historical engine implementation is retained verbatim. Its current
    # cm × mass-coefficient convention differs from the density-aware alternative.
    legacy = EfficiencyCalibration(
        C1=coeffs[0],
        C2=coeffs[1],
        C3=coeffs[2],
        C4=coeffs[3],
        geometry_factor_A=values["A"],
        al_window_T1_um=values["window_um"],
        dead_layer_DL_um=values["dead_layer_observed_um"],
        detector_thickness_DI_cm=values["detector_thickness_cm"],
        incident_angle_AI_deg=values["angle_deg"],
    )
    legacy_identity = CurveIdentity(
        "current_engine_header_model",
        "model",
        refs,
        geo,
        source.identity.energy_range_keV,
        "unvalidated_model",
        density_model.identity.attenuation_data,
        (
            "existing EfficiencyCalibration.efficiency: cm times mu/rho, "
            "no density; ln residual; existing clipping retained"
        ),
        "existing XCOM log-log linear; audit forbids extrapolation",
    )
    legacy_curve = AuditCurve(legacy_identity, lambda e: float(legacy.efficiency(e)))
    unsupported = AuditCurve(
        CurveIdentity(
            "vendor_mcmaster_model",
            "model",
            refs,
            geo,
            source.identity.energy_range_keV,
            "unvalidated_model",
            "PGT McMaster routines with vendor edge/line extensions: unavailable",
            (
                "reported detector model times logarithmic residual; "
                "log convention/deployed version not established"
            ),
            "vendor behavior unknown",
        ),
        None,
        "Vendor attenuation routines unavailable; exported DI mapping unresolved",
    )

    effective = []
    yield_rows = []
    for nuclide in report.nuclides:
        for peak in nuclide.peaks:
            line_ref = f"{refs[1]}:line:{peak['source_line_number']}"
            historical = GammaYield(
                peak["rad_int"], "percent", line_ref, "historical_report"
            )
            # No modern evaluated library is acquired or silently substituted.
            yield_rows.append(
                dict(
                    isotope=nuclide.isotope,
                    energy_keV=peak["center_keV"],
                    historical_vs_modern=compare_gamma_yields(historical, None),
                    existing_bundled_reference_qc=qg_yield_diagnostic(
                        nuclide.isotope, peak["center_keV"], peak["rad_int"]
                    ),
                )
            )
            if peak["activity_unit"] != "uCi":
                raise ValueError(
                    "Bounded example requires the printed uCi activity unit"
                )
            point = report_effective_point(
                net_counts=peak["net_counts"],
                activity_bq=peak["activity"] * 37000,
                live_time_s=report.live_time,
                historical_yield=historical,
                report_ref=line_ref,
            )
            effective.append((peak["center_keV"], point))
    effective.sort(key=lambda x: x[0])
    effective_curve = report_effective_curve(
        [x[0] for x in effective],
        [x[1] for x in effective],
        geometry=geo,
        source_refs=refs,
    )
    alternatives = [
        source_table_curve(table, unit_assumption="fraction"),
        legacy_curve,
        density_model,
        unsupported,
        effective_curve,
    ]
    comparison = compare_efficiencies(
        source,
        alternatives,
        table.energies.tolist(),
        geometry=geo,
        selected_method=selected_method,
    )
    literal_refs = [
        ReferencePoint(
            f"source_holdout_{e}",
            e,
            v,
            table.sha256,
            "source_export_holdout",
            geo,
            1e-12,
        )
        for e, v in [
            (100, 0.0010023),
            (500, 0.000821244),
            (1000, 0.000444251),
            (1999, 0.00029946),
        ]
    ]
    controls = check_references(source, literal_refs, fitted_source_ids=[])
    if not all(c["control"]["passed"] for c in controls):
        raise ValueError("Literal source reference controls failed")
    # Reproduce every source knot, including preserved nonpositive exclusions.
    positive = [r for r in table.rows if r["efficiency_reported"] > 0]
    max_delta = max(
        abs(
            source.at(r["energy_keV"], geo)["efficiency_fraction"]
            - r["efficiency_reported"] / 100
        )
        for r in positive
    )
    fe = read_processed_txt(here / "fixtures/Fe-Cd-RAFM-1.txt")
    fe_geo = Geometry(
        "South HPGe", fe.efficiency.source_distance_cm, "near_contact_wire", True
    )
    near_contact = source.at(1099.2, fe_geo)
    if near_contact["admissible_for_comparison"]:
        raise ValueError("Near-contact Fe-Cd erroneously admitted")
    identity = source_identity(
        ROOT,
        ["src/fluxforge/physics/efficiency_fidelity.py"],
    )
    engine = identity["revision"]
    for name in [
        "physics/efficiency_fidelity.py",
        "analysis/qg_calibration.py",
        "io/flux_wire.py",
        "data/xcom.py",
        "data/rafm_decay.json",
    ]:
        path = ROOT / "src/fluxforge" / name
        if path.exists():
            inputs[f"src/fluxforge/{name}"] = sha(path)
    receipt = {
        "schema": "source-efficiency-fidelity-v1",
        "engine_commit": engine,
        "source_identity": identity,
        "required_engine_ancestor": "4615e61bbb262d974e0326a44107bc952e4cb903",
        "implementation_sha256": sha(
            ROOT / "src/fluxforge/physics/efficiency_fidelity.py"
        ),
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "inputs": inputs,
        "source_fixture_origin": manifest["origin"],
        "selected_method": selected_method,
        "source_header": header,
        "header_export_audit": header_audit,
        "saved_vs_report_conflict_control": conflict,
        "comparison_metadata": {k: v for k, v in comparison.items() if k != "rows"},
        "comparison_rows": len(comparison["rows"]),
        "statuses": dict(Counter(r["status"] for r in comparison["rows"])),
        "source_range_keV": list(source.identity.energy_range_keV),
        "source_rows": len(table.rows),
        "nonpositive_source_rows": len(table.rows) - len(positive),
        "maximum_positive_source_knot_delta_fraction": max_delta,
        "references": controls,
        "report_effective_points": effective,
        "gamma_yields": yield_rows,
        "near_contact_fe_cd": near_contact,
        "scientific_admission": False,
        "independent_absolute_calibration_validation": (
            "unavailable; source holds/algebra controls are not physical validation"
        ),
        "limits": [
            "Saved-header fields do not establish final-report settings",
            (
                "Percent export units are conditional; acquisition date/certificate/"
                "covariance/active identity unavailable"
            ),
            (
                "PGT manual v4.04.00 Appendix B; deployed version and "
                "proprietary McMaster details unavailable"
            ),
            (
                "CSV MuWin/MuGe fields are not accepted as independently identified "
                "attenuation tables"
            ),
            (
                "Local XCOM-labeled tables require separate source qualification; "
                "this example tests implementation only"
            ),
            (
                "Natural log is an explicit alternative convention, "
                "not a verified vendor implementation"
            ),
            (
                "The 0.5mL/25cm geometry for Co-Cd is an explicit nominal "
                "comparison assumption"
            ),
            (
                "Bundled decay_2012 QC is separate from modern evaluated yields, "
                "which are unavailable here"
            ),
            (
                "No activity adjustment, sample-specific fit or "
                "physical workflow integration performed"
            ),
        ],
    }
    # Read and calculate first; fresh output only, originals never modified.
    output.mkdir(parents=True, exist_ok=False)
    (output / "AUDIT.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "energy_deltas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "energy_keV",
            "method_id",
            "kind",
            "validation_status",
            "efficiency_fraction",
            "status",
            "admissible_for_comparison",
            "scientific_admission",
            "physical_qualified",
            "reference_method",
            "delta_fraction",
            "delta_relative_to_reference",
            "selected",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison["rows"])
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh output directory"
    )
    parser.add_argument(
        "--selected-method",
        required=True,
        choices=[
            "south_source_export_percent",
            "south_source_export_fraction",
            "current_engine_header_model",
            "pgt_shaped_xcom_density_aware_ln",
            "vendor_mcmaster_model",
            "historical_report_effective",
        ],
    )
    args = parser.parse_args()
    result = run(args.output, args.selected_method)
    print(
        json.dumps(
            {
                "engine_commit": result["engine_commit"],
                "rows": result["comparison_rows"],
                "source_knot_delta": result[
                    "maximum_positive_source_knot_delta_fraction"
                ],
                "scientific_admission": False,
            }
        )
    )
