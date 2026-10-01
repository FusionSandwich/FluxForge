"""Offline all-report/ROI source-QC replay using the authorized raw curve.

No activities/rates are corrected and no missing uncertainty is synthesized.
Output is additive; private mail and workbook contents are never copied.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import subprocess

import numpy as np

from fluxforge.analysis.qg_calibration import QGEfficiencyTable
from fluxforge.analysis.qg_report_qc import qg_yield_diagnostic
from fluxforge.examples.rafm_workflow import write_rows_csv
from fluxforge.uncertainty.reaction_rate_budget import (
    RateUncertaintyBudget,
    UncertaintyComponent,
    rate_covariance,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--count-matrix", type=Path, required=True)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    curve_dir = Path("examples/RAFM_irradiation/calibration")
    curve = QGEfficiencyTable(
        curve_dir / "South Small Vial 25cm.csv",
        curve_dir / "South Small Vial 25cm.provenance.json",
    )
    roi_path = args.reconciliation / "report_roi_rows.csv"
    inputs = {
        str(p.resolve()): sha(p)
        for p in [
            args.inventory,
            roi_path,
            args.count_matrix,
            args.operator,
            curve.path,
            curve_dir / "South Small Vial 25cm.provenance.json",
        ]
    }
    for p, record in inventory["manifest_sources"].items():
        if "sha256" in record:
            assert sha(p) == record["sha256"], p
            inputs[p] = record["sha256"]
    matrix = json.loads(args.count_matrix.read_text())
    counts = {r["measurement_id"]: r for r in matrix["counts"]}
    reports = []
    source_by_sha = {}
    common_exclusions = [
        "original_gamma_library_settings_unqualified",
        "active_calibration_certificate_covariance_unqualified",
        "complete_campaign_operating_log_unavailable",
        "irradiation_timezone_clock_join_unqualified",
        "full_rate_uncertainty_unqualified",
    ]
    for rec in inventory["reports"]:
        p = Path(rec["source_path"])
        assert sha(p) == rec["provenance_sha256"] == rec["actual_sha256"]
        inputs[str(p)] = sha(p)
        source_by_sha[sha(p)] = p
        report = dict(
            rec,
            diagnostic_use="all source ROI rows retained",
            physical_included=False,
            physical_exclusion_reasons=common_exclusions.copy(),
        )
        for companion in rec["raw_companions"].values():
            if companion["available"]:
                assert sha(companion["path"]) == companion["sha256"]
                inputs[companion["path"]] = companion["sha256"]
        asc = rec["raw_companions"]["ASC"]
        if asc["available"]:
            header_bytes = (
                Path(asc["path"]).read_bytes().split(b"Channel Contents", 1)[0]
            )
            report["ASC_header_observation"] = {
                "inspection": "ASCII prefix before Channel Contents, empirical label observation",
                "efficiency_constants_label_present": b"Efficiency" in header_bytes,
                "physical_calibration_qualified": False,
            }
        ans = rec["raw_companions"]["ANS"]
        if ans["available"]:
            native_prefix = Path(ans["path"]).read_bytes()[:1548]
            report["ANS_prefix_observation"] = {
                "inspection": "first 1548 bytes only; empirical ASCII labels, no vendor binary schema inferred",
                "South_label_present": b"South" in native_prefix,
                "Energy_label_present": b"Energy" in native_prefix,
                "GammaLib_mdb_label_present": b"GammaLib.mdb" in native_prefix,
                "coefficient_or_uncertainty_binary_fields_identified": False,
                "physical_calibration_qualified": False,
            }
        reports.append(report)
    roi = list(csv.DictReader(roi_path.open(encoding="utf-8", newline="")))
    assert len(reports) == 31 and len(roi) == 288
    assert sum(r["ROI_count"] for r in reports) == len(roi)
    diagnostic = []
    for rec in roi:
        path = source_by_sha[rec["report_sha256"]]
        actual_line = (
            path.read_bytes()
            .splitlines()[int(rec["line_number"]) - 1]
            .decode("utf-8", errors="replace")
        )
        assert actual_line == rec["original_line"]
        count = counts[rec["measurement_id"]]
        header = count["QG_report_header"]
        nominal = (
            header["detector_id"] == "South"
            and header["source_distance_cm_printed"] == 25
            and all(
                count.get("candidate_C1_C4_A_within_QG_printed_rounding", {}).values()
            )
        )
        energy = (
            float(rec["assignment_energy_keV"])
            if rec["assignment_energy_keV"]
            else None
        )
        rad = float(rec["rad_int_printed"]) if rec["rad_int_printed"] else None
        yield_qc = qg_yield_diagnostic(
            rec["nuclide"],
            energy if energy is not None else float("nan"),
            rad if rad is not None else float("nan"),
        )
        curve_qc = (
            curve.diagnostic_at(energy, unit_assumption="percent")
            if energy is not None
            else {"status": "excluded_missing_line_energy", "efficiency_fraction": None}
        )
        activity = float(rec["roi_activity_uCi"]) if rec["roi_activity_uCi"] else None
        net = float(rec["net_counts"]) if rec["net_counts"] else None
        live = header["live_s"]
        implied = None
        if all(
            v is not None and math.isfinite(v) and v > 0
            for v in (activity, net, rad, live)
        ):
            implied = net / (37000 * activity * live * (rad / 100))
        diagnostic.append(
            dict(
                measurement_id=rec["measurement_id"],
                report_sha256=rec["report_sha256"],
                source_line=int(rec["line_number"]),
                nuclide=rec["nuclide"],
                energy_keV=energy,
                raw_rad_int=rad,
                raw_line_activity_uCi=activity,
                activity_corrected=False,
                yield_status=yield_qc["yield_qc_status"],
                curve_status=(
                    curve_qc["status"]
                    if nominal
                    else "excluded_separate_or_unqualified_profile"
                ),
                nominal_detector_distance_coefficient_match=nominal,
                nominal_match_note="DI numeric mismatch, geometry, active identity and unit/version applicability remain unqualified",
                curve_efficiency_fraction_if_percent=(
                    curve_qc["efficiency_fraction"] if nominal else None
                ),
                conditional_effective_efficiency_if_rad_percent=implied,
                conditional_effective_efficiency_if_rad_fraction=(
                    implied / 100 if implied is not None else None
                ),
                efficiency_standard_uncertainty=None,
                scientific_admission=False,
            )
        )
    args.out.mkdir(parents=True, exist_ok=False)
    write_rows_csv(curve.rows, args.out / "curve_rows_qc.csv")
    write_rows_csv(diagnostic, args.out / "all_roi_diagnostics.csv")
    operator = json.loads(args.operator.read_text())
    baseline = json.loads(
        Path(
            "artifacts/validation/publication_integrity_20260930/operator_final/receipt.json"
        ).read_text()
    )
    assert len(operator["sources"]) == len(baseline["sources"]) == 13
    budgets = []
    for after, before in zip(operator["sources"], baseline["sources"]):
        assert after["observation_id"] == before["observation_id"]
        assert after["replayed_rate"] == before["replayed_rate"]
        assert after["replayed_rate_unc"] == before["replayed_rate_unc"]
        data = dict(after["rate_uncertainty_budget"])
        data["components"] = [UncertaintyComponent(**c) for c in data["components"]]
        budgets.append(RateUncertaintyBudget(**data))
    for p, h in operator["input_sha256"].items():
        assert sha(p) == h
        inputs[p] = h
    covariance = rate_covariance(budgets)
    assert covariance.shape == (13, 13)
    assert all(len(b.missing) == 6 for b in budgets)
    try:
        rate_covariance(budgets, require_complete=True)
    except ValueError as exc:
        strict_error = str(exc)
    else:
        raise AssertionError("Unknown actual uncertainties passed strict gate")
    background = Path("examples/RAFM_irradiation/background.ASC")
    inputs[str(background.resolve())] = sha(background)
    named_inputs = [
        dict(
            role="shared_background",
            path=str(background),
            sha256=sha(background),
            physical_qualified=False,
            exclusion_reason="background acquisition/applicability/covariance unqualified",
        )
    ]
    for p in sorted(Path("examples/RAFM_irradiation/metadata").glob("*.json")):
        inputs[str(p.resolve())] = sha(p)
        named_inputs.append(
            dict(
                role="maintained_schedule_or_metadata",
                path=str(p),
                sha256=sha(p),
                physical_qualified=False,
                exclusion_reason="legacy timing/geometry assumptions; correspondence Aug5 chronology does not qualify Aug4 maintained schedule",
            )
        )
    nuclear = Path("src/fluxforge/data/rafm_decay_data.json")
    inputs[str(nuclear.resolve())] = sha(nuclear)
    named_inputs.append(
        dict(
            role="bundled_decay_reference",
            path=str(nuclear),
            sha256=sha(nuclear),
            physical_qualified=False,
            exclusion_reason="curated decay_2012/actigamma; active vendor data and full nuclear covariance unqualified",
        )
    )
    additional_roles = []
    recovery = inventory.get("raw_recovery", {})
    for rec in recovery.get("outside_current_campaign_ASC", []):
        assert sha(rec["path"]) == rec["sha256"]
        inputs[rec["path"]] = rec["sha256"]
        additional_roles.append(
            dict(
                rec,
                role="historical ASC outside current observation hash set",
                physical_included=False,
            )
        )
    for rec in inventory.get("additional_native_measurements", []):
        for kind in ("ASC", "ANS"):
            assert sha(rec[kind]["path"]) == rec[kind]["sha256"]
            inputs[rec[kind]["path"]] = rec[kind]["sha256"]
        additional_roles.append(
            dict(
                rec,
                physical_included=False,
                exclusion_reason="No corroborated QG report; native/array source identity only",
            )
        )
    assert all(sha(p) == h for p, h in inputs.items())
    source_paths = [
        "src/fluxforge/analysis/qg_calibration.py",
        "src/fluxforge/analysis/qg_report_qc.py",
        "src/fluxforge/analysis/flux_unfold.py",
        "src/fluxforge/examples/rafm_workflow.py",
        "src/fluxforge/uncertainty/reaction_rate_budget.py",
        "src/fluxforge/physics/operating_history.py",
        "tools/validate_uwnr_data_use.py",
        "tools/validate_monitor_response_rafm.py",
        "tests/test_uwnr_source_inputs.py",
    ]
    receipt = dict(
        schema="uwnr-data-use-v1",
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        input_sha256=inputs,
        source_sha256={p: sha(p) for p in source_paths},
        reports=reports,
        named_inputs=named_inputs,
        calibration=curve.provenance,
        curve_rows=len(curve.rows),
        curve_nonpositive_rows=int(np.sum(curve.values <= 0)),
        roi_diagnostic_status_counts=dict(
            Counter(r["curve_status"] for r in diagnostic)
        ),
        report_count=31,
        roi_count=288,
        summary_count=sum(r["summary_count"] for r in reports),
        diagnostic_reports_included=31,
        diagnostic_roi_included=288,
        physical_reports_included=0,
        physical_roi_included=0,
        raw_ASC_available=sum(r["raw_companions"]["ASC"]["available"] for r in reports),
        raw_ANS_available=sum(r["raw_companions"]["ANS"]["available"] for r in reports),
        campaign_ANS_available=recovery.get(
            "ans_rows_hashverified_against_acquisition"
        ),
        campaign_ASC_available=recovery.get(
            "asc_rows_hashverified_against_acquisition"
        ),
        raw_missing_both=[
            r["measurement_id"]
            for r in reports
            if not any(x["available"] for x in r["raw_companions"].values())
        ],
        actual_rate_budgets=[
            dict(
                row_id=b.row_id,
                rate=b.rate,
                components=[vars(c) for c in b.components],
                missing=b.missing,
                diagnostic_assumptions=b.diagnostic_assumptions,
                irradiation_log_binding=b.irradiation_log_binding,
            )
            for b in budgets
        ],
        covariance_shape=list(covariance.shape),
        rate_covariance=covariance.tolist(),
        measured_calibration_covariance=None,
        measured_history_covariance=None,
        shared_source_ids=sorted(
            {
                c.correlation_group
                for b in budgets
                for c in b.components
                if c.correlation_group
            }
        ),
        strict_unknown_uncertainty_error=strict_error,
        unavailable_inputs=common_exclusions
        + [
            "native GammaLib.mdb contents, ANH configuration and vendor binary field meanings unavailable"
        ],
        additional_source_roles=additional_roles,
        repeated_count_identity={
            "Ti-RAFM-1/Ti-RAFM-1a/Ti-RAFM-1b": {
                "role": "repeat counts of one irradiated monitor; no independent activation information inferred",
                "shared_calibration_covariance": "unknown; repeated observations do not create independent calibration sources",
            }
        },
        private_operating_workbook=dict(
            public_role="Historical cumulative/snapshot context, not campaign power history",
            sha256="62119bd38ccf36f19e50a0eb1d3ce6c5776c0c08bfbffa6742551d8d7a38d486",
            bytes=64867,
            dated_records=680,
            coverage_start="2009-09-17",
            coverage_end="2025-04-28",
            campaign_date_covered=False,
            intraday_history_available=False,
            rod_columns="Physical snapshot positions in inches; not a power time series",
            contents_published=False,
            coverage_basis="Source-owner independently checked receipt; workbook contents are not read by this replay",
        ),
        nuclear_references=[
            dict(url=u, role=role, local_bytes_bound=False)
            for u, role in [
                (
                    "https://www.nndc.bnl.gov/ensnds/48/Ti/beta_decay.pdf",
                    "ENSDF Nov2021 yield cross-check; not vendor library",
                ),
                (
                    "https://nds.iaea.org/sgnucdat/safeg2008.pdf",
                    "Historical percent yields; no independent current covariance",
                ),
                (
                    "https://ludlums.com/images/product_manuals/QTMmanual.pdf",
                    "Historical Quantum4.04 processing; deployed applicability unknown",
                ),
                (
                    "https://doi.org/10.59161/JCGM102-2011",
                    "First-order source covariance propagation; nonlinear validation separate",
                ),
            ]
        ],
        raw_parity_recomputed=False,
        rate_values_unchanged=True,
        rate_or_summary_correction=False,
        scientific_admission=False,
        limitations=[
            "Conditional curve/yield algebra is not independent calibration",
            "No numerical uncertainty inferred from vendor Error, scatter, mail, workbook snapshots or missing logs",
            "All assumed model terms remain diagnostic; actual full shared covariance remains unavailable",
        ],
    )
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                receipt_sha256=sha(args.out / "receipt.json"),
                reports=31,
                roi=288,
                invalid_curve_rows=receipt["curve_nonpositive_rows"],
                covariance_shape=[13, 13],
                scientific_admission=False,
            )
        )
    )


if __name__ == "__main__":
    main()
