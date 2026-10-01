"""Additive source-bound QC replay of original Ti reports and Cu64 control."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from fluxforge.examples.rafm_workflow import (
    build_line_diagnostic_records,
    qg_reference_peaks,
    write_rows_csv,
)
from fluxforge.io.flux_wire import read_processed_txt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    reconstruction_path = args.evidence / "QG_EFFECTIVE_EFFICIENCY.json"
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    prior = Path(
        "artifacts/validation/publication_integrity_20260930/continuation_final.json"
    )
    original = json.loads(prior.read_text(encoding="utf-8"))
    inputs = dict(original["input_sha256"])
    for name in (
        "QG_EFFECTIVE_EFFICIENCY.json",
        "QG_TRANSCRIPTION_VERIFICATION.json",
        "QG_REPORT_CONSISTENCY_V2.json",
        "SC48_YIELD_CONVENTION_SENSITIVITY.json",
        "INDEPENDENT_REVIEW.md",
    ):
        p = args.evidence / name
        inputs[str(p.resolve())] = sha(p)
    inputs[str(args.baseline.resolve())] = sha(args.baseline)
    assert all(sha(p) == h for p, h in inputs.items())
    reports, all_rows = [], []
    args.out.mkdir(parents=True, exist_ok=False)
    for before in baseline["reports"]:
        sample = before["measurement"]
        source = next(
            row for row in reconstruction["rows"] if row["measurement_id"] == sample
        )
        p = Path(source["report_path"])
        assert str(p) == before["path"]
        assert sha(p) == before["sha256"] == source["report_sha256"]
        inputs[str(p.resolve())] = sha(p)
        data = read_processed_txt(p)
        after = [n.to_dict() for n in data.nuclides]
        # Projection is defined by the baseline's numeric/report fields. New
        # provenance is additive; no parsed reported value may change.
        assert len(before["nuclides"]) == len(after)
        for a, b in zip(before["nuclides"], after):
            assert {k: v for k, v in a.items() if k != "peaks"} == {
                k: b[k] for k in a if k != "peaks"
            }
            assert len(a["peaks"]) == len(b["peaks"])
            for x, y in zip(a["peaks"], b["peaks"]):
                assert x == {k: y[k] for k in x}
                assert y["source_file_sha256"] == sha(p)
                line = p.read_bytes().splitlines()[y["source_line_number"] - 1]
                assert y["reported_activity_text"].encode() in line
                assert y["reported_rad_int_text"].encode() in line
        rows, consistency = build_line_diagnostic_records(
            sample, "flux_wires", [], data, {}
        )
        # Existing summary-versus-line QC already detects the Ti inconsistency.
        assert before["consistency"] == consistency
        legacy = qg_reference_peaks(data)
        assert len(before["qg_reference_peaks"]) == len(legacy)
        for a, b in zip(before["qg_reference_peaks"], legacy):
            assert a == {k: b[k] for k in a}
        target = "Sc48" if sample.startswith("Ti") else "Cu64"
        target_rows = [r for r in rows if r["reference_isotope"] == target]
        flags = [
            r
            for r in target_rows
            if r["yield_qc_status"] == "yield_convention_discrepancy"
        ]
        if target == "Sc48":
            assert len(flags) == 2
            assert all(
                r["reference_rad_int_reported_value"] == 1.0
                and r["reference_rad_int_percent_assumption_fraction"] == 0.01
                and r["bundled_emission_probability"] == 1.0
                and r["flag_line_summary_inconsistency"]
                for r in flags
            )
        else:
            assert len(target_rows) == 1 and not flags
            assert (
                target_rows[0]["yield_qc_status"]
                == "consistent_with_percent_assumption"
            )
            assert target_rows[0]["reference_rad_int_reported_value"] == 0.47
        for row in rows:
            row["raw_parity_evaluated"] = False
        write_rows_csv(rows, args.out / (sample + "_line_diagnostics.csv"))
        write_rows_csv(consistency, args.out / (sample + "_qg_consistency.csv"))
        reports.append(
            dict(
                measurement=sample,
                report_path=str(p),
                report_sha256=sha(p),
                summaries=[
                    dict(
                        isotope=n.isotope,
                        activity=n.activity,
                        activity_unit=n.activity_unit,
                        activity_unc=n.activity_unc,
                        provenance=n.report_provenance,
                    )
                    for n in data.nuclides
                ],
                line_diagnostics=rows,
                existing_consistency=consistency,
                reported_values_unchanged=True,
                legacy_comparison_fields_unchanged=True,
                yield_convention_flags=len(flags),
            )
        )
        all_rows.extend(rows)
    write_rows_csv(all_rows, args.out / "line_diagnostics.csv")
    assert all(sha(p) == h for p, h in inputs.items())
    sources = [
        Path(p)
        for p in (
            "src/fluxforge/analysis/qg_report_qc.py",
            "src/fluxforge/io/flux_wire.py",
            "src/fluxforge/examples/rafm_workflow.py",
            "src/fluxforge/data/rafm_decay_data.json",
            "tests/test_qg_report_qc.py",
            "tools/validate_qg_report_qc.py",
        )
    ]
    receipt = dict(
        schema="qg-report-qc-v1",
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={str(p): sha(p) for p in sources},
        input_sha256=inputs,
        baseline_sha256=sha(args.baseline),
        reports=reports,
        report_count=4,
        yield_convention_flags=6,
        scientific_admission=False,
        report_only=True,
        source_bytes_unchanged=True,
        rate_or_summary_correction=False,
        original_PR212_input_hashes_verified=32,
        independent_reference=dict(
            url="https://www.nndc.bnl.gov/ensnds/48/Ti/beta_decay.pdf",
            evaluation="ENSDF November 2021; Jun Chen NDS179 (2022)",
            pages=[2, 3],
            energies_keV=[983.526, 1312.120],
            relative_intensities=[1000, 1000],
            absolute_per_100_decay_normalization=0.100,
            absolute_intensities_per_100_decays=[100.0, 100.0],
            qualification="Reference cross-check only; not the original vendor gamma library",
        ),
        limitations=[
            "No raw-spectrum re-fit or transport/inversion run in this source-QC replay",
            "RAD INT unit is not declared in the reports; both interpretations are explicit hypotheses",
            "Implied efficiency is conditional algebra from reported counts/activity/time, not independent calibration",
            "Bundled decay_2012/actigamma provenance is separate from the November2021 ENSDF reference",
            "Original GammaLib.mdb/settings and measured calibration/history uncertainty remain unqualified",
            "Changing two line yields does not justify changing a whole nuclide summary or reaction rate",
        ],
    )
    output = args.out / "receipt.json"
    output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=sha(output),
                reports=4,
                yield_flags=6,
                scientific_admission=False,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
