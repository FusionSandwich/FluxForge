"""Run RAFM3/4 raw recovery without QG count/activity substitution.

QG reports are read only after extraction, for comparison. The receipt records
discrepancies; running successfully is not evidence of measurement accuracy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fluxforge.examples.rafm_workflow as workflow
from fluxforge.io.flux_wire import read_raw_asc


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_digest(artifact):
    return hashlib.sha256(
        json.dumps(
            {
                "peaks": artifact["peaks"],
                "isotopes": artifact["isotopes"],
                "fit_diagnostics": artifact.get("targeted_fit_diagnostics", []),
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def replay_sample(
    raw, qg, metadata, paths, result_tree, background, library, half_lives, sample_key
):
    if raw.parent.name == "flux_wires":
        return workflow.analyze_flux_wire_sample(
            raw, metadata, paths, result_tree, background, qg, sample_key
        )
    return workflow.analyze_generic_sample(
        raw, metadata, paths, result_tree, library, half_lives, background, qg
    )


def validate_checkpoint(previous, current, selected, output, original_driver_hash):
    """Reuse only a verified prefix with unchanged scientific code and inputs."""
    driver = "tools/audit_rafm_raw_recovery.py"
    assert previous["source_sha256"].keys() == current["source_sha256"].keys()
    for path, expected in previous["source_sha256"].items():
        if path == driver and expected != current["source_sha256"][path]:
            assert original_driver_hash == expected, "Unreviewed original audit driver"
        else:
            assert current["source_sha256"][path] == expected, path
    for key in ("metadata_sha256", "background_sha256", "effective_configuration"):
        assert previous[key] == current[key], key
    for name, expected in current.get("runtime_data_sha256", {}).items():
        if "runtime_data_sha256" in previous:
            assert previous["runtime_data_sha256"][name] == expected, name
        else:
            # Legacy checkpoints lack the external profile byte hash. Require
            # its semantic identity with the original launch's committed file.
            original = subprocess.check_output(
                ["git", "show", f"{previous['git_head']}:{name}"], cwd=ROOT
            )
            assert json.loads(original) == json.loads((ROOT / name).read_bytes()), name
    completed = previous["samples"]
    assert len(completed) <= len(selected)
    for row, (raw, qg) in zip(completed, selected):
        assert row["sample"] == raw.stem
        assert row["raw_sha256"] == digest(raw)
        assert row["qg_sha256"] == (digest(qg) if qg else None)
        artifact_path = output / "raw_iec/analysis_json" / f"{raw.stem}.json"
        assert Path(row["artifact"]).resolve() == artifact_path.resolve()
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert prediction_digest(artifact) == row["prediction_sha256"]
        assert artifact["validation"] == row["validation"]
        assert artifact["measurement_time_audit"] == row["measurement_time_audit"]
        assert artifact["validation"]["reference_used_for_analysis"] is False
        for key in ("n_detected_peaks", "n_unidentified_peaks"):
            assert row[key] == artifact[key], key
        assert row["n_ambiguous_peaks"] == artifact.get("n_ambiguous_peaks", 0)
        assert row["isotopes"] == sorted(artifact["isotopes"])
        assert row["targeted_fit_diagnostics"] == artifact.get(
            "targeted_fit_diagnostics", []
        )
        assert row["report"] == artifact["comparison_report_txt"]
        if current.get("runtime_data_sha256"):
            profiles = json.loads(
                (ROOT / "src/fluxforge/data/rafm_profiles.json").read_bytes()
            )
            profile = profiles[current["effective_configuration"]["profile_name"]]
            effective = artifact["analysis_configuration"]
            assert (
                effective["energy_calibration_coefficients"]
                == profile["energy_calibration"]
            )
            assert effective["resolution_coefficients"] == profile["resolution"]
            for key, value in profile["efficiency"].items():
                assert effective["efficiency_parameters"][key] == value, key
    if completed:
        first = completed[0]
        assert first["withheld_report_prediction_identical"]
        withheld_path = (
            output / "withheld_report/analysis_json" / f"{first['sample']}.json"
        )
        withheld = json.loads(withheld_path.read_text(encoding="utf-8"))
        assert prediction_digest(withheld) == first["prediction_sha256"]
        assert withheld["validation"]["passed"] is None
        assert withheld["validation"]["reference_used_for_analysis"] is False
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and resume an interrupted checkpoint",
    )
    parser.add_argument(
        "--original-driver-sha256",
        help="Reviewed original driver hash when only the audit driver changed",
    )
    parser.add_argument(
        "--reviewed-driver-sha256", help="Pin the reviewed resume driver bytes"
    )
    parser.add_argument(
        "--all-raw",
        action="store_true",
        help="Include all committed raw RAFM and flux-wire specimens",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-spectra", type=int)
    selection.add_argument(
        "--sample", action="append", help="Replay a named specimen; may be repeated"
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    example = ROOT / "examples" / "RAFM_irradiation"
    metadata = workflow.load_rafm_example_metadata(example)
    metadata.config["generic_targeted_counting_method"] = "iec_tiered"
    metadata.config["flux_wire_counting_method"] = "iec_tiered"
    paths = workflow.default_paths(example, results_root=output / "raw_iec")
    library, half_lives = workflow.build_generic_gamma_library(metadata)
    files = workflow.discover_input_files(paths)
    pairs, _, unmatched_qg = workflow.pair_input_files(
        files["raw"], files["qg"], metadata.pairing_aliases
    )
    sample_keys = {raw: key for raw, _, key in pairs}
    selected = [
        (raw, qg) for raw, qg, _ in pairs if raw.parent.name in {"RAFM3", "RAFM4"}
    ]
    assert len(selected) == 16, "Expected the committed 12 RAFM3 and 4 RAFM4 specimens"
    if args.all_raw:
        selected = [(raw, qg) for raw, qg, _ in pairs]
        assert len(selected) == 29, "Expected all 29 committed raw spectra"
    if args.sample:
        requested = set(args.sample)
        available = {raw.stem for raw, _ in selected}
        if requested - available:
            raise ValueError(
                f"Unknown RAFM3/4 specimens: {sorted(requested - available)}"
            )
        selected = [(raw, qg) for raw, qg in selected if raw.stem in requested]
    if args.max_spectra is not None:
        if args.max_spectra <= 0:
            raise ValueError("max-spectra must be positive")
        selected = selected[: args.max_spectra]
    if args.resume:
        assert args.reviewed_driver_sha256 == digest(
            Path(__file__)
        ), "Resume driver changed after review"
        assert output.is_dir()
        assert not (
            output / "raw_recovery_receipt.json"
        ).exists(), "Run already terminal"
    else:
        output.mkdir(parents=True, exist_ok=False)
    tree = workflow.ensure_results_tree(paths.results_root)
    background = read_raw_asc(
        paths.background_path, profile_name=metadata.config["profile_name"]
    ).spectrum
    assert background is not None

    def forbidden_reference_substitution(*args, **kwargs):
        raise AssertionError(
            "Raw audit must never copy QG counts or isotope activities"
        )

    workflow.apply_generic_qg_report_parity = forbidden_reference_substitution
    workflow.reference_isotope_payload = forbidden_reference_substitution
    from fluxforge.analysis import flux_wire_analysis

    flux_wire_analysis.apply_qg_report_parity = forbidden_reference_substitution

    def analyze(raw, qg, result_tree):
        return replay_sample(
            raw,
            qg,
            metadata,
            paths,
            result_tree,
            background,
            library,
            half_lives,
            sample_keys[raw],
        )

    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            name: digest(ROOT / name)
            for name in [
                "src/fluxforge/examples/rafm_workflow.py",
                "src/fluxforge/analysis/flux_wire_analysis.py",
                "src/fluxforge/analysis/peakfit.py",
                "src/fluxforge/analysis/spectrum_math.py",
                "src/fluxforge/io/spe.py",
                "src/fluxforge/io/flux_wire.py",
                "src/fluxforge/data/rafm_profile.py",
                "tools/audit_rafm_raw_recovery.py",
            ]
        },
        "metadata_sha256": {
            path.name: digest(path)
            for path in sorted(paths.metadata_root.glob("*.json"))
        },
        "background_sha256": digest(paths.background_path),
        "runtime_data_sha256": {
            "src/fluxforge/data/rafm_profiles.json": digest(
                ROOT / "src/fluxforge/data/rafm_profiles.json"
            )
        },
        "effective_configuration": metadata.config,
        "samples": [],
        "limits": [
            "QG is a comparison report, not ground truth; stage, calibration and systematic uncertainty require review.",
            "Configured profile energy calibration and counting methods remain unchanged apart from selecting raw iec_tiered.",
            "Raw extraction complete does not mean isotope identification or absolute activities are qualified.",
        ],
    }
    if args.resume:
        previous = json.loads(
            (output / "raw_recovery_progress.json").read_text(encoding="utf-8")
        )
        receipt["samples"] = validate_checkpoint(
            previous, receipt, selected, output, args.original_driver_sha256
        )
        receipt["resumed_from"] = {
            "timestamp_utc": previous["timestamp_utc"],
            "git_head": previous["git_head"],
            "source_sha256": previous["source_sha256"],
            "completed_samples": len(receipt["samples"]),
            "prior_resume": previous.get("resumed_from"),
        }
    for index, (raw, qg) in enumerate(selected):
        if index < len(receipt["samples"]):
            continue
        assert qg is None or qg.is_file()
        artifact = analyze(raw, qg, tree)
        basis = artifact["validation"]["comparison_basis"]
        assert basis in {"raw_estimate_vs_report", "not_evaluated"}
        if not qg:
            assert basis == "not_evaluated"
        if basis == "not_evaluated":
            assert artifact["validation"]["passed"] is not True
            assert not artifact["validation"].get("available_comparison_domains")
        assert artifact["validation"]["reference_used_for_analysis"] is False
        zero_width = [p for p in artifact["peaks"] if p["fwhm_keV"] == 0.0]
        assert (
            not zero_width
        ), "A reference-only zero-width peak leaked into raw recovery"
        row = {
            "sample": raw.stem,
            "raw_sha256": digest(raw),
            "qg_sha256": digest(qg) if qg else None,
            "prediction_sha256": prediction_digest(artifact),
            "n_detected_peaks": artifact["n_detected_peaks"],
            "n_unidentified_peaks": artifact["n_unidentified_peaks"],
            "validation": artifact["validation"],
            "n_ambiguous_peaks": artifact.get("n_ambiguous_peaks", 0),
            "measurement_time_audit": artifact["measurement_time_audit"],
            "targeted_fit_diagnostics": artifact.get("targeted_fit_diagnostics", []),
            "isotopes": sorted(artifact["isotopes"]),
            "artifact": str(tree["artifacts"] / f"{raw.stem}.json"),
            "report": artifact["comparison_report_txt"],
        }
        # For the first actual specimen, withholding the comparison report
        # must leave every predicted peak and isotope result identical.
        if index == 0:
            withheld_tree = workflow.ensure_results_tree(output / "withheld_report")
            withheld = analyze(raw, None, withheld_tree)
            assert prediction_digest(withheld) == row["prediction_sha256"]
            assert withheld["validation"]["passed"] is None
            row["withheld_report_prediction_identical"] = True
        receipt["samples"].append(row)
        (output / "raw_recovery_progress.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "sample": raw.stem,
                    "peaks": row["n_detected_peaks"],
                    "unidentified": row["n_unidentified_peaks"],
                    "missing": len(row["validation"]["missing_peaks"]),
                    "raw_comparison_passed": row["validation"]["passed"],
                }
            ),
            flush=True,
        )
    receipt["status"] = "RUN_COMPLETE_REVIEW_REQUIRED"
    receipt["n_spectra"] = len(selected)
    receipt["all_committed_rafm3_rafm4_replayed"] = (
        sum(raw.parent.name in {"RAFM3", "RAFM4"} for raw, _ in selected) == 16
    )
    receipt["all_committed_raw_replayed"] = len(selected) == 29
    receipt["comparison_summary"] = workflow.summarize_validation_artifacts(
        [
            {"sample_id": row["sample"], "validation": row["validation"]}
            for row in receipt["samples"]
        ],
        unmatched_qg=unmatched_qg if args.all_raw else (),
    )
    target = output / "raw_recovery_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": str(target),
                "n_spectra": len(selected),
                "status": receipt["status"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
