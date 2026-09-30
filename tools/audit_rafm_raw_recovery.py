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
            {"peaks": artifact["peaks"], "isotopes": artifact["isotopes"]},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-spectra", type=int)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    example = ROOT / "examples" / "RAFM_irradiation"
    metadata = workflow.load_rafm_example_metadata(example)
    metadata.config["generic_targeted_counting_method"] = "iec_tiered"
    paths = workflow.default_paths(example, results_root=output / "raw_iec")
    tree = workflow.ensure_results_tree(paths.results_root)
    library, half_lives = workflow.build_generic_gamma_library(metadata)
    files = workflow.discover_input_files(paths)
    pairs, _, _ = workflow.pair_input_files(
        files["raw"], files["qg"], metadata.pairing_aliases
    )
    selected = [
        (raw, qg) for raw, qg, _ in pairs if raw.parent.name in {"RAFM3", "RAFM4"}
    ]
    assert len(selected) == 16, "Expected the committed 12 RAFM3 and 4 RAFM4 specimens"
    if args.max_spectra is not None:
        if args.max_spectra <= 0:
            raise ValueError("max-spectra must be positive")
        selected = selected[: args.max_spectra]
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
            ]
        },
        "metadata_sha256": {
            path.name: digest(path)
            for path in sorted(paths.metadata_root.glob("*.json"))
        },
        "background_sha256": digest(paths.background_path),
        "effective_configuration": metadata.config,
        "samples": [],
        "limits": [
            "QG is a comparison report, not ground truth; stage, calibration and systematic uncertainty require review.",
            "Configured profile energy calibration and counting methods remain unchanged apart from selecting raw iec_tiered.",
            "Raw extraction complete does not mean isotope identification or absolute activities are qualified.",
        ],
    }
    for index, (raw, qg) in enumerate(selected):
        assert qg is not None and qg.is_file()
        artifact = workflow.analyze_generic_sample(
            raw, metadata, paths, tree, library, half_lives, background, qg
        )
        assert artifact["validation"]["comparison_basis"] == "raw_estimate_vs_report"
        assert artifact["validation"]["reference_used_for_analysis"] is False
        zero_width = [p for p in artifact["peaks"] if p["fwhm_keV"] == 0.0]
        assert (
            not zero_width
        ), "A reference-only zero-width peak leaked into raw recovery"
        row = {
            "sample": raw.stem,
            "raw_sha256": digest(raw),
            "qg_sha256": digest(qg),
            "prediction_sha256": prediction_digest(artifact),
            "n_detected_peaks": artifact["n_detected_peaks"],
            "n_unidentified_peaks": artifact["n_unidentified_peaks"],
            "validation": artifact["validation"],
            "isotopes": sorted(artifact["isotopes"]),
            "artifact": str(tree["artifacts"] / f"{raw.stem}.json"),
            "report": artifact["comparison_report_txt"],
        }
        # For the first actual specimen, withholding the comparison report
        # must leave every predicted peak and isotope result identical.
        if index == 0:
            withheld_tree = workflow.ensure_results_tree(output / "withheld_report")
            withheld = workflow.analyze_generic_sample(
                raw,
                metadata,
                paths,
                withheld_tree,
                library,
                half_lives,
                background,
                None,
            )
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
    receipt["all_committed_rafm3_rafm4_replayed"] = len(selected) == 16
    receipt["comparison_summary"] = workflow.summarize_validation_artifacts(
        [
            {"sample_id": row["sample"], "validation": row["validation"]}
            for row in receipt["samples"]
        ]
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
