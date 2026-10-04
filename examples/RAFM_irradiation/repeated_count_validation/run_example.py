"""Bounded, hash-bound three-count Ti diagnostic; no spectrum refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy
import scipy

from fluxforge.analysis.repeated_count_validation import (
    CountObservation,
    SIMPLE_DECAY,
    UNIFORM_ACCEPTANCE,
    compare_repeated_counts,
)
from fluxforge.io.flux_wire import read_processed_txt

HERE = Path(__file__).resolve().parent
ENGINE_BASE = "a7bcc680d1f5e06b1d9dae405241fc380087ca2b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bound_report(path, fixture):
    if digest(path) != fixture["sha256"] or path.stat().st_size != fixture["bytes"]:
        raise ValueError(
            "Source report hash/size mismatch: " + fixture["measurement_id"]
        )
    return read_processed_txt(path)


def run_example():
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ENGINE_BASE, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads((HERE / "fixtures.json").read_text(encoding="utf-8-sig"))
    groups, summaries, sources = {}, {}, []
    for fixture in manifest["reports"]:
        path = ROOT / fixture["local_path"]
        parsed = read_bound_report(path, fixture)
        # Compare the same printed response/geometry parameters; do not evaluate
        # or infer an absolute efficiency from agreement with target activities.
        calibration = hashlib.sha256(
            json.dumps(parsed.efficiency.to_dict(), sort_keys=True).encode()
        ).hexdigest()
        sources.append(
            {
                **fixture,
                "printed_id": parsed.sample_id,
                "count_start_unzoned": parsed.start_time.isoformat(),
                "live_time_s": parsed.live_time,
                "real_time_s": parsed.real_time,
                "printed_calibration_sha256": calibration,
                "reported_nuclides_and_lines": [n.to_dict() for n in parsed.nuclides],
            }
        )
        for n in parsed.nuclides:
            common = dict(
                measurement_id=fixture["measurement_id"],
                specimen_id=fixture["original_catalog_specimen_id"],
                nuclide=n.isotope,
                calibration_id=calibration,
                background_id=None,
                count_basis="vendor_comparison",
                source_sha256=fixture["sha256"],
                identity_basis="source catalog distinct count labels; shared specimen unqualified",
                half_life_s=n.half_life_seconds,
                count_start=parsed.start_time,
                real_time_s=parsed.real_time,
                live_time_s=parsed.live_time,
                exclusion_reason=(
                    "Sc48 report yields/references inconsistent; excluded without substitution"
                    if n.isotope == "Sc48"
                    else None
                ),
            )
            summaries.setdefault(n.isotope, []).append(
                CountObservation(
                    **common,
                    channel_id="vendor_isotope_summary",
                    value=n.activity_bq,
                    input_kind="reference_activity",
                    includes_count_decay=None,
                    activity_reference=None,
                )
            )
            for peak in n.peaks:
                key = peak["assignment"]
                row = CountObservation(
                    **common,
                    channel_id=key,
                    value=peak["net_counts"] / parsed.live_time,
                    input_kind="count_average",
                    includes_count_decay=False,
                    acceptance_assumption=UNIFORM_ACCEPTANCE,
                    activity_unit="response_scaled_counts_per_s",
                )
                groups.setdefault(key, []).append(row)
    strict = {
        k: compare_repeated_counts(v, relative_tolerance=0.05, decay_assumption=None)
        for k, v in groups.items()
    }
    # Hypothesis is deliberately separate from original identities/backgrounds.
    # Relative N/L decay comparisons cancel a time-independent same-line response;
    # they make no choice about the vendor's printed activity reference.
    conditional = {}
    for key, rows in groups.items():
        assumed = [
            replace(
                r,
                specimen_id="conditional_shared_ti_wire",
                identity_basis="explicit shared-wire hypothesis; original count IDs retained",
                background_id="conditional_same_vendor_background_scenario_unknown",
            )
            for r in rows
        ]
        conditional[key] = compare_repeated_counts(
            assumed,
            relative_tolerance=0.05,
            decay_assumption=SIMPLE_DECAY,
            common_naive_clock="QG header shared naive local clock scenario; time zone remains unknown",
        )
    report_reference = {
        k: compare_repeated_counts(v, relative_tolerance=0.05, decay_assumption=None)
        for k, v in summaries.items()
    }
    original_hashes_after = {
        f["measurement_id"]: digest(ROOT / f["local_path"]) for f in manifest["reports"]
    }
    if any(
        original_hashes_after[f["measurement_id"]] != f["sha256"]
        for f in manifest["reports"]
    ):
        raise ValueError("Original changed during diagnostic")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "schema": "bounded-ti-repeated-count-example-v1",
        "engine_base": ENGINE_BASE,
        "worktree_head": head,
        "code_sha256": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in [
                ROOT / "src/fluxforge/analysis/repeated_count_validation.py",
                ROOT / "src/fluxforge/physics/activation.py",
                Path(__file__),
                HERE / "fixtures.json",
                ROOT / "src/fluxforge/io/flux_wire.py",
            ]
        },
        "dependencies": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "physical_counts": 3,
        "comparison_scenario_counts": 3,
        "historical_reproduction_performed": False,
        "scientific_admission": False,
        "sources": sources,
        "original_hashes_after": original_hashes_after,
        "strict_source_validation": strict,
        "conditional_same_wire_same_clock_line_diagnostics": conditional,
        "processed_reference_validation": report_reference,
        "limits": [
            "Sc46 889.3/1120.5 and Sc47 159.4 same-line comparisons are conditionally admissible only under the recorded identity/clock/response/background/decay assumptions.",
            "Sc48 all comparisons excluded: ambiguous 1.00 report yields and inconsistent line activities remain unchanged.",
            "No qualified shared physical specimen identity, time zone, background applicability, exact vendor reference convention or irradiation history is established.",
            "5% symmetric log-ratio reporting tolerance is fixed for all lines, not fitted to QG activities; no significance or uncertainty claim.",
            "Activities with response_scaled_counts_per_s units are decay-normalized line rates, not Bq; no reaction rate or flux is emitted.",
            "Component checks do not establish #232/#220 full-workflow parity or physical qualification.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON receipt path; existing files are refused",
    )
    args = parser.parse_args()
    result = run_example()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v["status"]
                for k, v in result[
                    "conditional_same_wire_same_clock_line_diagnostics"
                ].items()
            },
            indent=2,
        )
    )
