"""Diagnostic RAFM ROI sweep; never changes production settings or reference IDs.

Cases are selected from a previous audit, so this is a diagnostic study, not a
blind sensitivity benchmark. Target energies always come from the independent
gamma library. Production detection significance stays at 2 sigma.
"""

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import fluxforge
from fluxforge.analysis.flux_wire_analysis import (
    analyze_raw_spectrum_targeted,
    build_gamma_library,
)
from fluxforge.examples.rafm_workflow import (
    build_generic_gamma_library,
    default_paths,
    load_rafm_example_metadata,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import read_raw_asc


VARIANTS = {
    "baseline": (4.0, 1, 0.0),
    "wider_sidebands": (4.0, 8, 0.5),
    "narrower_roi": (2.5, 1, 0.0),
    "narrower_roi_wider_sidebands": (2.5, 8, 0.5),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_library_line(row, library):
    lines = [line for line in library if line.isotope == row["reported_isotope"]]
    if not lines:
        raise ValueError(f"No independent library isotope: {row['reported_isotope']}")
    return min(
        lines, key=lambda line: abs(line.energy_keV - row["reported_energy_keV"])
    )


def trace_targeted_call(data, lines, options):
    """Observe uncertainty components without changing the detector function.

    Only the target function is traced, at its final net-area positivity check.
    Unsupported centroids rejected earlier never become positive observations.
    The source line must be unique; a refactor fails instead of hiding evidence.
    """
    function = analyze_raw_spectrum_targeted
    source, first_line = inspect.getsourcelines(function)
    checks = [
        first_line + i
        for i, line in enumerate(source)
        if line.strip() in {"if net <= 0.0:", "if net <= 0:"}
    ]
    if len(checks) != 1:
        raise RuntimeError("Cannot identify the detector's final net-area check")
    observations = []

    def local_trace(frame, event, arg):
        if event == "line" and frame.f_lineno == checks[0]:
            values = frame.f_locals
            observations.append(
                {
                    "isotope": values["line"].isotope,
                    "library_energy_keV": float(values["line"].energy_keV),
                    "fitted_energy_keV": float(values["peak_energy"]),
                    "channel": int(round(values["peak_channel"])),
                    **{
                        key: float(values[key])
                        for key in (
                            "net",
                            "net_unc",
                            "roi_net",
                            "roi_unc",
                            "fit_net",
                            "fit_unc",
                            "selected_net",
                            "selected_unc",
                        )
                    },
                    "standards_policy": values["standards_policy"],
                    "used_fit_area": bool(values["use_fit_net"]),
                }
            )
        return local_trace

    def global_trace(frame, event, arg):
        return local_trace if frame.f_code is function.__code__ else None

    candidates = []
    previous = sys.gettrace()
    try:
        sys.settrace(global_trace)
        detected = function(
            data, lines, low_significance_candidates=candidates, **options
        )
    finally:
        sys.settrace(previous)
    return detected, candidates, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if not Path(fluxforge.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("Import must come from the chosen worktree")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    cache = json.loads(args.native.read_text(encoding="utf-8"))
    for path, digest in cache["input_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Native input changed: {path}")
    paths = default_paths(root / "examples/RAFM_irradiation")
    metadata = load_rafm_example_metadata(paths.example_root)
    library, _ = build_generic_gamma_library(metadata)
    wire_library = build_gamma_library()
    calibration = workflow_profile_energy_calibration(metadata.config)
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=calibration,
        profile_name=metadata.config["profile_name"],
    ).spectrum
    samples = {
        item["qg_file"].replace("\\", "/").split("QG_processed_gamma_data/")[1]: item
        for item in cache["samples"]
    }
    cases = [
        row
        for row in audit["rows"]
        if row["status"]
        in ("reference_nondetection", "tentative_same_id", "tentative_native_same_id")
    ]
    # Strong controls guard against optimizing only the weak reference cases.
    for report, isotope, energy in (
        ("RAFM4/RAFM4-A_15dEOI.txt", "Cr51", 320.0),
        ("RAFM4/RAFM4-N_15dEOI.txt", "Ta182", 264.28),
        ("RAFM3/RAFM3-N_24hrEOI.txt", "Mn56", 846.8),
    ):
        cases.append(
            min(
                (
                    row
                    for row in audit["rows"]
                    if row["report"] == report
                    and row.get("expected_isotope", row["reported_isotope"]) == isotope
                ),
                key=lambda row: abs(row["reported_energy_keV"] - energy),
            )
        )
    records, case_records = [], []
    out.mkdir(parents=True)
    for case in cases:
        sample = samples.get(case["report"])
        info = dict(case)
        info["paired_raw_available"] = sample is not None
        # The corrected source is a control; use its reviewed expected identity.
        target_case = dict(
            case,
            reported_isotope=case.get("expected_isotope", case["reported_isotope"]),
        )
        lines = wire_library if case["report"].startswith("flux_wires/") else library
        line = nearest_library_line(target_case, lines)
        info["library_energy_keV"] = line.energy_keV
        info["reported_centroid_offset_keV"] = (
            case["reported_energy_keV"] - line.energy_keV
        )
        if sample is None:
            case_records.append(info)
            continue
        # Historical native results are independent of this new diagnostic sweep.
        cached = [
            p
            for p in sample["peaks"]
            if p["isotope"] == line.isotope
            and abs(p["energy_keV"] - line.energy_keV) <= 2.0
        ]
        info["previous_native_fits_at_library_energy"] = cached
        info["previous_native_candidates_at_library_energy"] = [
            p
            for p in sample.get("candidates", [])
            if p["isotope"] == line.isotope
            and abs(p["energy_keV"] - line.energy_keV) <= 2.0
        ]
        case_records.append(info)
        data = read_raw_asc(
            root / sample["raw_file"],
            energy_calibration_override=calibration,
            profile_name=metadata.config["profile_name"],
        )
        data.sample_id = sample["sample"]
        neighbors = [
            candidate
            for candidate in lines
            if abs(candidate.energy_keV - line.energy_keV) <= 12.0
        ]
        for variant, (width, sidebands, gap) in VARIANTS.items():
            failed_joint_fits = []
            detected, candidates, diagnostic = trace_targeted_call(
                data,
                neighbors,
                dict(
                    peak_threshold=2.0,
                    max_assignment_energy_delta_fwhm=1.0,
                    min_energy_keV=80.0,
                    max_energy_keV=3000.0,
                    background_spectrum=background,
                    profile_name=metadata.config["profile_name"],
                    counting_method="iec_tiered",
                    roi_width_fwhm=width,
                    background_width_channels=sidebands,
                    background_gap_fwhm=gap,
                    comparison_background_model="linear",
                    fit_diagnostics=failed_joint_fits,
                ),
            )
            matching = [
                p
                for p in detected + candidates
                if p.isotope == line.isotope
                and abs(p.energy_keV - line.energy_keV) <= 2.0
            ]
            observed = [
                p
                for p in diagnostic
                if p["isotope"] == line.isotope
                and p["library_energy_keV"] == line.energy_keV
            ]
            record = dict(
                report=case["report"],
                source_line_number=case["source_line_number"],
                original_status=case["status"],
                isotope=line.isotope,
                library_energy_keV=line.energy_keV,
                variant=variant,
                roi_width_fwhm=width,
                sideband_channels_each=sidebands,
                gap_fwhm=gap,
                detected=any(p in detected for p in matching),
                diagnostic=observed,
                neighbor_diagnostics=diagnostic,
                failed_joint_fits=failed_joint_fits,
                returned_neighbor_fits=[
                    {
                        "energy_keV": p.energy_keV,
                        "isotope": p.isotope,
                        "assignment_ambiguous": p.assignment_ambiguous,
                        "activity_estimation_state": p.activity_estimation_state,
                        "significance": p.significance,
                    }
                    for p in detected + candidates
                ],
            )
            if matching:
                peak = min(matching, key=lambda p: abs(p.energy_keV - line.energy_keV))
                record.update(
                    net_counts=peak.net_counts,
                    net_counts_unc=peak.net_counts_unc,
                    significance=peak.significance,
                    fitted_energy_keV=peak.energy_keV,
                )
            records.append(record)
        print(case["report"], line.isotope, line.energy_keV, flush=True)
    payload = dict(
        scientific_admission=False,
        scope="Reference-selected diagnostic; library-energy targets with neighbor fits, measured background and variance retained; no parameter tuning applied to production",
        cases=case_records,
        experiments=records,
        audit_sha256=sha(args.audit),
        native_sha256=sha(args.native),
        input_sha256=cache["input_sha256"],
        tool_sha256=sha(Path(__file__)),
        runtime_source_sha256={
            p.relative_to(root).as_posix(): sha(p)
            for p in (root / "src/fluxforge").rglob("*.py")
        },
    )
    (out / "parameter_sweep.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "report",
        "source_line_number",
        "original_status",
        "isotope",
        "library_energy_keV",
        "variant",
        "roi_width_fwhm",
        "sideband_channels_each",
        "gap_fwhm",
        "detected",
        "net_counts",
        "net_counts_unc",
        "significance",
        "fitted_energy_keV",
    ]
    with (out / "parameter_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
