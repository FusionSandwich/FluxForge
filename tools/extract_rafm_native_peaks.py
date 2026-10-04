import argparse
import hashlib
import fluxforge
from pathlib import Path
import json
from fluxforge.examples.rafm_workflow import (
    default_paths,
    load_rafm_example_metadata,
    workflow_profile_energy_calibration,
    build_generic_gamma_library,
    discover_input_files,
    pair_input_files,
    select_generic_targeted_lines,
    merge_detected_and_targeted_peaks,
    qg_reference_peaks,
)
from fluxforge.analysis.flux_wire_analysis import (
    analyze_raw_spectrum,
    analyze_raw_spectrum_targeted,
    build_gamma_library,
    get_expected_isotopes,
)
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.io.flux_wire import read_raw_asc, read_processed_txt


def main():
    parser = argparse.ArgumentParser(
        description="Extract native RAFM peak IDs without a Quantum Gold overlay."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not Path(fluxforge.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("Import must come from the chosen worktree")
    output.parent.mkdir(parents=True, exist_ok=True)
    example = root / "examples/RAFM_irradiation"
    paths = default_paths(example)
    m = load_rafm_example_metadata(example)
    c = m.config
    cal = workflow_profile_energy_calibration(c)
    bg = read_raw_asc(
        paths.background_path,
        energy_calibration_override=cal,
        profile_name=c["profile_name"],
    ).spectrum
    lib, _ = build_generic_gamma_library(m)
    files = discover_input_files(paths)
    pairs, ur, uq = pair_input_files(files["raw"], files["qg"], m.pairing_aliases)
    input_files = (
        files["raw"]
        + files["qg"]
        + [paths.background_path]
        + list(paths.metadata_root.glob("*.json"))
    )
    input_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in input_files
    }
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "src/fluxforge").rglob("*.py")
    }
    rows = []
    for raw, qg, key in pairs:
        if qg is None or (args.sample and raw.stem not in args.sample):
            continue
        d = read_raw_asc(
            raw, energy_calibration_override=cal, profile_name=c["profile_name"]
        )
        d.sample_id = raw.stem
        wire = raw.parent.name == "flux_wires"
        if wire:
            detected = []
            targets = build_gamma_library(
                isotope_filter=get_expected_isotopes(raw.stem)
            )
        else:
            adj = subtract_measured_background(
                d.spectrum, bg, mode="live", negative_policy="hybrid", warn_missing=True
            )
            detected = analyze_raw_spectrum(
                adj,
                efficiency=d.efficiency,
                gamma_library=lib,
                peak_threshold=c["peak_significance_sigma"],
                min_energy_keV=c["min_peak_energy_keV"],
                max_energy_keV=c["max_peak_energy_keV"],
                background_subtract=False,
            )
            targets = select_generic_targeted_lines(detected, lib, c)
        candidates = []
        fit_diagnostics = []
        targeted = analyze_raw_spectrum_targeted(
            d,
            targets,
            low_significance_candidates=candidates,
            fit_diagnostics=fit_diagnostics,
            max_assignment_energy_delta_fwhm=1.0,
            peak_threshold=0.0 if wire else c["targeted_peak_significance_sigma"],
            min_energy_keV=c["min_peak_energy_keV"],
            max_energy_keV=c["max_peak_energy_keV"],
            background_spectrum=bg,
            profile_name=c["profile_name"],
            counting_method="iec_tiered",
            roi_width_fwhm=c["flux_wire_roi_width_fwhm"],
            background_width_channels=c["flux_wire_background_width_channels"],
            background_gap_fwhm=c["flux_wire_background_gap_fwhm"],
            comparison_background_model=(
                c["flux_wire_comparison_background_model"]
                if wire
                else c["generic_comparison_background_model"]
            ),
            broad_window_max_raw_gross_ratio=c[
                "generic_broad_window_max_raw_gross_ratio"
            ],
        )
        peaks = (
            targeted
            if wire
            else merge_detected_and_targeted_peaks(detected, targeted, c)
        )
        report = read_processed_txt(qg, profile_name=c["profile_name"])
        refs = qg_reference_peaks(report)
        entry = {
            "sample": raw.stem,
            "raw_file": str(raw.relative_to(root)),
            "qg_file": str(qg.relative_to(root)),
            "refs": refs,
            "peaks": [
                {
                    "energy_keV": p.energy_keV,
                    "isotope": p.isotope,
                    "channel": p.channel,
                    "significance": p.significance,
                    "net": p.net_counts,
                    "sigma": p.net_counts_unc,
                    "line_energy": p.gamma_line.energy_keV if p.gamma_line else None,
                    "assignment_ambiguous": p.assignment_ambiguous,
                }
                for p in peaks
            ],
        }
        entry["candidates"] = [
            {
                "energy_keV": p.energy_keV,
                "isotope": p.isotope,
                "channel": p.channel,
                "significance": p.significance,
                "net": p.net_counts,
                "sigma": p.net_counts_unc,
                "line_energy": p.gamma_line.energy_keV if p.gamma_line else None,
                "assignment_ambiguous": p.assignment_ambiguous,
            }
            for p in candidates
        ]
        rows.append(entry)
        output.write_text(
            json.dumps(
                {
                    "complete": len(rows) == sum(q is not None for _, q, _ in pairs),
                    "stage": "native independent detection before reference overlay",
                    "counting_method": "iec_tiered",
                    "input_sha256": input_hashes,
                    "runtime_source_sha256": source_hashes,
                    "samples": rows,
                    "unmatched_qg": [str(p.relative_to(root)) for p in uq],
                    "unmatched_raw": [str(p.relative_to(root)) for p in ur],
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(raw.stem, len(refs), len(peaks), flush=True)


if __name__ == "__main__":
    main()
