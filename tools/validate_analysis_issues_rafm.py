"""Bounded RAFM before/after replay for analysis issue repairs.

Run with PYTHONPATH pointing to either the frozen baseline or the repaired src.
All inputs are read-only; the caller must supply a new output path.
QG comparisons are diagnostics, not known physical truth or admission.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import fluxforge
from fluxforge.analysis.flux_wire_analysis import (
    analyze_raw_spectrum_targeted,
    build_gamma_library,
    get_expected_isotopes,
)
from fluxforge.examples.rafm_workflow import (
    build_generic_gamma_library,
    default_paths,
    discover_input_files,
    load_rafm_example_metadata,
    pair_input_files,
    qg_reference_peaks,
    workflow_profile_energy_calibration,
)
from fluxforge.io.flux_wire import read_processed_txt, read_raw_asc
from fluxforge.physics.activation import GammaLineMeasurement
from fluxforge.unfolding import MLSeedUnfolder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.source_root = args.source_root.resolve()
    assert Path(fluxforge.__file__).resolve().is_relative_to(args.source_root.resolve())
    if args.out.exists():
        raise FileExistsError(args.out)
    example = args.root / "examples/RAFM_irradiation"
    paths = default_paths(example)
    metadata = load_rafm_example_metadata(example)
    calibration = workflow_profile_energy_calibration(metadata.config)
    background = read_raw_asc(
        paths.background_path,
        energy_calibration_override=calibration,
        profile_name=metadata.config["profile_name"],
    ).spectrum
    library, _ = build_generic_gamma_library(metadata)
    requested = [
        ("Cr51", 320.0835),
        ("Mn54", 834.838),
        ("Co60", 1173.228),
        ("Co60", 1332.492),
        ("W187", 685.74),
    ]
    generic_lines = [
        min(
            (line for line in library if line.isotope == isotope),
            key=lambda line: abs(line.energy_keV - energy),
        )
        for isotope, energy in requested
    ]
    off_energy = [
        ("RAFM4-C_15dEOI", "W187", 206.25),
        ("RAFM4-N_15dEOI", "Tb154m", 247.94),
        ("RAFM4-A_15dEOI", "V52", 1434.09),
        ("RAFM4-A_15dEOI", "Mn56", 2113.09),
    ]
    inputs = {
        str(paths.background_path.relative_to(args.root)): sha(paths.background_path)
    }
    for path in sorted(paths.metadata_root.glob("*.json")):
        inputs[str(path.relative_to(args.root))] = sha(path)
    files = discover_input_files(paths)
    pairs, _, _ = pair_input_files(files["raw"], files["qg"], metadata.pairing_aliases)
    samples, clock_rows, rejected_cases = [], [], []
    for raw_path, qg_path, key in pairs:
        inputs[str(raw_path.relative_to(args.root))] = sha(raw_path)
        data = read_raw_asc(
            raw_path,
            energy_calibration_override=calibration,
            profile_name=metadata.config["profile_name"],
        )
        reference = None
        if qg_path:
            inputs[str(qg_path.relative_to(args.root))] = sha(qg_path)
            reference = read_processed_txt(
                qg_path, profile_name=metadata.config["profile_name"]
            )
        qg = qg_reference_peaks(reference) if reference else []
        wire = raw_path.parent.name == "flux_wires"
        lines = (
            build_gamma_library(isotope_filter=get_expected_isotopes(data.sample_id))
            if wire
            else generic_lines
        )
        peaks = analyze_raw_spectrum_targeted(
            data,
            lines,
            peak_threshold=2.0,
            min_energy_keV=80.0,
            background_spectrum=background,
            profile_name=metadata.config["profile_name"],
            counting_method="qg",
        )
        records = []
        for peak in peaks:
            candidates = [
                row
                for row in qg
                if row["isotope"] == peak.isotope
                and abs(row["energy_keV"] - peak.energy_keV) <= 2.0
            ]
            match = (
                min(
                    candidates, key=lambda row: abs(row["energy_keV"] - peak.energy_keV)
                )
                if candidates
                else None
            )
            records.append(
                {
                    "isotope": peak.isotope,
                    "expected_energy": peak.gamma_line.energy_keV,
                    "fitted_energy": peak.energy_keV,
                    "physical_net_counts": peak.net_counts,
                    "physical_net_sigma": peak.net_counts_unc,
                    "raw_comparison_net_counts": peak.comparison_net_counts,
                    "raw_comparison_net_sigma": peak.comparison_net_counts_unc,
                    "significance": peak.significance,
                    "qg_net_counts": match["net_counts"] if match else None,
                    "qg_net_sigma": match["net_unc"] if match else None,
                    "relative_qg_count_difference": (
                        peak.net_counts / match["net_counts"] - 1 if match else None
                    ),
                }
            )
        samples.append(
            {
                "sample": raw_path.stem,
                "wire": wire,
                "requested_lines": len(lines),
                "found": records,
            }
        )
        if wire and reference:
            for row in qg:
                nuclide = next(
                    (n for n in reference.nuclides if n.isotope == row["isotope"]), None
                )
                probability = row["rad_int_fraction"]
                efficiency = float(reference.efficiency.efficiency(row["energy_keV"]))
                if nuclide is None or probability <= 0 or efficiency <= 0:
                    continue
                line = GammaLineMeasurement(
                    row["net_counts"],
                    reference.live_time,
                    efficiency,
                    probability,
                    nuclide.half_life_seconds,
                    dead_time_fraction=1 - reference.live_time / reference.real_time,
                )
                clock_rows.append(
                    {
                        "sample": raw_path.stem,
                        "isotope": nuclide.isotope,
                        "energy": row["energy_keV"],
                        "live_time_s": reference.live_time,
                        "real_time_s": reference.real_time,
                        "shared_helper_count_start_activity_bq": line.activity_at_reference(),
                        "qg_line_activity_bq": row["line_activity_bq"],
                    }
                )
        for name, isotope, energy in off_energy:
            if raw_path.stem != name:
                continue
            target = min(
                (line for line in library if line.isotope == isotope),
                key=lambda line: abs(line.energy_keV - energy),
            )
            outputs = analyze_raw_spectrum_targeted(
                data,
                [target],
                peak_threshold=2.0,
                min_energy_keV=80.0,
                background_spectrum=background,
                profile_name=metadata.config["profile_name"],
                counting_method="iec_tiered",
            )
            rejected_cases.append(
                {
                    "sample": name,
                    "isotope": isotope,
                    "expected_energy": target.energy_keV,
                    "found": [
                        {"energy": peak.energy_keV, "net": peak.net_counts}
                        for peak in outputs
                    ],
                }
            )
        print(f"{raw_path.stem}: {len(peaks)}/{len(lines)} selected lines", flush=True)
    receipt_path = (
        args.root / "artifacts/validation/registry_uncertainty_20261001/receipt.json"
    )
    inputs[str(receipt_path.relative_to(args.root))] = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    response = np.asarray(receipt["row_cross_sections_barn"]) * 1e-24 * 1e11
    measured = np.asarray(receipt["replay_rates"])
    sigma = np.asarray(receipt["replay_rate_uncertainties"])
    unfolding = []
    for prior in (None, np.ones(response.shape[1])):
        for warm in (0, 8):
            cases = []
            for units in (np.ones(len(measured)), 1 / sigma):
                result = MLSeedUnfolder().unfold(
                    measured * units,
                    response * units[:, None],
                    initial_flux=prior,
                    measurement_uncertainty=sigma * units,
                    warm_start_iterations=warm,
                )
                residual = (response @ result.flux - measured) / sigma
                cases.append(
                    {
                        "flux": result.flux.tolist(),
                        "weighted_residual_sum_squares": float(residual @ residual),
                        "accepted": result.converged,
                        "confidence": result.parameters_used["confidence_score"],
                    }
                )
            unfolding.append(
                {
                    "prior": "ones" if prior is not None else "automatic",
                    "warm": warm,
                    "raw": cases[0],
                    "row_whitened": cases[1],
                    "max_absolute_unit_difference": float(
                        np.max(np.abs(np.asarray(cases[0]["flux"]) - cases[1]["flux"]))
                    ),
                }
            )
    sources = {
        str(p.relative_to(args.source_root)): sha(p)
        for p in sorted((args.source_root / "fluxforge").rglob("*.py"))
    }
    payload = {
        "scientific_admission": False,
        "baseline_commit": "1148e18",
        "scope": "All bundled raw spectra, selected declared line families; not a full gamma-library or transport validation",
        "input_sha256": inputs,
        "runtime_source_sha256": sources,
        "samples": samples,
        "clock_activity_rows": clock_rows,
        "off_energy_cases": rejected_cases,
        "unfolding_unit_cases": unfolding,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
