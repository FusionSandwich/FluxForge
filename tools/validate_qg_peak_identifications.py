"""Audit every QG ROI against a frozen native extraction, including missing data."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import fluxforge
from fluxforge.analysis.qg_peak_ids import corrected_reference_ids
from fluxforge.examples.rafm_workflow import (
    default_paths,
    discover_input_files,
    load_rafm_example_metadata,
    match_peak_set,
    pair_input_files,
    qg_reference_peaks,
)
from fluxforge.io.flux_wire import read_processed_txt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_peaks(rows):
    result = []
    for row in rows:
        values = dict(row)
        for target, alias in (("net_counts", "net"), ("net_counts_unc", "sigma")):
            value = values.get(target, values.get(alias))
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Native peak needs finite nonnegative {target}")
            values[target] = float(value)
        result.append(SimpleNamespace(**values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not Path(fluxforge.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("Import must come from the chosen worktree")
    example = root / "examples/RAFM_irradiation"
    paths, metadata = default_paths(example), load_rafm_example_metadata(example)
    files = discover_input_files(paths)
    pairs, unmatched_raw, unmatched_qg = pair_input_files(
        files["raw"], files["qg"], metadata.pairing_aliases
    )
    cache = json.loads(args.native.read_text(encoding="utf-8"))
    if cache.get("complete") is not True:
        raise ValueError("Native extraction is incomplete")
    if cache.get("stage") != "native independent detection before reference overlay":
        raise ValueError("Native extraction must precede all reference overlays")
    if not cache.get("input_sha256"):
        raise ValueError("Native extraction must bind its inputs")
    required_inputs = {
        path.relative_to(root).as_posix()
        for raw, qg, _ in pairs
        if qg
        for path in (raw, qg)
    }
    bound_inputs = {key.replace("\\", "/") for key in cache["input_sha256"]}
    if not required_inputs <= bound_inputs:
        raise ValueError(
            "Native extraction must hash every paired raw spectrum and report"
        )
    for path, digest in cache["input_sha256"].items():
        if sha(root / path) != digest:
            raise ValueError(f"Native input changed: {path}")
    for path, digest in cache.get("candidate_replay_input_sha256", {}).items():
        if sha(root / path) != digest:
            raise ValueError(f"Candidate replay input changed: {path}")
    native = {item["qg_file"]: item for item in cache["samples"]}
    expected = {qg.relative_to(root).as_posix() for _, qg, _ in pairs if qg}
    native = {key.replace("\\", "/"): value for key, value in native.items()}
    if set(native) != expected or len(native) != len(cache["samples"]):
        raise ValueError(
            "Native extraction must cover every paired report exactly once"
        )
    for raw, qg, _ in pairs:
        if qg:
            sample = native[qg.relative_to(root).as_posix()]
            if (
                sample["raw_file"].replace("\\", "/")
                != raw.relative_to(root).as_posix()
            ):
                raise ValueError("Native raw spectrum/report pairing changed")
    correction_file = paths.metadata_root / "qg_peak_id_corrections.json"
    corrections = json.loads(correction_file.read_text(encoding="utf-8"))["corrections"]
    known_reports = {path.relative_to(paths.qg_root).as_posix() for path in files["qg"]}
    if any(item["report"] not in known_reports for item in corrections):
        raise ValueError("Identity correction names an unknown report")
    records, report_records, source_hashes = [], [], {}
    for path in files["qg"]:
        report_key = path.relative_to(paths.qg_root).as_posix()
        source_hashes[path.relative_to(root).as_posix()] = sha(path)
        report = read_processed_txt(path, profile_name=metadata.config["profile_name"])
        positive = qg_reference_peaks(report)
        refs = corrected_reference_ids(report_key, positive, corrections)
        detected = native.get(path.relative_to(root).as_posix())
        peaks = native_peaks(detected["peaks"]) if detected else []
        matches = match_peak_set(refs, peaks, metadata.config)
        used_channels = {peak.channel for peak, _ in matches if peak is not None}
        weak = native_peaks(detected.get("candidates", [])) if detected else []
        weak = [peak for peak in weak if peak.channel not in used_channels]
        missing_indices = [
            index for index, (peak, _) in enumerate(matches) if peak is None
        ]
        tentative = dict(
            zip(
                missing_indices,
                match_peak_set(
                    [refs[index] for index in missing_indices], weak, metadata.config
                ),
            )
        )
        by_line = {}
        for index, (ref, (peak, same_id)) in enumerate(zip(refs, matches)):
            original_id = ref.get("reported_isotope", ref["isotope"])
            candidate, candidate_same_id = tentative.get(index, (None, False))
            threshold = float(
                metadata.config.get("targeted_peak_significance_sigma", 2.0)
            )
            if detected is None:
                status = "missing_raw_spectrum"
            elif peak is None:
                status = "tentative_same_id" if candidate_same_id else "missing_peak"
            elif not same_id:
                status = "different_or_ambiguous_id"
            elif peak.significance < threshold:
                status = "tentative_native_same_id"
            else:
                status = (
                    "corrected_reference_id"
                    if "identity_correction" in ref
                    else "same_id"
                )
            by_line[ref["report_source_line_number"]] = dict(
                status=status,
                expected_isotope=ref["isotope"],
                reported_isotope=original_id,
                native_isotope=peak.isotope if peak else None,
                native_energy_keV=peak.energy_keV if peak else None,
                native_channel=peak.channel if peak else None,
                native_significance=peak.significance if peak else None,
                confirmed_detection=(
                    peak is not None
                    and same_id
                    and peak.significance
                    >= float(
                        metadata.config.get("targeted_peak_significance_sigma", 2.0)
                    )
                ),
                tentative_isotope=candidate.isotope if candidate else None,
                tentative_energy_keV=candidate.energy_keV if candidate else None,
                tentative_channel=candidate.channel if candidate else None,
                tentative_significance=candidate.significance if candidate else None,
                correction=ref.get("identity_correction"),
            )
        for nuclide in report.nuclides:
            for peak in nuclide.peaks:
                line = peak["source_line_number"]
                row = dict(
                    report=report_key,
                    source_sha256=source_hashes[path.relative_to(root).as_posix()],
                    source_line_number=line,
                    reported_isotope=nuclide.isotope,
                    reported_energy_keV=peak["center_keV"],
                    net_counts=peak["net_counts"],
                    net_counts_unc=peak["net_unc"],
                    activity_available=peak["activity_available"],
                )
                row.update(by_line.get(line, {"status": "reference_nondetection"}))
                records.append(row)
        report_records.append(
            dict(
                report=report_key,
                raw_file=detected["raw_file"] if detected else None,
                positive_roi_count=len(refs),
                roi_count=sum(len(n.peaks) for n in report.nuclides),
            )
        )
    counts = dict(Counter(row["status"] for row in records))
    failed = sum(
        counts.get(key, 0)
        for key in (
            "missing_raw_spectrum",
            "missing_peak",
            "different_or_ambiguous_id",
            "tentative_same_id",
            "tentative_native_same_id",
        )
    )
    payload = dict(
        passed=failed == 0,
        scientific_admission=False,
        scope="Native pre-reference identities; positive ROIs checked one-to-one; zero-net ROIs retained as nondetections",
        metrics=dict(
            reports=len(report_records),
            roi_rows=len(records),
            status_counts=counts,
            paired_identity_labels_agree=not any(
                counts.get(key, 0)
                for key in ("missing_peak", "different_or_ambiguous_id")
            ),
            tentative_candidates_are_confirmed_detections=False,
            matched_native_fits=sum(
                row.get("native_isotope") is not None
                and row["status"]
                in ("same_id", "corrected_reference_id", "tentative_native_same_id")
                for row in records
            ),
            confirmed_detections=sum(
                row.get("confirmed_detection", False) for row in records
            ),
            detection_significance_threshold=float(
                metadata.config.get("targeted_peak_significance_sigma", 2.0)
            ),
        ),
        reports=report_records,
        rows=records,
        unmatched_raw=[p.relative_to(root).as_posix() for p in unmatched_raw],
        unmatched_qg=[p.relative_to(root).as_posix() for p in unmatched_qg],
        report_sha256=source_hashes,
        native_extraction_sha256=sha(args.native),
        correction_manifest_sha256=sha(correction_file),
        native_extraction_source_commit=cache.get("source_commit"),
        native_extraction_runtime_source_sha256=cache.get("runtime_source_sha256"),
        runtime_source_sha256={
            p.relative_to(root).as_posix(): sha(p)
            for p in (root / "src/fluxforge").rglob("*.py")
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["metrics"]))
    print("All-reports identity acceptance:", payload["passed"])


if __name__ == "__main__":
    main()
