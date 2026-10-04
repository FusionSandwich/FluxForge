"""Recover five study acquisitions from immutable, source-bound native ANS bytes.

The checked layout is limited to the rev4 RAFM archive, not a general ANS reader.
No reference peak areas, activities or identities enter the generated spectrum.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


SOURCE_COMMIT = "46096eb1d8f760645b4c498b2a7bb50c9b63f262"
SOURCE_ROOT = "examples/RAFM_irradiation/quantumgold_reference"
RECOVER = {
    "Cu-Cd-RAFM-1": "flux_wires/Cu-Cd-RAFM-1_25cm",
    "Fe-Cd-RAFM-1": "flux_wires/Fe-Cd-RAFM-1_0cm",
    "RAFM-A-24hr": "RAFM3/RAFM3-A_24hrEOI",
    "RAFM-A-300s": "RAFM3/RAFM3-A_300sEOI",
    "RAFM-A-4d": "RAFM3/RAFM3-A_4dEOI",
}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def report_equivalence(original, canonical):
    if original == canonical:
        return "identical_bytes"
    # One legacy canonical report replaced all five cp1252 plus/minus signs
    # with UTF-8 replacement characters. Permit that exact conversion only.
    try:
        if canonical.decode("utf-8") == original.decode("cp1252").replace(
            "\u00b1", "\ufffd"
        ):
            return "only_plus_minus_replaced_by_utf8_replacement_character"
    except UnicodeDecodeError:
        pass
    raise ValueError("Canonical report differs from original")


def decode_study_ans(blob, measurement):
    """Validate independent native channels and timing against archival anchors."""
    if len(blob) < 1548:
        raise ValueError("Truncated ANS header")
    revision = struct.unpack_from("<h", blob, 0)[0]
    first, last = struct.unpack_from("<hh", blob, 1020)
    nroi = struct.unpack_from("<h", blob, 1026)[0]
    if revision != 4 or (first, last) != (0, 8191) or nroi < 0:
        raise ValueError("Unsupported study ANS layout")
    if len(blob) != 1548 + 8192 * 4 + nroi * 50:
        raise ValueError("ANS header/channel/ROI length does not close")
    array_blob = blob[1548 : 1548 + 8192 * 4]
    if sha(array_blob) != measurement["channel_array_sha256"]:
        raise ValueError("Native channel array hash mismatch")
    coefficients = struct.unpack_from("<3f", blob, 424)
    if list(coefficients) != measurement["native_energy_polynomial_keV"]:
        raise ValueError("Native calibration mismatch")
    start_days = struct.unpack_from("<d", blob, 80)[0]
    real, live = struct.unpack_from("<dd", blob, 96)
    if not all(math.isfinite(v) for v in (start_days, real, live, *coefficients)):
        raise ValueError("Nonfinite native header")
    if not 0 < live <= real:
        raise ValueError("Invalid native acquisition duration")
    start = datetime(1899, 12, 30) + timedelta(days=start_days)
    report = measurement["QG_header"]
    if (
        abs(
            (
                start - datetime.fromisoformat(report["clock_local_unzoned"])
            ).total_seconds()
        )
        > 0.01
    ):
        raise ValueError("Native/report acquisition clock mismatch")
    if abs(live - report["live_s"]) > 0.005 or abs(real - report["real_s"]) > 0.005:
        raise ValueError("Native/report duration mismatch")
    return dict(
        counts=struct.unpack("<8192I", array_blob),
        coefficients=coefficients,
        start=start,
        live=live,
        real=real,
        spectrum_id=blob[2:74].decode("cp1252").strip(" \0"),
    )


def render_asc(decoded):
    """Lossless integer channel export; native unzoned clock is kept verbatim."""
    a, b, c = decoded["coefficients"]
    header = (
        f"ID: {decoded['spectrum_id']}\n\n"
        f"Acquisition Date: {decoded['start']:%d-%b-%Y %H:%M:%S}\n"
        f"Elapsed Real Time: {decoded['real']:.8f}\n"
        f"Elapsed Live Time: {decoded['live']:.8f}\n"
        f"Conversion Gain: 8192                 Calibration\n"
        f"High Voltage: 0               A = {a:.12E}\n"
        f"Coarse Gain: 0               B = {b:.12E}\n"
        f"Fine Gain: 1.00               C = {c:.12E}\n\n"
        "Channel     Contents\n"
    )
    return (
        header + "".join(f"{i:4d} {n:12d}\n" for i, n in enumerate(decoded["counts"]))
    ).encode("ascii")


def recover(root):
    root = Path(root).resolve()
    example = root / "examples/RAFM_irradiation"
    source_out = example / "recovered_qg_sources"
    if source_out.exists():
        raise FileExistsError(source_out)

    def git_blob(path):
        return subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=root
        )

    manifest_blob = git_blob(f"{SOURCE_ROOT}/manifest.json")
    manifest = json.loads(manifest_blob)
    resources = {r["path"]: r for r in manifest["resources"]}
    writes, receipts, selected = [], [], []
    for measurement in manifest["measurements"]:
        mid = measurement["measurement_id"]
        if mid not in RECOVER:
            continue
        stem = RECOVER[mid]
        local_report = example / "QG_processed_gamma_data" / (stem + ".txt")
        ans_path, report_path = (measurement["files"][k] for k in ("ANS", "QG_report"))
        ans_blob, report_blob = git_blob(ans_path), git_blob(report_path)
        for path, blob in ((ans_path, ans_blob), (report_path, report_blob)):
            if (
                sha(blob) != resources[path]["sha256"]
                or len(blob) != resources[path]["bytes"]
            ):
                raise ValueError("Archive resource mismatch: " + path)
        canonical_blob = local_report.read_bytes()
        report_match = report_equivalence(report_blob, canonical_blob)
        decoded = decode_study_ans(ans_blob, measurement)
        output = example / "raw_gamma_spec" / (stem + ".ASC")
        if output.exists():
            raise FileExistsError(output)
        asc_blob = render_asc(decoded)
        writes.extend(
            (
                (source_out / (mid + ".ANS"), ans_blob),
                (source_out / (mid + ".txt"), report_blob),
                (output, asc_blob),
            )
        )
        selected.append(measurement)
        receipts.append(
            dict(
                measurement_id=mid,
                raw_file=output.relative_to(root).as_posix(),
                original_git_path=ans_path,
                original_ans_sha256=sha(ans_blob),
                original_report_sha256=sha(report_blob),
                exported_asc_sha256=sha(asc_blob),
                canonical_report_sha256=sha(canonical_blob),
                report_equivalence=report_match,
                channel_array_sha256=measurement["channel_array_sha256"],
                native_start_local_unzoned=decoded["start"].isoformat(),
                live_time_s=decoded["live"],
                real_time_s=decoded["real"],
                separate_original_asc_header=measurement["ASC_header"],
                qualification="Native clock corroborates report; source ASC clocks differ; no timezone inferred. Counts are observed native channels, not report ROI counts.",
            )
        )
    if len(receipts) != len(RECOVER):
        raise ValueError("Archive lacks requested acquisitions")
    for path, blob in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(blob)
    receipt = dict(
        source_commit=SOURCE_COMMIT,
        source_manifest_sha256=sha(manifest_blob),
        layout_scope="Empirically channel-verified RAFM rev4 archive only",
        scientific_admission=False,
        recovered=receipts,
    )
    (source_out / "recovery_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    (source_out / "source_manifest.json").write_bytes(manifest_blob)
    (source_out / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.root), indent=2))


if __name__ == "__main__":
    main()
