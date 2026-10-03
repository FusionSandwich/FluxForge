"""Exercise measured RAFM data without turning demonstrations into qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fluxforge.analysis.spectrum_math import add_spectra, subtract_measured_background
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.reader_factory import read_spectrum_any
from fluxforge.io.spe import GammaSpectrum
from fluxforge.io.spectrum_export import SpectrumExporter, SpectrumMetadata
from fluxforge.standards.e1297 import currie_mda
from fluxforge.examples import rafm_workflow as workflow


def digest(path: Path, *, normalize_newlines: bool = False) -> str:
    data = path.read_bytes()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_assets(root: Path) -> dict:
    """Reject missing, altered, duplicated or escaping manifest assets."""
    root = root.resolve()
    manifest = json.loads((root / "metadata/archive_manifest.json").read_text())
    seen = set()
    for asset in manifest["assets"]:
        path = (root / asset["path"]).resolve()
        if not path.is_relative_to(root) or path in seen:
            raise ValueError(f"Invalid or duplicate asset path: {asset['path']}")
        seen.add(path)
        if (
            not path.is_file()
            or digest(
                path, normalize_newlines=asset.get("hash_basis") == "lf_normalized"
            )
            != asset["sha256"]
        ):
            raise ValueError(f"Missing or changed asset: {asset['path']}")
    for name, expected in manifest.get("runtime_data_sha256", {}).items():
        if name != "rafm_profiles.json":
            raise ValueError(f"Unknown runtime data binding: {name}")
        verify_runtime_input(Path(__file__).parents[1] / "data" / name, expected)
    raw = {p.resolve() for p in (root / "raw_gamma_spec").rglob("*.ASC")}
    recorded = {
        (root / a["path"]).resolve()
        for a in manifest["assets"]
        if a["role"] == "canonical_raw"
    }
    if raw != recorded or len(raw) != manifest["canonical_raw_count"]:
        raise ValueError("Raw spectrum inventory differs from manifest")
    if "converted_sample_count" in manifest:
        converted_root = root / "converted_gamma_spec"
        files = {p.resolve() for p in converted_root.rglob("*") if p.is_file()}
        recorded = {
            (root / a["path"]).resolve()
            for a in manifest["assets"]
            if a["role"] in {"converted_counts_only", "unsupported_binary_spc"}
        }
        if files != recorded:
            raise ValueError("Converted spectrum inventory differs from manifest")
        partners = []
        for extension in ("CHN", "SPE", "SPC"):
            group = {
                p.relative_to(converted_root / extension).with_suffix(".ASC")
                for p in (converted_root / extension).rglob("*." + extension)
            }
            if len(group) != manifest["converted_sample_count"]:
                raise ValueError(f"Incomplete converted {extension} inventory")
            partners.append(group)
        if not (partners[0] == partners[1] == partners[2]):
            raise ValueError("Converted format partners differ")
    return manifest


def verify_runtime_input(path: Path, expected: str) -> None:
    if not path.is_file() or digest(path, normalize_newlines=True) != expected:
        raise ValueError(f"Missing or changed runtime input: {path.name}")


def compare_conversion(canonical: GammaSpectrum, converted: GammaSpectrum) -> dict:
    """Exact bin identity is required; matching total counts is insufficient."""
    counts_equal = np.array_equal(canonical.counts, converted.counts)
    channels_equal = np.array_equal(canonical.channels, converted.channels)
    if not counts_equal or not channels_equal:
        raise ValueError("Converted spectrum changed counts or channel positions")
    times_equal = (canonical.live_time, canonical.real_time, canonical.start_time) == (
        converted.live_time,
        converted.real_time,
        converted.start_time,
    )
    a = canonical.calibration.get("energy", [])
    b = converted.calibration.get("energy", [])
    energy_equal = len(a) == len(b) and bool(np.allclose(a, b, rtol=1e-7, atol=1e-9))
    return {
        "counts_equal": True,
        "channels_equal": True,
        "acquisition_times_equal": times_equal,
        "energy_calibration_equal": energy_equal,
        "converted_live_time_s": converted.live_time,
        "converted_real_time_s": converted.real_time,
        "converted_start_time": str(converted.start_time),
        "converted_energy_coefficient_count": len(b),
        "state": (
            "metadata_preserved" if times_equal and energy_equal else "counts_only"
        ),
        "absolute_activity_qualified": False,
    }


def run_example(root: Path, output: Path) -> dict:
    manifest = verify_assets(root)
    if manifest.get("converted_sample_count") != 29:
        raise ValueError("This example requires all 29 converted acquisition partners")
    if set(manifest.get("runtime_data_sha256", {})) != {"rafm_profiles.json"}:
        raise ValueError("Example requires a bound detector profile")
    bound = {a["path"] for a in manifest["assets"]}
    required = {"background.ASC"} | {
        f"metadata/{name}.json"
        for name in (
            "workflow_config",
            "sample_schedule",
            "sample_schedules",
            "flux_wire_metadata",
            "pairing_aliases",
            "sample_gamma_library",
        )
    }
    if not required.issubset(bound):
        raise ValueError("Example requires bound background and workflow metadata")
    output.mkdir(parents=True, exist_ok=True)
    background = read_raw_asc(root / "background.ASC").spectrum
    schedule = workflow.load_rafm_example_metadata(root)
    rows = []
    for path in sorted((root / "raw_gamma_spec").rglob("*.ASC")):
        spectrum = read_raw_asc(path).spectrum
        if not (
            np.all(np.isfinite(spectrum.counts))
            and spectrum.live_time > 0
            and spectrum.real_time >= spectrum.live_time
            and spectrum.start_time
        ):
            raise ValueError(f"Invalid acquisition: {path.name}")
        corrected = subtract_measured_background(
            spectrum, background, negative_policy="hybrid"
        )
        # Sparse covariance must survive project-style JSON round trips.
        payload = corrected.to_dict()
        restored = GammaSpectrum.from_dict(json.loads(json.dumps(payload)))
        if not np.array_equal(restored.counts, corrected.counts):
            raise ValueError("Signed count round trip failed")
        if not np.array_equal(
            restored.counts_uncertainty, corrected.counts_uncertainty
        ):
            raise ValueError("Uncertainty round trip failed")
        if corrected.counts_covariance is not None:
            if (
                restored.counts_covariance is None
                or (restored.counts_covariance != corrected.counts_covariance).nnz
            ):
                raise ValueError("Covariance round trip failed")
        timing = workflow.resolve_measurement_timing(
            path.stem, spectrum.start_time, schedule
        )
        rows.append(
            {
                "sample": path.stem,
                "source_sha256": digest(path),
                "channels": len(spectrum.counts),
                "total_raw_counts": float(spectrum.counts.sum()),
                "start_time": spectrum.start_time.isoformat(),
                "live_time_s": spectrum.live_time,
                "real_time_s": spectrum.real_time,
                "embedded_energy_coefficients": spectrum.calibration.get("energy"),
                "negative_corrected_channels": int(np.sum(corrected.counts < 0)),
                "covariance_roundtrip": True,
                "schedule_group": timing.sample_group,
                "eoi_available": timing.compare_eoi,
            }
        )
    conversions = []
    for path in sorted((root / "converted_gamma_spec").rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root / "converted_gamma_spec")
        raw = root / "raw_gamma_spec" / Path(*rel.parts[1:]).with_suffix(".ASC")
        if not raw.is_file():
            raise ValueError(f"Missing canonical partner for {rel}")
        if path.suffix.upper() == ".SPC":
            try:
                read_spectrum_any(path)
            except ValueError:
                result = {
                    "state": "unsupported_binary_spc",
                    "absolute_activity_qualified": False,
                }
            else:
                raise ValueError("Binary SPC support changed; review this example gate")
        else:
            result = compare_conversion(
                read_raw_asc(raw).spectrum, read_spectrum_any(path)
            )
        conversions.append(
            {
                "path": rel.as_posix(),
                "canonical": raw.relative_to(root).as_posix(),
                "sha256": digest(path),
                **result,
            }
        )
    # Conditional sensitivity calculation uses the recorded profile, not an invented calibration.
    co = read_raw_asc(
        root / "raw_gamma_spec/flux_wires/Co-RAFM-1_25cm.ASC", profile_name="rafm_25cm"
    )
    sample = co.spectrum
    adjusted = subtract_measured_background(sample, background)
    energies = sample.channel_to_energy(sample.channels)
    roi = (energies >= 1168) & (energies <= 1178)
    weights = roi.astype(float)
    variance = (
        float(weights @ adjusted.counts_covariance @ weights)
        if adjusted.counts_covariance is not None
        else float(np.sum(adjusted.counts_uncertainty[roi] ** 2))
    )
    efficiency = float(co.efficiency.efficiency(1173.2))
    # Use the energy-aligned, scaled contribution from the actual subtraction.
    scaled_background = float(np.sum((sample.counts - adjusted.counts)[roi]))
    mda = currie_mda(
        background_counts=scaled_background,
        live_time_s=sample.live_time,
        efficiency=efficiency,
        emission_probability=0.9985,
    )
    # Distinct Co and Co-Cd acquisitions share the same embedded energy grid.
    co_cd = read_raw_asc(
        root / "raw_gamma_spec/flux_wires/Co-Cd-RAFM-1_25cm.ASC"
    ).spectrum
    summed = add_spectra(sample, co_cd)
    np.testing.assert_array_equal(summed.counts, sample.counts + co_cd.counts)
    np.testing.assert_allclose(
        summed.counts_uncertainty**2,
        sample.counts_uncertainty**2 + co_cd.counts_uncertainty**2,
    )
    exporter = SpectrumExporter(
        sample.counts.tolist(),
        SpectrumMetadata(
            sample_id=sample.spectrum_id,
            live_time=sample.live_time,
            real_time=sample.real_time,
            start_time=sample.start_time,
            energy_coefficients=sample.calibration["energy"],
        ),
        energies=energies.tolist(),
        uncertainties=sample.counts_uncertainty.tolist(),
    )
    csv = output / "co_raw.csv"
    exporter.to_csv(csv)
    restored_csv = read_spectrum_any(csv)
    np.testing.assert_array_equal(restored_csv.counts, sample.counts)
    np.testing.assert_allclose(restored_csv.energies, energies, atol=1e-3, rtol=1e-5)
    np.testing.assert_allclose(
        restored_csv.counts_uncertainty,
        sample.counts_uncertainty,
        atol=0.005001,
        rtol=0,
    )
    result = {
        "state": "EXAMPLE_EXECUTED_REVIEW_LIMITS",
        "accuracy_qualified": False,
        "archive_manifest_sha256": digest(root / "metadata/archive_manifest.json"),
        "runtime_data_sha256": manifest["runtime_data_sha256"],
        "runner_sha256": digest(Path(__file__)),
        "raw_acquisitions": rows,
        "conversions": conversions,
        "sum_distinct_raw_acquisitions": "passed",
        "csv_count_energy_roundtrip": "passed",
        "csv_uncertainty_roundtrip": "passed_at_export_precision_0.01_counts",
        "co1173_roi": {
            "bounds_keV": [1168, 1178],
            "signed_net_counts": float(adjusted.counts[roi].sum()),
            "count_uncertainty": float(np.sqrt(variance)),
            "conditional_currie_mda_bq": mda,
            "efficiency": efficiency,
            "emission_probability": 0.9985,
            "qualification": "Sensitivity demonstration: profile efficiency uncertainty and independent certificate unavailable.",
        },
        "limitations": manifest["source_notes"],
    }
    (output / "feature_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example-root", type=Path, default=Path("examples/RAFM_irradiation")
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_example(args.example_root.resolve(), args.output_root.resolve())
    print(
        f"Checked {len(result['raw_acquisitions'])} raw acquisitions and {len(result['conversions'])} conversions; accuracy remains unqualified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
