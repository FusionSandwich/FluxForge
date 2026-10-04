"""Source-bound South-background Co-Cd joint-Poisson model-check pilot (#249).

This is an opt-in diagnostic. It never loads QuantumGold target activities and it
never changes RAFM workflow defaults. The recovered South background remains a
later, temporally unqualified measurement. Numerical convergence is kept
separate from model adequacy and physical qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import csv
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import scipy

from fluxforge.analysis.flux_wire_analysis import estimate_peak_area_local_background
from fluxforge.analysis.joint_poisson import (
    BackgroundChoice,
    CountObservation,
    PeakResponse,
    fit_joint_poisson,
)
from fluxforge.analysis.peakfit import FWHM_SIG_RATIO
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.data.rafm_profile import load_rafm_profile
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.io.spe import GammaSpectrum

BASE_HEAD = "701288a30f418ed7dc331e6deda0044eb220e72e"
SAMPLE_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/originals/ASC/Co-Cd-RAFM-1.ASC"
)
SOUTH_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/"
    "South 4hr Background Terminal.ANS"
)
MANIFEST_REL = Path("examples/RAFM_irradiation/quantumgold_reference/manifest.json")
SUPPLEMENT_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/manifest.json"
)
CONFIG_REL = Path(
    "examples/RAFM_irradiation/quantumgold_reference/runtime/metadata/workflow_config.json"
)
PROFILE_REL = Path("src/fluxforge/data/rafm_profiles.json")
IEC_RECEIPT_REL = Path(
    "artifacts/validation/quantumgold_integration_20261004/south_native/"
    "CURRENT_LINE_COMBINATIONS.json"
)
SOUTH_SHA256 = "96f2e47eb2edc68db227157aa08c601be6cd0ec4e46f1abfa114d46e2d509344"
SAMPLE_SHA256 = "7482016df6c9d370b68e5d5646cd64c5fa74def9fe586e58d19d89dab0c87cba"
CO60_LINES = ((1173.228, 0.9985), (1332.492, 0.999826))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def midpoint_edges(energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    return np.r_[
        energies[0] - (energies[1] - energies[0]) / 2,
        (energies[:-1] + energies[1:]) / 2,
        energies[-1] + (energies[-1] - energies[-2]) / 2,
    ]


def native_observation(data, path: Path, edges, lo: int, hi: int) -> CountObservation:
    return CountObservation(
        data.spectrum.counts[lo : hi + 1],
        edges[lo : hi + 2],
        data.live_time,
        sha(path),
        data.start_time.isoformat() if data.start_time else None,
    )


def south_native_background() -> tuple[GammaSpectrum, dict]:
    supplement_path = ROOT / SUPPLEMENT_REL
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    pins = [
        item
        for item in supplement["resources"]
        if item["role"] == "recovered_South_native_background_not_ASC"
    ]
    if len(pins) != 1:
        raise ValueError("South source identity is ambiguous")
    pin = pins[0]
    if Path(pin["path"]) != SOUTH_REL or pin["sha256"] != SOUTH_SHA256:
        raise ValueError("South source manifest binding changed")
    path = ROOT / SOUTH_REL
    blob = path.read_bytes()
    if len(blob) != pin["bytes"] or sha(path) != pin["sha256"]:
        raise ValueError("South source identity mismatch")
    if len(blob) != 36616 or b"South 4 hr background terminal" not in blob[:1548]:
        raise ValueError("Unsupported South background layout")

    coefficients = struct.unpack_from("<3f", blob, 424)
    real_time = struct.unpack_from("<d", blob, 96)[0]
    live_time = struct.unpack_from("<d", blob, 104)[0]
    serial = struct.unpack_from("<d", blob, 80)[0]
    counts = np.asarray(struct.unpack_from("<8192I", blob, 1548), dtype=float)
    channels = np.arange(8192)
    energies = (
        coefficients[0]
        + coefficients[1] * channels
        + coefficients[2] * channels**2
    )
    if live_time != 14400 or not live_time <= real_time < 14500:
        raise ValueError("Unsupported South background timing")
    if int(counts.sum()) != 543427 or not np.all(np.diff(energies) > 0):
        raise ValueError("Unsupported South count/calibration anchors")
    start = datetime(1899, 12, 30) + timedelta(days=serial)
    details = {
        "detector": "South",
        "source_path": SOUTH_REL.as_posix(),
        "source_sha256": sha(path),
        "start_time_local_unzoned": start.isoformat(),
        "live_time_s": live_time,
        "real_time_s": real_time,
        "energy_polynomial_keV": list(coefficients),
        "channel_count": 8192,
        "count_sum": int(counts.sum()),
        "payload_offset": 1548,
        "temporal_applicability": "UNRESOLVED",
        "calibration_covariance": "UNAVAILABLE",
        "layout_status": "empirical_source-bound_layout_not_vendor_verified",
    }
    return (
        GammaSpectrum(
            counts=counts,
            channels=channels,
            energies=energies,
            live_time=live_time,
            real_time=real_time,
            start_time=start,
            spectrum_id="South measured source-bound background",
            detector_id="South",
            calibration={"energy": list(coefficients)},
            metadata=details,
        ),
        details,
    )


def _deviance_residuals(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    term = np.empty_like(observed)
    positive = observed > 0
    term[positive] = (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - (observed[positive] - expected[positive])
    )
    term[~positive] = expected[~positive]
    return np.sign(observed - expected) * np.sqrt(np.maximum(2 * term, 0))


def _residual_summary(observed, expected, edges) -> dict:
    dev = _deviance_residuals(observed, expected)
    pearson = (np.asarray(observed) - np.asarray(expected)) / np.sqrt(
        np.maximum(expected, 1e-12)
    )
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    order = np.argsort(np.abs(dev))[::-1][:5]
    return {
        "max_abs_deviance_residual": float(np.max(np.abs(dev))),
        "n_abs_deviance_residual_gt3": int(np.count_nonzero(np.abs(dev) > 3)),
        "n_abs_deviance_residual_gt5": int(np.count_nonzero(np.abs(dev) > 5)),
        "deviance_residuals": dev,
        "pearson_residuals": pearson,
        "largest_bins": [
            {
                "energy_keV": float(centers[i]),
                "observed": float(observed[i]),
                "expected": float(expected[i]),
                "deviance_residual": float(dev[i]),
            }
            for i in order
        ],
    }


def _joint_payload(fit, observation, background_observation) -> dict:
    return {
        "success": fit.success,
        "status": fit.status,
        "message": fit.message,
        "area_full_response_counts": fit.area,
        "normalization": fit.normalization,
        "exposure_scale": fit.exposure_scale,
        "profile": asdict(fit.interval) if fit.interval else None,
        "deviance": fit.deviance,
        "identifiability_ratio": fit.identifiability_ratio,
        "model_diagnostics": fit.model_diagnostics,
        "parameter_names": list(fit.parameter_names),
        "parameters": fit.parameters,
        "sample_expected": fit.sample_expected,
        "background_expected": fit.background_expected,
        "sample_residuals": fit.sample_residuals,
        "background_residuals": fit.background_residuals,
        "sample_residual_diagnostics": _residual_summary(
            observation.counts, fit.sample_expected, observation.energy_edges_keV
        ),
        "background_residual_diagnostics": (
            _residual_summary(
                background_observation.counts,
                fit.background_expected,
                background_observation.energy_edges_keV,
            )
            if background_observation is not None
            else None
        ),
        "provenance": fit.provenance,
    }


def _current_iec_control() -> dict:
    path = ROOT / IEC_RECEIPT_REL
    rows = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row
        for row in rows
        if row.get("sample_id") == "Co-Cd-RAFM-1_25cm"
        and row.get("isotope") == "Co60"
    ]
    if len(matches) != 1:
        raise ValueError("Current South IEC Co-Cd control is missing or ambiguous")
    methods = matches[0]["methods"]
    inverse = [row for row in methods if row["method"] == "inverse_variance"]
    if len(inverse) != 1:
        raise ValueError("Current South IEC inverse-variance control is ambiguous")
    row = inverse[0]
    return {
        "source_path": IEC_RECEIPT_REL.as_posix(),
        "source_sha256": sha(path),
        "method": row["method"],
        "analysis_role": row["analysis_role"],
        "engine_identity": row["engine_identity"],
        "activity_bq": row["activity_bq"],
        "sigma_bq": row["sigma_bq"],
        "input_lines": row["input_lines"],
        "unavailable_components": row["unavailable_components"],
        "status": row["status"],
        "role_in_this_pilot": "external current-engine control; never an inference target",
    }


def _synthetic_challenges() -> list[dict]:
    edges = np.linspace(-6, 6, 49)
    peak = PeakResponse(0.0, 0.8)
    p = peak.integrated(edges)
    widths = np.diff(edges)
    vendor = BackgroundChoice(
        "no_separate_ambient_vendor",
        "deterministic synthetic challenge with no separate ambient",
        continuum="none",
    )
    rows = []
    well_counts = np.rint(1500 * p + 40 * widths).astype(int)
    well = fit_joint_poisson(
        CountObservation(well_counts, edges, 1.0, "synthetic-well"),
        None,
        peak,
        vendor,
        sample_continuum="constant",
        confidence=None,
    )
    rows.append(
        {
            "case": "well_specified_rounded_asimov",
            "success": well.success,
            "status": well.status,
            "area": well.area,
            "adequacy_flag": well.model_diagnostics["adequacy_flag"],
            "deviance": well.deviance,
            "purpose": "positive control; absence of a strong-lack flag is not physical qualification",
        }
    )

    shoulder = PeakResponse(1.2, 0.8).integrated(edges)
    bad_counts = np.rint(1200 * p + 300 * shoulder + 40 * widths).astype(int)
    bad = fit_joint_poisson(
        CountObservation(bad_counts, edges, 1.0, "synthetic-shoulder"),
        None,
        peak,
        vendor,
        sample_continuum="constant",
        confidence=None,
    )
    rows.append(
        {
            "case": "unmodelled_shifted_shoulder",
            "success": bad.success,
            "status": bad.status,
            "area": bad.area,
            "adequacy_flag": bad.model_diagnostics["adequacy_flag"],
            "deviance": bad.deviance,
            "purpose": "negative control; a converged fit must retain model-inadequacy evidence",
        }
    )

    b = 40 * widths + 800 * p
    sample_counts = np.rint(1000 * p + b + 10 * widths).astype(int)
    bg_counts = np.rint(4 * b).astype(int)
    trade = fit_joint_poisson(
        CountObservation(sample_counts, edges, 10, "synthetic-trade-sample"),
        CountObservation(bg_counts, edges, 40, "synthetic-trade-background"),
        peak,
        BackgroundChoice(
            "later_conditional",
            "synthetic free-normalization tradeoff",
            normalization="free",
            continuum="constant",
            peaks=(peak,),
        ),
        sample_continuum="constant",
        confidence=None,
    )
    rows.append(
        {
            "case": "free_normalization_tradeoff",
            "success": trade.success,
            "status": trade.status,
            "identifiability_ratio": trade.identifiability_ratio,
            "purpose": "negative control; rank-deficient scale/peak/continuum tradeoff must not be accepted",
        }
    )
    return rows


def run_pilot() -> dict:
    sample_path = ROOT / SAMPLE_REL
    main_manifest_path = ROOT / MANIFEST_REL
    supplement_path = ROOT / SUPPLEMENT_REL
    config_path = ROOT / CONFIG_REL
    if sha(sample_path) != SAMPLE_SHA256:
        raise ValueError("Source-bound Co-Cd ASC identity changed")
    main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
    source_rows = [
        item
        for item in main_manifest["resources"]
        if item["path"] == SAMPLE_REL.as_posix()
    ]
    if len(source_rows) != 1 or source_rows[0]["sha256"] != SAMPLE_SHA256:
        raise ValueError("Co-Cd ASC is not uniquely source-bound in the manifest")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = load_rafm_profile(config["profile_name"])
    sample = read_raw_asc(
        sample_path,
        profile_name=profile.name,
        energy_calibration_override=profile.energy_calibration,
    )
    south, south_details = south_native_background()
    if sample.spectrum is None or sample.efficiency is None:
        raise ValueError("Co-Cd original count spectrum/profile efficiency unavailable")

    original_hashes = {
        rel.as_posix(): sha(ROOT / rel)
        for rel in (
            SAMPLE_REL, SOUTH_REL, MANIFEST_REL, SUPPLEMENT_REL, CONFIG_REL, PROFILE_REL
        )
    }
    sample_edges = midpoint_edges(sample.spectrum.energies)
    south_edges = midpoint_edges(south.energies)
    adjusted = subtract_measured_background(
        sample.spectrum, south, mode="live", negative_policy="preserve"
    )
    if not adjusted.metadata["background_subtraction"]["energy_aligned"]:
        raise ValueError("South background was not aligned by the current count-conserving engine")

    rows = []
    for energy, intensity in CO60_LINES:
        center = int(sample.energy_to_channel(energy))
        fwhm = sample.fwhm_at_energy(energy)
        slope = (
            profile.energy_calibration[1]
            + 2 * profile.energy_calibration[2] * center
        )
        fwhm_channels = fwhm / slope
        half = int(round(config["flux_wire_roi_width_fwhm"] / 2 * fwhm_channels))
        lo, hi = center - half, center + half
        selected = np.flatnonzero(
            (south_edges[:-1] < sample_edges[hi + 1])
            & (south_edges[1:] > sample_edges[lo])
        )
        blo, bhi = int(selected[0]), int(selected[-1])
        observation = native_observation(sample, sample_path, sample_edges, lo, hi)
        background_observation = CountObservation(
            south.counts[blo : bhi + 1],
            south_edges[blo : bhi + 2],
            south.live_time,
            SOUTH_SHA256,
            south.start_time.isoformat() if south.start_time else None,
        )
        peak = PeakResponse(energy, fwhm / FWHM_SIG_RATIO)
        efficiency = float(sample.efficiency.efficiency(energy))
        conversion = 1 / (sample.live_time * efficiency * intensity)

        iec_south = estimate_peak_area_local_background(
            adjusted.counts,
            center,
            fwhm_channels,
            roi_width_fwhm=float(config["flux_wire_roi_width_fwhm"]),
            background_width_channels=int(
                config["flux_wire_background_width_channels"]
            ),
            background_gap_fwhm=float(config["flux_wire_background_gap_fwhm"]),
            spectrum_data=adjusted,
        )
        iec_off = estimate_peak_area_local_background(
            sample.spectrum.counts,
            center,
            fwhm_channels,
            roi_width_fwhm=float(config["flux_wire_roi_width_fwhm"]),
            background_width_channels=int(
                config["flux_wire_background_width_channels"]
            ),
            background_gap_fwhm=float(config["flux_wire_background_gap_fwhm"]),
            spectrum_data=sample.spectrum,
        )
        if tuple(iec_south[4]) != (lo, hi) or tuple(iec_off[4]) != (lo, hi):
            raise RuntimeError("Fixed IEC/Covell control did not preserve the joint ROI")

        fits = {}
        for scenario in ("south_native", "ambient_off"):
            for continuum in ("linear", "step"):
                if scenario == "south_native":
                    choice = BackgroundChoice(
                        "later_conditional",
                        "South detector background postdates the sample; source matched but temporal applicability remains unresolved",
                        continuum="linear",
                        peaks=(peak,),
                    )
                    bg_obs = background_observation
                else:
                    choice = BackgroundChoice(
                        "no_separate_ambient_vendor",
                        "ambient-off sensitivity only; saved settings do not establish final vendor report behavior",
                        continuum="none",
                    )
                    bg_obs = None
                fit = fit_joint_poisson(
                    observation,
                    bg_obs,
                    peak,
                    choice,
                    sample_continuum=continuum,
                )
                item = _joint_payload(fit, observation, bg_obs)
                item["count_average_activity_bq_conditional"] = (
                    fit.area * conversion if fit.success else None
                )
                fits[f"{scenario}_{continuum}"] = item

        free = fit_joint_poisson(
            observation,
            background_observation,
            peak,
            BackgroundChoice(
                "later_conditional",
                "South free-normalization sensitivity; no independent normalization auxiliary",
                normalization="free",
                continuum="linear",
                peaks=(peak,),
            ),
            sample_continuum="linear",
            confidence=None,
        )
        fits["south_native_free_normalization"] = _joint_payload(
            free, observation, background_observation
        )

        rows.append(
            {
                "energy_keV": energy,
                "yield": intensity,
                "efficiency": efficiency,
                "fixed_fwhm_keV": fwhm,
                "fixed_roi_sample_channels_inclusive": [lo, hi],
                "fixed_roi_sample_edges_keV": [sample_edges[lo], sample_edges[hi + 1]],
                "south_native_channels_inclusive": [blo, bhi],
                "sample_original_counts": observation.counts,
                "south_original_counts": background_observation.counts,
                "sample_native_edges_keV": observation.energy_edges_keV,
                "south_native_edges_keV": background_observation.energy_edges_keV,
                "fixed_roi_iec_covell_control": {
                    "south_native": {
                        "net_counts": iec_south[0],
                        "std_counts_full_covariance": iec_south[1],
                        "gross_counts": iec_south[2],
                        "local_continuum_counts": iec_south[3],
                        "roi": list(iec_south[4]),
                        "count_average_activity_bq_conditional": iec_south[0]
                        * conversion,
                        "count_average_activity_std_bq_conditional": iec_south[1]
                        * conversion,
                    },
                    "ambient_off": {
                        "net_counts": iec_off[0],
                        "std_counts": iec_off[1],
                        "gross_counts": iec_off[2],
                        "local_continuum_counts": iec_off[3],
                        "roi": list(iec_off[4]),
                        "count_average_activity_bq_conditional": iec_off[0]
                        * conversion,
                        "count_average_activity_std_bq_conditional": iec_off[1]
                        * conversion,
                    },
                    "method": "current count-conserving South subtraction followed by fixed-ROI local-continuum IEC/Covell component",
                },
                "joint_fits": fits,
            }
        )

    after_hashes = {
        rel.as_posix(): sha(ROOT / rel)
        for rel in (
            SAMPLE_REL, SOUTH_REL, MANIFEST_REL, SUPPLEMENT_REL, CONFIG_REL, PROFILE_REL
        )
    }
    if original_hashes != after_hashes:
        raise RuntimeError("Source-bound inputs changed during pilot")

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None

    return {
        "issue": 249,
        "integration_base_head": BASE_HEAD,
        "observed_git_head": git_head,
        "scope": "source-bound two-line South-background joint-Poisson model check",
        "software_status": "pilot executable; model adequacy and physical qualification separate",
        "vendor_agreement_status": "not an optimization objective; QuantumGold target activities are not loaded",
        "physical_qualification_status": "NOT_QUALIFIED",
        "vendor_targets_loaded": False,
        "source_bound_inputs_unchanged": True,
        "inputs_sha256": original_hashes,
        "engine_files_sha256": {
            rel: sha(ROOT / rel)
            for rel in (
                "src/fluxforge/analysis/joint_poisson.py",
                "src/fluxforge/analysis/spectrum_math.py",
                "src/fluxforge/analysis/flux_wire_analysis.py",
                "src/fluxforge/io/flux_wire.py",
                "src/fluxforge/io/spe.py",
                "src/fluxforge/data/rafm_profile.py",
            )
        },
        "pilot_script_sha256": sha(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "sample": {
            "source_path": SAMPLE_REL.as_posix(),
            "source_sha256": sha(sample_path),
            "id": sample.sample_id,
            "start_time": sample.start_time.isoformat() if sample.start_time else None,
            "live_time_s": sample.live_time,
            "roi_energy_calibration": list(profile.energy_calibration),
            "roi_calibration_basis": "current nominal RAFM profile for direct current-engine comparability; original ASC counts unchanged",
        },
        "south_background": south_details,
        "background_applicability": "later_conditional; same detector but temporal applicability unresolved",
        "calibration_covariance": "UNAVAILABLE; never treated as zero",
        "grid_policy": "joint likelihood keeps native integer observations; IEC control uses current integrated-count overlap rebin with propagated covariance",
        "activity_reference": "count_average_live_normalized conditional conversion only; no decay/summing/attenuation or calibration covariance",
        "current_integrated_iec_control": _current_iec_control(),
        "synthetic_challenges": _synthetic_challenges(),
        "limitations": [
            "South background temporal applicability to the August 2025 sample is unresolved.",
            "Calibration/efficiency shared covariance is unavailable and excluded from conditional line uncertainties.",
            "Fixed Gaussian response and profile calibration are model assumptions; failed adequacy screens are preserved.",
            "Profile intervals are asymptotic and do not calibrate sparse/nonregular coverage.",
            "Ambient-off is a sensitivity control, not evidence of final QuantumGold settings.",
            "North background results from issue #239 remain cross-detector sensitivity only and are not used here.",
            "No QuantumGold activity or net-count target is loaded, fitted, or used to select a model.",
        ],
        "rows": rows,
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_outputs(payload: dict, output: Path) -> None:
    if output.exists():
        raise FileExistsError("output directory must be new")
    output.mkdir(parents=True)
    (output / "south_joint_poisson_pilot.json").write_text(
        json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary_fields = [
        "energy_keV",
        "scenario",
        "sample_continuum",
        "status",
        "success",
        "area_counts",
        "activity_bq_conditional",
        "sample_deviance",
        "background_deviance",
        "approximate_tail_probability",
        "adequacy_flag",
        "identifiability_ratio",
    ]
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for row in payload["rows"]:
            for name, fit in row["joint_fits"].items():
                scenario, continuum = (
                    ("south_native_free_normalization", "linear")
                    if name == "south_native_free_normalization"
                    else name.rsplit("_", 1)
                )
                diagnostic = fit["model_diagnostics"]
                writer.writerow(
                    {
                        "energy_keV": row["energy_keV"],
                        "scenario": scenario,
                        "sample_continuum": continuum,
                        "status": fit["status"],
                        "success": fit["success"],
                        "area_counts": fit["area_full_response_counts"],
                        "activity_bq_conditional": fit.get(
                            "count_average_activity_bq_conditional"
                        ),
                        "sample_deviance": diagnostic["sample_poisson_deviance"],
                        "background_deviance": diagnostic[
                            "background_poisson_deviance"
                        ],
                        "approximate_tail_probability": diagnostic[
