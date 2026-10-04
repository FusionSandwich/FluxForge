"""Opt-in source-bound Co-Cd/South model checks for issue #249.

Run with existing dependencies and a fresh output directory:
  python examples/RAFM_irradiation/south_joint_poisson_pilot.py --output NEW_FOLDER
No campaign, source acquisition, vendor target fitting or default mutations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.special import kl_div, ndtr

from fluxforge.analysis.flux_wire_analysis import estimate_peak_area_local_background
from fluxforge.analysis.joint_poisson import (
    BackgroundChoice,
    CountObservation,
    PeakResponse,
    fit_joint_poisson,
)
from fluxforge.analysis.peakfit import FWHM_SIG_RATIO
from fluxforge.analysis.spectrum_math import (
    _resample_background_to_sample_energy,
    subtract_measured_background,
)
from fluxforge.data.rafm_profile import load_rafm_profile
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.validation.example_identity import source_identity

LINES = ((1173.228, 0.9985), (1332.492, 0.999826))
INTEGRATION_BASE = "701288a30f418ed7dc331e6deda0044eb220e72e"
SOURCE_FILES = (
    "pyproject.toml",
    "examples/RAFM_irradiation/south_joint_poisson_pilot.py",
    "examples/RAFM_irradiation/run_portable_qg_example.py",
    "src/fluxforge/analysis/joint_poisson.py",
    "src/fluxforge/analysis/flux_wire_analysis.py",
    "src/fluxforge/analysis/peakfit.py",
    "src/fluxforge/analysis/spectrum_math.py",
    "src/fluxforge/io/flux_wire.py",
    "src/fluxforge/io/spe.py",
    "src/fluxforge/data/rafm_profile.py",
    "src/fluxforge/data/rafm_profiles.json",
    "src/fluxforge/data/xcom.py",
    "src/fluxforge/core/calibration.py",
    "src/fluxforge/validation/example_identity.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_driver():
    path = ROOT / "examples/RAFM_irradiation/run_portable_qg_example.py"
    spec = importlib.util.spec_from_file_location("south_pilot_source_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def midpoint_edges(energy):
    return np.r_[
        energy[0] - (energy[1] - energy[0]) / 2,
        (energy[:-1] + energy[1:]) / 2,
        energy[-1] + (energy[-1] - energy[-2]) / 2,
    ]


def poisson_diagnostics(counts, expected, edges, centroid):
    """Signed deviance and Pearson residuals, including exact zero means.

    Undefined/infinite residuals are JSON null with an explicit status. Segment
    sums are descriptive localization, not calibrated multiple-test p values.
    """
    counts, expected = np.asarray(counts), np.asarray(expected)
    contribution = 2 * kl_div(counts, expected)
    deviance = np.sign(counts - expected) * np.sqrt(contribution)
    pearson = np.divide(
        counts - expected,
        np.sqrt(expected),
        out=np.zeros_like(expected),
        where=expected > 0,
    )
    pearson[(expected == 0) & (counts > 0)] = np.inf
    centers = (edges[:-1] + edges[1:]) / 2
    segments = {}
    for label, mask in (
        ("below_centroid", centers < centroid),
        ("at_or_above_centroid", centers >= centroid),
    ):
        value = float(contribution[mask].sum())
        segments[label] = dict(
            observed=float(counts[mask].sum()),
            expected=float(expected[mask].sum()),
            deviance=value if np.isfinite(value) else None,
        )
    return dict(
        signed_deviance=[float(x) if np.isfinite(x) else None for x in deviance],
        pearson=[float(x) if np.isfinite(x) else None for x in pearson],
        nonfinite_bins=np.flatnonzero(~np.isfinite(contribution)).tolist(),
        status="FINITE" if np.all(np.isfinite(contribution)) else "INFINITE_DEVIANCE",
        segments=segments,
    )


def fit_receipt(fit, sample, background, peak, conversion=None):
    # Candidate parameters survive every failure; activity conversion is withheld.
    return dict(
        success=fit.success,
        status=fit.status,
        message=fit.message,
        candidate_area=fit.area,
        candidate_area_in_sample_ROI=fit.area
        * peak.integrated(sample.energy_edges_keV).sum(),
        count_average_activity_bq=(
            fit.area * conversion if fit.success and conversion is not None else None
        ),
        physical_activity_qualified=False,
        profile=asdict(fit.interval) if fit.interval else None,
        normalization=fit.normalization,
        exposure_scale=fit.exposure_scale,
        deviance=fit.deviance,
        model_diagnostics=fit.model_diagnostics,
        identifiability_ratio=fit.identifiability_ratio,
        parameter_names=fit.parameter_names,
        parameters=fit.parameters,
        sample_expected=fit.sample_expected,
        ambient_expected=fit.background_expected,
        sample_residuals=fit.sample_residuals,
        ambient_residuals=fit.background_residuals,
        sample_poisson_residuals=poisson_diagnostics(
            sample.counts,
            fit.sample_expected,
            sample.energy_edges_keV,
            peak.centroid_keV,
        ),
        ambient_poisson_residuals=(
            poisson_diagnostics(
                background.counts,
                fit.background_expected,
                background.energy_edges_keV,
                peak.centroid_keV,
            )
            if background
            else None
        ),
        provenance=fit.provenance,
    )


def fixed_iec_control(spectrum, center, fwhm_channels, config, conversion):
    # Fixed Covell singlet component of the current IEC-inspired policy. Supplying
    # spectrum_data retains W C W.T and ROI/sideband cross terms after rebinning.
    net, std, gross, continuum, bounds = estimate_peak_area_local_background(
        spectrum.counts,
        center,
        fwhm_channels,
        roi_width_fwhm=config["flux_wire_roi_width_fwhm"],
        background_width_channels=config["flux_wire_background_width_channels"],
        background_gap_fwhm=config["flux_wire_background_gap_fwhm"],
        spectrum_data=spectrum,
    )
    return dict(
        method="current_IEC_inspired_fixed_Covell_singlet_control",
        roi_sample_channels_inclusive=list(bounds),
        net=net,
        std_full_covariance=std,
        gross=gross,
        continuum=continuum,
        count_average_activity_bq=net * conversion,
        physical_activity_qualified=False,
        scope="fixed ROI control; full iec_tiered moving-minimum/fit selection is not invoked",
    )


def synthetic_challenges():
    """Independent rounded Asimov native counts; bounded diagnostic challenges.

    Not a coverage calibration: independent CDF truth uses unequal grids and a
    coincident ambient peak. Strong deliberate misspecification must stay visible.
    """
    es, eb = np.linspace(-8, 8, 65), np.linspace(-8.1, 8.1, 82)
    sigma, ratio = 0.8, 4.0
    peak = PeakResponse(0, sigma)
    ps, pb = np.diff(ndtr(es / sigma)), np.diff(ndtr(eb / sigma))
    bs, bb = 40 * np.diff(es) + 200 * ps, 40 * np.diff(eb) + 200 * pb
    choice = BackgroundChoice(
        "later_conditional",
        "synthetic independent background",
        continuum="constant",
        peaks=(peak,),
    )
    rows = []
    for name, area, signal, free, maxiter in (
        ("zero_signal", 0, ps, False, 2000),
        ("weak_signal", 50, ps, False, 2000),
        ("strong_signal", 4000, ps, False, 2000),
        (
            "shifted_response_misspecification",
            4000,
            np.diff(ndtr((es - 0.8) / sigma)),
            False,
            2000,
        ),
        (
            "broad_response_misspecification",
            4000,
            np.diff(ndtr(es / (sigma * 1.8))),
            False,
            2000,
        ),
        ("normalization_unidentifiable", 4000, ps, True, 2000),
        ("optimizer_budget_failure", 4000, ps, False, 1),
    ):
        s = CountObservation(
            np.rint(area * signal + bs + 10 * np.diff(es)), es, 10, "synthetic_sample"
        )
        b = CountObservation(np.rint(ratio * bb), eb, 40, "synthetic_ambient")
        declared = BackgroundChoice(
            **{
                **asdict(choice),
                "peaks": (peak,),
                "normalization": "free" if free else "fixed",
            }
        )
        fit = fit_joint_poisson(
            s, b, peak, declared, sample_continuum="constant", maxiter=maxiter
        )
        rows.append(
            dict(
                name=name,
                truth_area=area,
                generation="rounded independent CDF Asimov; no random sampling",
                sample_counts=s.counts,
                ambient_counts=b.counts,
                joint=fit_receipt(fit, s, b, peak),
            )
        )
    return rows


def run_pilot():
    driver = load_driver()
    runtime = driver.runtime_compatibility(ROOT)
    if runtime["status"] != "COMPATIBLE":
        raise RuntimeError("Declared core dependencies incompatible: " + str(runtime))
    checked = driver.verify_inputs(ROOT, driver.MANIFEST_PATH)
    manifest = checked["manifest"]
    row = next(
        r for r in manifest["measurements"] if r["measurement_id"] == "Co-Cd-RAFM-1"
    )
    sample_path = driver.bound_path(ROOT, row["files"]["ASC"])
    config_path = ROOT / "examples/RAFM_irradiation/metadata/workflow_config.json"
    config = json.loads(config_path.read_text())
    profile = load_rafm_profile(config["profile_name"])
    # Match the current integration's nominal sample grid. The original ASC and
    # ANS calibrations remain explicit; no claim of recovered calibration is made.
    sample = read_raw_asc(
        sample_path,
        profile_name=profile.name,
        energy_calibration_override=profile.energy_calibration,
    )
    if not np.array_equal(
        sample.spectrum.counts, checked["arrays"][row["measurement_id"]]
    ):
        raise ValueError("Original ASC counts differ from source-bound native array")
    sample.spectrum.detector_id = "South"
    background, background_details = driver.background_scenario(
        ROOT, checked, "south_native"
    )
    supplement_path = (
        ROOT
        / "examples/RAFM_irradiation/quantumgold_reference/supplemental_inputs/manifest.json"
    )
    supplement = json.loads(supplement_path.read_text())
    background_pin = next(
        p
        for p in supplement["resources"]
        if p["role"] == "recovered_South_native_background_not_ASC"
    )
    input_paths = [
        config_path,
        driver.MANIFEST_PATH,
        supplement_path,
        ROOT / "src/fluxforge/data/rafm_profiles.json",
        driver.bound_path(ROOT, background_pin["path"]),
        *[driver.bound_path(ROOT, name) for name in row["files"].values() if name],
    ]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths}
    es, eb = midpoint_edges(sample.spectrum.energies), midpoint_edges(
        background.energies
    )
    adjusted = subtract_measured_background(
        sample.spectrum, background, mode="live", negative_policy="preserve"
    )
    aligned, _ = _resample_background_to_sample_energy(sample.spectrum, background)
    # Conservation applies to overlapping support, not bins outside the target.
    covered = np.maximum(
        0, np.minimum(eb[1:], es[-1]) - np.maximum(eb[:-1], es[0])
    ) / np.diff(eb)
    overlap_count = float(covered @ background.counts)
    if not np.isclose(aligned.counts.sum(), overlap_count, rtol=1e-12, atol=1e-8):
        raise RuntimeError("Count-conserving comparison engine failed overlap receipt")
    conservation = dict(
        native_total=float(background.counts.sum()),
        expected_overlap_total=overlap_count,
        rebinned_total=float(aligned.counts.sum()),
        outside_target_support=float(background.counts.sum() - overlap_count),
        residual=float(aligned.counts.sum() - overlap_count),
        joint_observations_rebinned=False,
        comparison_covariance_propagated=adjusted.counts_covariance is not None,
    )
    rows = []
    for energy, intensity in LINES:
        center = int(sample.energy_to_channel(energy))
        fwhm = sample.fwhm_at_energy(energy)
        slope = (
            profile.energy_calibration[1] + 2 * profile.energy_calibration[2] * center
        )
        half = max(1, int(round(config["flux_wire_roi_width_fwhm"] / 2 * fwhm / slope)))
        lo, hi = center - half, center + half
        selected = np.flatnonzero((eb[:-1] < es[hi + 1]) & (eb[1:] > es[lo]))
        blo, bhi = int(selected[0]), int(selected[-1])
        s = CountObservation(
            sample.spectrum.counts[lo : hi + 1],
            es[lo : hi + 2],
            sample.live_time,
            sha(sample_path),
            sample.start_time.isoformat(),
        )
        b = CountObservation(
            background.counts[blo : bhi + 1],
            eb[blo : bhi + 2],
            background.live_time,
            background_details["source_sha256"],
            background.start_time.isoformat(),
        )
        peak = PeakResponse(energy, fwhm / FWHM_SIG_RATIO)
        efficiency = float(sample.efficiency.efficiency(energy))
        conversion = 1 / (sample.live_time * efficiency * intensity)
        for mode in ("south_native", "ambient_off"):
            ambient = b if mode == "south_native" else None
            current = adjusted if ambient else sample.spectrum
            iec = fixed_iec_control(current, center, fwhm / slope, config, conversion)
            if iec["roi_sample_channels_inclusive"] != [lo, hi]:
                raise RuntimeError("IEC and Poisson fixed ROI differ")
            choice = (
                BackgroundChoice(
                    "later_conditional",
                    "South 2025-10-03 postdates 2025-08-28 sample; temporal applicability unresolved",
                    continuum="linear",
                    peaks=(peak,),
                )
                if ambient
                else BackgroundChoice(
                    "no_separate_ambient_vendor",
                    "declared ambient-off sensitivity; saved setting does not establish final vendor behavior",
                    continuum="none",
                )
            )
            # Baseline signed residuals show an antisymmetric centroid pattern
            # and core/wing structure in both modes. A finite declared response
            # challenge uses +/- half a sample bin and +/-20% width, separately.
            # This is not a calibration fit, uncertainty distribution or search
            # for vendor agreement. Ambient response stays nominal on its own grid.
            responses = [
                ("nominal", peak, "linear"),
                ("nominal", peak, "step"),
                (
                    "centroid_minus_half_bin",
                    PeakResponse(energy - slope / 2, peak.sigma_keV),
                    "linear",
                ),
                (
                    "centroid_plus_half_bin",
                    PeakResponse(energy + slope / 2, peak.sigma_keV),
                    "linear",
                ),
                (
                    "width_times_0.8",
                    PeakResponse(energy, peak.sigma_keV * 0.8),
                    "linear",
                ),
                (
                    "width_times_1.2",
                    PeakResponse(energy, peak.sigma_keV * 1.2),
                    "linear",
                ),
            ]
            for response_name, candidate_peak, continuum in responses:
                fit = fit_joint_poisson(
                    s, ambient, candidate_peak, choice, sample_continuum=continuum
                )
                rows.append(
                    dict(
                        energy_keV=energy,
                        scenario=mode,
                        response=response_name,
                        sample_continuum=continuum,
                        roi_sample_channels_inclusive=[lo, hi],
                        roi_ambient_channels_inclusive=[blo, bhi] if ambient else None,
                        sample_original_counts=s.counts,
                        ambient_original_counts=b.counts if ambient else None,
                        sample_native_edges_keV=s.energy_edges_keV,
                        ambient_native_edges_keV=(
                            b.energy_edges_keV if ambient else None
                        ),
                        fixed_fwhm_keV=fwhm,
                        efficiency=efficiency,
                        yield_fraction=intensity,
                        iec=iec,
                        joint=fit_receipt(fit, s, ambient, candidate_peak, conversion),
                    )
                )
        free_choice = BackgroundChoice(
            "later_conditional",
            "free normalization without independent auxiliary; tradeoff control",
            normalization="free",
            continuum="linear",
            peaks=(peak,),
        )
        fit = fit_joint_poisson(s, b, peak, free_choice, confidence=None)
        rows.append(
            dict(
                energy_keV=energy,
                scenario="south_free_normalization",
                response="nominal",
                sample_continuum="linear",
                roi_sample_channels_inclusive=[lo, hi],
                roi_ambient_channels_inclusive=[blo, bhi],
                sample_original_counts=s.counts,
                ambient_original_counts=b.counts,
                joint=fit_receipt(fit, s, b, peak, conversion),
            )
        )
    after = driver.verify_inputs(ROOT, driver.MANIFEST_PATH)
    if (
        before != {p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths}
        or checked["manifest_sha256"] != after["manifest_sha256"]
    ):
        raise RuntimeError("Source inputs changed during pilot")
    return dict(
        issue=249,
        status="BOUNDED_SOFTWARE_PILOT_COMPLETED",
        engine=source_identity(
            ROOT, SOURCE_FILES, required_ancestor=INTEGRATION_BASE, canonical_lf=True
        ),
        complete_engine=driver.engine_identity(ROOT),
        inputs_sha256=before,
        dataset_manifest_sha256=checked["manifest_sha256"],
        runtime=runtime,
        originals_byte_identical_after_run=True,
        conservation=conservation,
        sample_identity=dict(
            measurement_id=row["measurement_id"],
            detector="South",
            count_basis="independent_original_ASC_matches_ANS",
            live_time_s=sample.live_time,
            real_time_s=sample.real_time,
            start_time_unzoned=sample.start_time.isoformat(),
            used_energy_polynomial_keV=sample.energy_calibration,
            original_ASC_header=row["ASC_header"],
            original_ANS_energy_polynomial_keV=row["native_energy_polynomial_keV"],
            timeline=row["timeline"],
        ),
        ambient_identity={
            **background_details,
            "real_time_s": background.real_time,
            "used_energy_polynomial_keV": background.calibration["energy"],
            "source_path": background_pin["path"],
        },
        scientific_admission=False,
        exact_vendor_parity=False,
        physical_inversion_qualified=False,
        activity_reference="count_average_live_normalized; no count-decay correction",
        calibration_covariance="UNKNOWN; not zero, not included in conditional intervals",
        vendor_activity_reference="report Measurement Date; unresolved clock/reference semantics; no target enters inference",
        limitations=[
            "South temporal applicability unresolved; detector identity alone does not qualify the background",
            "Nominal sample profile and fixed response conditional; original ASC/ANS calibration disagreement retained",
            "Nominal efficiency/yield conversion excludes calibration covariance, summing, attenuation and decay",
            "Asymptotic deviance/profile screening is not calibrated sparse-count coverage",
            "Deviance totals across ambient-off and South include different observations; do not use them to choose background",
            "IEC fixed singlet control is distinct from full tiered campaign selection",
            "North remains a cross-detector sensitivity in the existing issue #239 pilot; no North qualification implied",
        ],
        rows=rows,
        synthetic_challenges=synthetic_challenges(),
    )


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_receipts(output, payload):
    output.mkdir(parents=True, exist_ok=False)
    (output / "pilot.json").write_text(
        json.dumps(payload, indent=2, default=json_value, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "energy_keV",
            "scenario",
            "response",
            "sample_continuum",
            "status",
            "adequacy",
            "candidate_area",
            "conditional_activity_bq",
            "deviance",
            "sample_deviance",
            "ambient_deviance",
            "iec_fixed_net",
            "iec_conditional_activity_bq",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            fit, iec = row["joint"], row.get("iec", {})
            writer.writerow(
                {
                    **{k: row[k] for k in fields[:4]},
                    "status": fit["status"],
                    "adequacy": fit["model_diagnostics"]["adequacy_flag"],
                    "candidate_area": fit["candidate_area"],
                    "conditional_activity_bq": fit["count_average_activity_bq"],
                    "deviance": fit["deviance"],
                    "sample_deviance": fit["model_diagnostics"][
                        "sample_poisson_deviance"
                    ],
                    "ambient_deviance": fit["model_diagnostics"][
                        "background_poisson_deviance"
                    ],
                    "iec_fixed_net": iec.get("net"),
                    "iec_conditional_activity_bq": iec.get("count_average_activity_bq"),
                }
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), layout="constrained")
    for col, (energy, _) in enumerate(LINES):
        for scenario_index, scenario in enumerate(("south_native", "ambient_off")):
            ax, residual_ax = (
                axes[2 * scenario_index, col],
                axes[2 * scenario_index + 1, col],
            )
            for row in payload["rows"]:
                if row["energy_keV"] != energy or row["scenario"] != scenario:
                    continue
                centers = (
                    np.array(row["sample_native_edges_keV"][:-1])
                    + row["sample_native_edges_keV"][1:]
                ) / 2
                fit = row["joint"]
                label = (
                    row["response"]
                    + "/"
                    + row["sample_continuum"]
                    + "; "
                    + fit["status"]
                )
                if row["sample_continuum"] == "linear" and row["response"] == "nominal":
                    ax.step(
                        centers,
                        row["sample_original_counts"],
                        where="mid",
                        color="black",
                        label="original sample",
                    )
                ax.plot(centers, fit["sample_expected"], label=label)
                residual_ax.plot(
                    centers,
                    fit["sample_poisson_residuals"]["signed_deviance"],
                    marker=".",
                    label=label,
                )
            ax.set(title=f"{energy:.3f} keV / {scenario}", ylabel="native counts")
            residual_ax.axhline(0, color="black", linewidth=0.7)
            residual_ax.set(xlabel="energy (keV)", ylabel="signed deviance residual")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Fixed ROIs; numerical fits remain conditional and physically unqualified"
    )
    fig.savefig(output / "sample_residuals.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    for col, (energy, _) in enumerate(LINES):
        for row in payload["rows"]:
            if (
                row["energy_keV"] != energy
                or row["scenario"] != "south_native"
                or row["response"] != "nominal"
            ):
                continue
            edges = np.asarray(row["ambient_native_edges_keV"])
            centers = (edges[:-1] + edges[1:]) / 2
            fit = row["joint"]
            if row["sample_continuum"] == "linear":
                axes[0, col].step(
                    centers,
                    row["ambient_original_counts"],
                    where="mid",
                    color="black",
                    label="native South",
                )
            axes[0, col].plot(
                centers, fit["ambient_expected"], label=row["sample_continuum"]
            )
            axes[1, col].plot(
                centers,
                fit["ambient_poisson_residuals"]["signed_deviance"],
                marker=".",
                label=row["sample_continuum"],
            )
        axes[0, col].set(
            title=f"South background / {energy:.3f} keV", ylabel="native counts"
        )
        axes[0, col].legend()
        axes[1, col].axhline(0, color="black", linewidth=0.7)
        axes[1, col].set(xlabel="energy (keV)", ylabel="signed deviance residual")
    fig.suptitle("Native South bins; temporal applicability remains unresolved")
    fig.savefig(output / "ambient_residuals.png", dpi=150)
    plt.close(fig)
    hashes = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "OUTPUT_HASHES.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    payload = run_pilot()
    save_receipts(args.output.resolve(), payload)
    print(
        json.dumps(
            dict(
                status=payload["status"],
                rows=len(payload["rows"]),
                scientific_admission=False,
            )
        )
    )
