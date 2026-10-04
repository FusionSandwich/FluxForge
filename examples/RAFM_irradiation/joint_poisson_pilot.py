"""Bounded opt-in two-line Co-Cd control; no campaign or default mutations.

Run from the repository root with existing Python:
  python examples/RAFM_irradiation/joint_poisson_pilot.py --output NEW.json
No QuantumGold target values enter inference or scenario selection. The later
North ambient is a cross-detector conditional control, not a qualified physical
background. The no-separate-ambient case is a vendor convention scenario only.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy

from fluxforge.analysis.flux_wire_analysis import _qg_style_linear_continuum_counts
from fluxforge.analysis.joint_poisson import (
    BackgroundChoice,
    CountObservation,
    PeakResponse,
    fit_joint_poisson,
)
from fluxforge.analysis.peakfit import FWHM_SIG_RATIO, fit_single_peak
from fluxforge.analysis.spectrum_math import subtract_measured_background
from fluxforge.data.rafm_profile import load_rafm_profile
from fluxforge.io.flux_wire import read_raw_asc
from fluxforge.validation.example_identity import bound_example_engine, source_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def midpoint_edges(energies):
    """Match the current rebin midpoint/exterior-half-spacing edge policy."""
    return np.r_[
        energies[0] - (energies[1] - energies[0]) / 2,
        (energies[:-1] + energies[1:]) / 2,
        energies[-1] + (energies[-1] - energies[-2]) / 2,
    ]


def native_observation(data, path, edges, lo, hi):
    return CountObservation(
        data.spectrum.counts[lo : hi + 1],
        edges[lo : hi + 2],
        data.live_time,
        sha(path),
        data.start_time.isoformat() if data.start_time else None,
    )


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_pilot(evidence_root=None, *, engine_profile="integrated"):
    base = ROOT / "examples/RAFM_irradiation"
    sample_path = base / "raw_gamma_spec/flux_wires/Co-Cd-RAFM-1_25cm.ASC"
    background_path = base / "background.ASC"
    profile_path = ROOT / "src/fluxforge/data/rafm_profiles.json"
    config_path = base / "metadata/workflow_config.json"
    inputs = [sample_path, background_path, profile_path, config_path]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    historical_evidence = {"state": "source_receipt_not_supplied", "commit": "46096eb"}
    if evidence_root is not None:
        evidence_root = Path(evidence_root)
        finding = (
            evidence_root
            / "artifacts/validation/quantumgold_documentation_20261003/FINDINGS.txt"
        )
        original = (
            evidence_root
            / "examples/RAFM_irradiation/raw_gamma_spec/flux_wires/Co-Cd-RAFM-1_25cm.ASC"
        )
        if sha(original) != sha(sample_path):
            raise ValueError("pilot sample differs from source-bound original")
        historical_evidence = {
            "state": "sample_bytes_match_source_bound_original",
            "commit": "46096eb",
            "sample_sha256": sha(original),
            "FINDINGS_txt_sha256": sha(finding),
        }
    config = json.loads(config_path.read_text())
    profile = load_rafm_profile(config["profile_name"])
    # Same existing profile calibration and efficiency for every method.
    sample = read_raw_asc(
        sample_path,
        profile_name=profile.name,
        energy_calibration_override=profile.energy_calibration,
    )
    bg = read_raw_asc(background_path)
    if sample.spectrum is None or bg.spectrum is None or sample.efficiency is None:
        raise ValueError("required original counts/calibration/efficiency missing")
    es = midpoint_edges(sample.spectrum.energies)
    eb = midpoint_edges(bg.spectrum.energies)
    # Existing subtraction path, including W C W.T, for comparison only.
    adjusted = subtract_measured_background(
        sample.spectrum, bg.spectrum, mode="live", negative_policy="preserve"
    )
    source_ref = "a7bcc680d1f5e06b1d9dae405241fc380087ca2b"
    engine_files = [
        "src/fluxforge/analysis/peakfit.py",
        "src/fluxforge/analysis/spectrum_math.py",
        "src/fluxforge/analysis/flux_wire_analysis.py",
        "src/fluxforge/io/flux_wire.py",
        "src/fluxforge/data/rafm_profiles.json",
    ]
    # These SHA256 pins were computed from the local a7bcc68 Git objects after
    # canonical-LF normalization. They preserve the exact source check offline.
    engine_pins = {
        "src/fluxforge/analysis/peakfit.py": "53a0001131d8410bfbac113ff2adb5bbd79be653784931ab8711d747ea97c226",
        "src/fluxforge/analysis/spectrum_math.py": "d5bac0936affbac6175216e0034047e8fbef25d2e29c480db4583ac1dbd0a488",
        "src/fluxforge/analysis/flux_wire_analysis.py": "5b5fe5516ff0bdc630cdbbdc1b7672d7eb88e1b6c662c36f5d91cf3a8cfe6b32",
        "src/fluxforge/io/flux_wire.py": "1a6d2fad25f4ac7863da1eb5ac7677bd091c73a36a01edfcdafc3e82a814d377",
        "src/fluxforge/data/rafm_profiles.json": "7fb3ab63f303bd8bacc3117257ff518834d5845669e8bcc9c2dc7adf6587acea",
    }
    if engine_profile == "integrated":
        identity = bound_example_engine(
            ROOT,
            ROOT / "examples/engine_bindings/qg_sensitivity_review.json",
            engine_files,
        )
    elif engine_profile == "historical":
        identity = source_identity(
            ROOT,
            engine_files,
            expected_sha256=engine_pins,
            canonical_lf=True,
            required_ancestor=source_ref,
        )
        identity.update(profile="historical", declared_source_revision=source_ref)
    else:
        raise ValueError("Unknown engine profile")
    engine = identity["files_sha256"]
    rows = []
    # Co60 energies/yields are fixed inputs, not report net-count targets.
    for energy, intensity in [(1173.228, 0.9985), (1332.492, 0.999826)]:
        center = int(sample.energy_to_channel(energy))
        fwhm = sample.fwhm_at_energy(energy)
        slope = (
            profile.energy_calibration[1] + 2 * profile.energy_calibration[2] * center
        )
        half = int(round(config["flux_wire_roi_width_fwhm"] / 2 * fwhm / slope))
        lo, hi = center - half, center + half
        # Use complete native ambient bins touching the same energy ROI; never
        # split integer counts into fractional Poisson observations.
        selected = np.flatnonzero((eb[:-1] < es[hi + 1]) & (eb[1:] > es[lo]))
        blo, bhi = int(selected[0]), int(selected[-1])
        s = native_observation(sample, sample_path, es, lo, hi)
        b = native_observation(bg, background_path, eb, blo, bhi)
        peak = PeakResponse(energy, fwhm / FWHM_SIG_RATIO)
        efficiency = float(sample.efficiency.efficiency(energy))
        conversion = 1 / (sample.live_time * efficiency * intensity)
        for has_ambient in (True, False):
            current = adjusted if has_ambient else sample.spectrum
            endpoint_net, _, gross, cont = _qg_style_linear_continuum_counts(
                current.counts, lo, hi
            )
            weights = np.zeros(len(current.counts))
            weights[lo : hi + 1] = 1
            weights[lo] -= (hi - lo + 1) / 2
            weights[hi] -= (hi - lo + 1) / 2
            endpoint_std = float(np.sqrt(current.weighted_counts_variance(weights)))
            gaussian = fit_single_peak(
                current.channels,
                current.counts,
                center,
                fit_width=half,
                initial_sigma=fwhm / slope / FWHM_SIG_RATIO,
                background_model="linear",
                counts_uncertainty=current.counts_uncertainty,
                counts_covariance=current.count_covariance_matrix(),
            )
            for continuum in ("linear", "step"):
                choice = (
                    BackgroundChoice(
                        "later_conditional",
                        "North ambient March 2026 postdates August 2025 South sample; cross-detector applicability unqualified",
                        continuum="linear",
                        peaks=(peak,),
                    )
                    if has_ambient
                    else BackgroundChoice(
                        "no_separate_ambient_vendor",
                        "saved study settings support a vendor ambient-off scenario; final report settings unverified",
                        continuum="none",
                    )
                )
                fit = fit_joint_poisson(
                    s,
                    b if has_ambient else None,
                    peak,
                    choice,
                    sample_continuum=continuum,
                )
                rows.append(
                    {
                        "energy_keV": energy,
                        "yield": intensity,
                        "efficiency": efficiency,
                        "efficiency_parameters": sample.efficiency.to_dict(),
                        "roi_sample_channels_inclusive": [lo, hi],
                        "roi_sample_edges_keV": [es[lo], es[hi + 1]],
                        "roi_ambient_channels_inclusive": (
                            [blo, bhi] if has_ambient else None
                        ),
                        "fixed_fwhm_keV": fwhm,
                        "sample_original_counts": s.counts,
                        "ambient_original_counts": b.counts if has_ambient else None,
                        "sample_native_edges_keV": s.energy_edges_keV,
                        "ambient_native_edges_keV": (
                            b.energy_edges_keV if has_ambient else None
                        ),
                        "applicability": choice.applicability,
                        "sample_continuum": continuum,
                        "current_endpoint_continuum": {
                            "method": "current_qg_style_endpoint_linear_continuum_same_fixed_ROI",
                            "net": endpoint_net,
                            "std_full_rebin_covariance": endpoint_std,
                            "gross": gross,
                            "continuum": cont,
                            "count_basis": (
                                "signed_live_scaled_ambient_residual"
                                if has_ambient
                                else "original_sample"
                            ),
                            "conversion_count_rate_over_efficiency_yield": endpoint_net
                            * conversion,
                        },
                        "current_gaussian": {
                            "method": "current_fit_single_peak_signed_GLS_same_fixed_ROI",
                            "success": gaussian.success,
                            "message": gaussian.message,
                            "area": gaussian.net_counts if gaussian.success else None,
                            "std": (
                                gaussian.net_counts_uncertainty
                                if gaussian.success
                                else None
                            ),
                            "residuals": gaussian.residuals,
                            "fit_region": gaussian.fit_region,
                            "centroid_width": "fitted; joint response is fixed from nominal energy/profile",
                            "count_basis": (
                                "signed_live_scaled_ambient_residual"
                                if has_ambient
                                else "original_sample"
                            ),
                        },
                        "joint": {
                            "success": fit.success,
                            "status": fit.status,
                            "message": fit.message,
                            "area": fit.area,
                            "area_in_sample_ROI": fit.area
                            * peak.integrated(s.energy_edges_keV).sum(),
                            "normalization": fit.normalization,
                            "exposure_scale": fit.exposure_scale,
                            "profile": asdict(fit.interval) if fit.interval else None,
                            "deviance": fit.deviance,
                            "model_diagnostics": fit.model_diagnostics,
                            "parameter_names": fit.parameter_names,
                            "parameters": fit.parameters,
                            "sample_expected": fit.sample_expected,
                            "ambient_expected": fit.background_expected,
                            "sample_residuals": fit.sample_residuals,
                            "ambient_residuals": fit.background_residuals,
                            "provenance": fit.provenance,
                            "conversion_count_rate_over_efficiency_yield": (
                                fit.area * conversion if fit.success else None
                            ),
                        },
                        "vendor_report_counts": None,
                        "whole_workflow_comparison": "pending integration #232/#220",
                    }
                )
        free = fit_joint_poisson(
            s,
            b,
            peak,
            BackgroundChoice(
                "later_conditional",
                "free normalization sensitivity; no independent normalization auxiliary",
                normalization="free",
                continuum="linear",
                peaks=(peak,),
            ),
            confidence=None,
        )
        rows.append(
            {
                "energy_keV": energy,
                "scenario": "free_normalization_tradeoff_control",
                "success": free.success,
                "status": free.status,
                "message": free.message,
                "identifiability_ratio": free.identifiability_ratio,
            }
        )
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    if before != after:
        raise RuntimeError("original input identity changed")
    return {
        "issue": 239,
        "engine_base": identity["declared_source_revision"],
        "historical_engine_binding": {
            "revision": source_ref,
            "files_sha256": engine_pins,
            "qualification": "Historical provenance, not the engine used in this run",
        },
        "engine_files_sha256_canonical_lf": engine,
        "engine_source_identity": identity,
        "implementation_sha256": sha(ROOT / "src/fluxforge/analysis/joint_poisson.py"),
        "pilot_script_sha256": sha(Path(__file__)),
        "inputs_sha256": before,
        "originals_byte_identical_after_run": True,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "numpy_repo_constraint": ">=1.26,<2.0",
            "dependency_qualification": (
                "numpy satisfies declared range; environment unchanged"
                if (1, 26)
                <= tuple(int(v) for v in np.__version__.split(".")[:2])
                < (2, 0)
                else "numpy outside declared range; environment unchanged"
            ),
        },
        "sample_identity": {
            "id": sample.sample_id,
            "date": str(sample.start_time),
            "live_time": sample.live_time,
            "energy_calibration": sample.energy_calibration,
        },
        "ambient_identity": {
            "id": bg.sample_id,
            "date": str(bg.start_time),
            "live_time": bg.live_time,
            "energy_calibration": bg.energy_calibration,
        },
        "scope": "two isolated Co60 ROIs; same ROI and efficiency across methods; no background selection or QG-target tuning",
        "source_evidence": "46096eb artifacts/validation/quantumgold_documentation_20261003/FINDINGS.txt; saved flags do not establish final report settings",
        "source_evidence_identity": historical_evidence,
        "grid_policy": "midpoint_centers_exterior_half_spacing; native original counts for joint; existing rebin/covariance for comparisons",
        "conversion_basis": "count-average rate divided by fixed efficiency and yield; no decay, summing, attenuation, calibration or efficiency uncertainty",
        "limitations": [
            "conditional later cross-detector ambient; no qualified physical background",
            "fixed peak shape/calibration",
            "asymptotic profiles, sparse-count coverage uncalibrated",
            "active nuisance boundaries and strong observed lack of fit: point estimates and nominal intervals exploratory only",
            "component controls do not reproduce current whole-workflow branch heuristics",
            "deviance values include different observations across ambient/vendor scenarios; do not compare them to select a background",
            "physical joint counts remain distinct from vendor comparison counts; defaults unchanged",
        ],
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engine-profile", choices=("integrated", "historical"), default="integrated"
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help="read-only local source-bound 46096eb checkout",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be new")
    payload = run_pilot(args.evidence_root, engine_profile=args.engine_profile)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=json_value, allow_nan=False)
    print(
        json.dumps(
            {
                "rows": len(payload["rows"]),
                "output": str(args.output),
                "statuses": [r.get("joint", r).get("status") for r in payload["rows"]],
            }
        )
    )
