"""k0 workflow helpers for normalized observations and governed artifacts.

This module provides the first standards-oriented boundary layer between peak
analysis and k0 interpretation. It keeps raw peak fitting/import separate from
k0 calculations by normalizing peak observations, applying explicit line
eligibility decisions, and producing structured inputs for detector/facility and
sample-analysis artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from fluxforge.analysis.detector_calibration import EfficiencyPoint, fit_efficiency_curve
from fluxforge.analysis.k0_naa import K0Calculator, K0Measurement, K0Parameters, create_k0_measurement_from_peak, identify_isotope_from_gamma, reset_k0_database, set_k0_database_from_governed_library
from fluxforge.data.efficiency import EfficiencyCurve
from fluxforge.data.k0_library import GovernedLibrary, get_active_auxiliary_correction_library, get_active_k0_library, get_k0_library_record, get_library_summary, load_governed_library, use_governed_libraries
from fluxforge.triga.k0 import TRIGAIrradiationParams, TRIGAk0Workflow, get_westcott_g, triple_monitor_method
from fluxforge.uncertainty.budget import create_k0_naa_budget


CAPABILITY_FLAGS: Dict[str, str] = {
    "supports_thermal_inaa": "validated",
    "supports_epithermal_cadmium_naa": "partial",
    "supports_low_energy_photon_mode": "not_validated",
    "supports_prompt_gamma_mode": "unsupported",
    "supports_fast_flux_threshold_corrections": "partial",
}


@dataclass(frozen=True)
class PeakObservation:
    """Normalized peak observation consumed by the k0 layer."""

    peak_id: str
    source_spectrum_id: str
    detector_id: str
    geometry_id: str
    position_mm: float | None
    line_energy_keV: float
    line_id: str
    assigned_radionuclide: str | None
    net_peak_area: float
    area_uncertainty: float
    live_time_s: float
    real_time_s: float
    count_start_time: str | None
    reference_time: str | None
    irradiation_time_s: float
    decay_time_s: float
    counting_time_s: float
    dead_time_correction_method: str
    baseline_method: str
    deconvolution_status: str
    interference_flags: tuple[str, ...] = ()
    analyst_review_status: str = "unreviewed"
    peak_area_provenance: str = "imported_peak_table"
    import_format: str = "peak_report"
    peak_class: str = "photopeak"
    reaction_family: str = "thermal_capture"
    gamma_yield: float = 1.0
    efficiency: float | None = None
    efficiency_uncertainty: float | None = None
    emission_probability: float | None = None
    sample_role: str = "sample"
    sample_mass_g: float | None = None
    project_id: str | None = None
    sample_id: str | None = None
    irradiation_id: str | None = None
    measurement_id: str | None = None
    g_th: float = 1.0
    g_ep: float = 1.0
    cd_factor: float = 1.0
    eligibility_class: str = "direct_k0_eligible"
    eligibility_accepted: bool = True
    rejection_reasons: tuple[str, ...] = ()
    expert_override: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["interference_flags"] = list(self.interference_flags)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PeakObservation":
        return cls(
            peak_id=str(payload.get("peak_id") or "peak"),
            source_spectrum_id=str(payload.get("source_spectrum_id") or ""),
            detector_id=str(payload.get("detector_id") or ""),
            geometry_id=str(payload.get("geometry_id") or ""),
            position_mm=_optional_float(payload.get("position_mm")),
            line_energy_keV=float(payload.get("line_energy_keV", 0.0) or 0.0),
            line_id=str(payload.get("line_id") or ""),
            assigned_radionuclide=payload.get("assigned_radionuclide"),
            net_peak_area=float(payload.get("net_peak_area", 0.0) or 0.0),
            area_uncertainty=float(payload.get("area_uncertainty", 0.0) or 0.0),
            live_time_s=float(payload.get("live_time_s", 0.0) or 0.0),
            real_time_s=float(payload.get("real_time_s", 0.0) or 0.0),
            count_start_time=payload.get("count_start_time"),
            reference_time=payload.get("reference_time"),
            irradiation_time_s=float(payload.get("irradiation_time_s", 0.0) or 0.0),
            decay_time_s=float(payload.get("decay_time_s", 0.0) or 0.0),
            counting_time_s=float(payload.get("counting_time_s", 0.0) or 0.0),
            dead_time_correction_method=str(payload.get("dead_time_correction_method") or "live_time_real_time"),
            baseline_method=str(payload.get("baseline_method") or "unspecified"),
            deconvolution_status=str(payload.get("deconvolution_status") or "not_evaluated"),
            interference_flags=tuple(str(item) for item in (payload.get("interference_flags") or [])),
            analyst_review_status=str(payload.get("analyst_review_status") or "unreviewed"),
            peak_area_provenance=str(payload.get("peak_area_provenance") or "imported_peak_table"),
            import_format=str(payload.get("import_format") or "peak_report"),
            peak_class=str(payload.get("peak_class") or "photopeak"),
            reaction_family=str(payload.get("reaction_family") or "thermal_capture"),
            gamma_yield=float(payload.get("gamma_yield", 1.0) or 1.0),
            efficiency=_optional_float(payload.get("efficiency")),
            efficiency_uncertainty=_optional_float(payload.get("efficiency_uncertainty")),
            emission_probability=_optional_float(payload.get("emission_probability")),
            sample_role=str(payload.get("sample_role") or "sample"),
            sample_mass_g=_optional_float(payload.get("sample_mass_g")),
            project_id=payload.get("project_id"),
            sample_id=payload.get("sample_id"),
            irradiation_id=payload.get("irradiation_id"),
            measurement_id=payload.get("measurement_id"),
            g_th=float(payload.get("g_th", 1.0) or 1.0),
            g_ep=float(payload.get("g_ep", 1.0) or 1.0),
            cd_factor=float(payload.get("cd_factor", 1.0) or 1.0),
            eligibility_class=str(payload.get("eligibility_class") or "direct_k0_eligible"),
            eligibility_accepted=bool(payload.get("eligibility_accepted", True)),
            rejection_reasons=tuple(str(item) for item in (payload.get("rejection_reasons") or [])),
            expert_override=bool(payload.get("expert_override", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        text = _optional_text(value)
        if text:
            return text
    return None


def resolve_governed_libraries(
    *,
    k0_library_file: str | Path | None = None,
    auxiliary_library_file: str | Path | None = None,
) -> tuple[GovernedLibrary, GovernedLibrary]:
    """Resolve the selected standard and auxiliary k0 libraries."""

    standard_library = load_governed_library(k0_library_file, library_kind="standard") if k0_library_file else get_active_k0_library()
    auxiliary_library = (
        load_governed_library(auxiliary_library_file, library_kind="auxiliary")
        if auxiliary_library_file
        else get_active_auxiliary_correction_library()
    )
    return standard_library, auxiliary_library


def classify_peak_observation(
    peak_class: str,
    *,
    gamma_yield: float,
    reaction_family: str,
    interference_flags: Iterable[str] = (),
    expert_override: bool = False,
    allow_advanced: bool = False,
) -> tuple[str, bool, list[str]]:
    """Return `(eligibility_class, accepted, reasons)` for one observation."""

    normalized_peak_class = str(peak_class or "photopeak").strip().lower()
    normalized_reaction = str(reaction_family or "thermal_capture").strip().lower()
    flags = [str(item) for item in interference_flags if str(item)]
    reasons: list[str] = []

    if gamma_yield <= 0.0:
        reasons.append("gamma_yield_non_positive")
        eligibility = "ineligible_no_gamma_yield"
        accepted = False
    elif normalized_peak_class in {"single_escape", "double_escape", "escape_peak"}:
        reasons.append("escape_peaks_are_not_base_k0_lines")
        eligibility = "escape_peak"
        accepted = False
    elif normalized_peak_class in {"sum_peak", "pure_sum_peak", "coincidence_sum"}:
        reasons.append("sum_peaks_are_not_base_k0_lines")
        eligibility = "sum_peak"
        accepted = False
    elif normalized_reaction not in {"thermal_capture", "thermal_fission", "(n,g)", "(n,f)"}:
        reasons.append(f"unsupported_reaction_family:{normalized_reaction}")
        eligibility = "unsupported_reaction_family"
        accepted = False
    elif any(flag.lower() in {"threshold", "threshold_interference", "fast_flux"} for flag in flags):
        reasons.append("threshold_or_fast_flux_sensitive")
        eligibility = "threshold_interference_sensitive"
        accepted = False
    elif flags:
        reasons.append("interference_flags_present")
        eligibility = "interference_affected"
        accepted = False
    else:
        eligibility = "direct_k0_eligible"
        accepted = True

    if not accepted and (expert_override or allow_advanced):
        reasons.append("expert_override_applied" if expert_override else "advanced_line_allowed")
        accepted = True
    return eligibility, accepted, reasons


def evaluate_detector_characterization(
    detector_payload: Dict[str, Any],
    energy_keV: float,
    *,
    position_mm: float | None = None,
) -> tuple[float | None, float | None, str]:
    """Evaluate detector efficiency and conversion uncertainty from an artifact."""

    model = detector_payload.get("efficiency_model") or {}
    coefficients = model.get("coefficients") or []
    energy_range = tuple(model.get("energy_range_keV") or (0.0, 1.0e6))
    if not coefficients:
        return None, None, "missing_model"
    curve = EfficiencyCurve.from_polynomial(
        coefficients=list(coefficients),
        energy_range=energy_range,
        detector_id=str(detector_payload.get("detector_id") or ""),
    )
    efficiency = float(curve.efficiency(float(energy_keV)))
    uncertainty = None
    method = "direct_position_calibration"
    reference_position = _optional_float(detector_payload.get("reference_position_mm"))
    conversions = (detector_payload.get("geometry_conversions") or {}).get("items") or {}
    if position_mm is not None and reference_position is not None and abs(position_mm - reference_position) > 1e-9:
        key = f"{position_mm:.6g}"
        conversion = conversions.get(key)
        if conversion is not None:
            efficiency *= float(conversion.get("ratio_mean", 1.0) or 1.0)
            uncertainty = float(conversion.get("ratio_std", 0.0) or 0.0)
            method = str(conversion.get("method") or "empirical_reference_conversion")
        else:
            method = "geometric_fallback"
            scale = (reference_position / position_mm) ** 2 if position_mm > 0 and reference_position > 0 else 1.0
            efficiency *= scale
            uncertainty = 0.15
    return efficiency, uncertainty, method


def peak_report_to_observations(
    peak_payload: Dict[str, Any],
    *,
    spectrum_payload: Dict[str, Any] | None = None,
    detector_payload: Dict[str, Any] | None = None,
    detector_id: str | None = None,
    geometry_id: str | None = None,
    irradiation_time_s: float = 0.0,
    decay_time_s: float = 0.0,
    counting_time_s: float | None = None,
    default_dead_time_correction_method: str = "live_time_real_time",
    baseline_method: str = "unspecified",
    analyst_review_status: str = "unreviewed",
    peak_area_provenance: str = "imported_peak_table",
    import_format: str = "peak_report",
    project_id: str | None = None,
    sample_id: str | None = None,
    irradiation_id: str | None = None,
    measurement_id: str | None = None,
    expert_override: bool = False,
    allow_advanced: bool = False,
) -> list[PeakObservation]:
    """Normalize a peak-report artifact into k0-ready peak observations."""

    spectrum = spectrum_payload.get("spectrum", {}) if spectrum_payload else {}
    live_time_s = float(peak_payload.get("live_time_s") or spectrum.get("live_time") or 0.0)
    real_time_s = float(spectrum.get("real_time") or live_time_s)
    count_start_time = spectrum.get("start_time")
    source_spectrum_id = str(peak_payload.get("spectrum_id") or spectrum.get("spectrum_id") or "")
    detector_id = str(detector_id or spectrum.get("detector_id") or "")
    geometry_id = str(geometry_id or spectrum.get("metadata", {}).get("geometry_id") or "")
    spectrum_metadata = spectrum.get("metadata", {}) or {}
    observations: list[PeakObservation] = []

    for index, row in enumerate(peak_payload.get("peaks", []) or []):
        energy_keV = float(row.get("analysis_peak_energy_keV") or row.get("energy_keV") or 0.0)
        net_area = float(row.get("net_counts") or row.get("area") or row.get("raw_counts") or 0.0)
        area_uncertainty = float(row.get("net_counts_unc") or max(np.sqrt(max(net_area, 0.0)), 0.0))
        assigned = row.get("assigned_radionuclide") or row.get("report_isotope") or identify_isotope_from_gamma(energy_keV)
        peak_class = str(row.get("peak_type") or row.get("peak_class") or "photopeak")
        flags: list[str] = []
        if row.get("interference_flags"):
            flags.extend(str(item) for item in row.get("interference_flags") or [])
        if row.get("threshold_interference_sensitive"):
            flags.append("threshold_interference")
        reaction_family = str(row.get("reaction_family") or "thermal_capture")
        gamma_yield = float(row.get("gamma_yield") or row.get("emission_probability") or 1.0)
        eligibility_class, accepted, reasons = classify_peak_observation(
            peak_class,
            gamma_yield=gamma_yield,
            reaction_family=reaction_family,
            interference_flags=flags,
            expert_override=expert_override,
            allow_advanced=allow_advanced,
        )
        position_mm = _optional_float(row.get("position_mm") or spectrum.get("metadata", {}).get("position_mm"))
        efficiency = _optional_float(row.get("efficiency"))
        efficiency_uncertainty = _optional_float(row.get("efficiency_uncertainty"))
        metadata = {
            "label": row.get("label"),
            "channel": row.get("channel"),
            "region": row.get("region"),
            "background_subtracted": row.get("background_subtracted"),
        }
        if detector_payload is not None and efficiency is None:
            efficiency, efficiency_uncertainty, conversion_method = evaluate_detector_characterization(
                detector_payload,
                energy_keV,
                position_mm=position_mm,
            )
            metadata["geometry_conversion_method"] = conversion_method
        observation = PeakObservation(
            peak_id=str(row.get("label") or row.get("line_id") or f"peak_{index + 1}"),
            source_spectrum_id=source_spectrum_id,
            detector_id=detector_id,
            geometry_id=geometry_id,
            position_mm=position_mm,
            line_energy_keV=energy_keV,
            line_id=str(row.get("line_id") or f"{assigned or 'unknown'}@{energy_keV:.3f}keV"),
            assigned_radionuclide=str(assigned) if assigned else None,
            net_peak_area=net_area,
            area_uncertainty=area_uncertainty,
            live_time_s=live_time_s,
            real_time_s=real_time_s,
            count_start_time=count_start_time,
            reference_time=spectrum.get("metadata", {}).get("reference_time"),
            irradiation_time_s=float(row.get("irradiation_time_s") or irradiation_time_s or 0.0),
            decay_time_s=float(row.get("decay_time_s") or decay_time_s or 0.0),
            counting_time_s=float(row.get("counting_time_s") or counting_time_s or live_time_s or 0.0),
            dead_time_correction_method=str(row.get("dead_time_correction_method") or default_dead_time_correction_method),
            baseline_method=str(row.get("baseline_method") or baseline_method),
            deconvolution_status=str(row.get("deconvolution_status") or ("manual_roi" if row.get("manual") else "not_evaluated")),
            interference_flags=tuple(flags),
            analyst_review_status=str(row.get("analyst_review_status") or analyst_review_status),
            peak_area_provenance=str(row.get("peak_area_provenance") or peak_area_provenance),
            import_format=str(row.get("import_format") or import_format),
            peak_class=peak_class,
            reaction_family=reaction_family,
            gamma_yield=gamma_yield,
            efficiency=efficiency,
            efficiency_uncertainty=efficiency_uncertainty,
            emission_probability=_optional_float(row.get("emission_probability")) or gamma_yield,
            sample_role=str(row.get("sample_role") or "sample"),
            sample_mass_g=_optional_float(row.get("sample_mass_g")),
            project_id=_first_nonempty(row.get("project_id"), project_id, peak_payload.get("project_id"), spectrum_metadata.get("project_id")),
            sample_id=_first_nonempty(row.get("sample_id"), sample_id, peak_payload.get("sample_id"), spectrum_metadata.get("sample_id"), source_spectrum_id),
            irradiation_id=_first_nonempty(row.get("irradiation_id"), irradiation_id, peak_payload.get("irradiation_id"), spectrum_metadata.get("irradiation_id")),
            measurement_id=_first_nonempty(row.get("measurement_id"), measurement_id, peak_payload.get("measurement_id"), spectrum_metadata.get("measurement_id"), f"{source_spectrum_id}:{index + 1}"),
            g_th=float(row.get("g_th", 1.0) or 1.0),
            g_ep=float(row.get("g_ep", 1.0) or 1.0),
            cd_factor=float(row.get("cd_factor", 1.0) or 1.0),
            eligibility_class=eligibility_class,
            eligibility_accepted=accepted,
            rejection_reasons=tuple(reasons),
            expert_override=expert_override and bool(reasons),
            metadata=metadata,
        )
        observations.append(observation)
    return observations


def build_detector_characterization(
    point_rows: Iterable[Dict[str, Any]],
    *,
    detector_id: str,
    reference_position_mm: float,
    degree: int = 2,
    peak_to_total_ratio: float | None = None,
    coincidence_mode: str = "not_applied",
) -> Dict[str, Any]:
    """Fit a reusable detector-characterization artifact from calibration points."""

    rows = [dict(row) for row in point_rows]
    grouped: Dict[float, list[EfficiencyPoint]] = {}
    point_payloads: list[Dict[str, Any]] = []
    for row in rows:
        position_mm = float(row.get("position_mm", reference_position_mm) or reference_position_mm)
        point = EfficiencyPoint(
            energy_keV=float(row["reference_energy_keV"]),
            net_counts=float(row["net_counts"]),
            live_time_s=float(row["live_time_s"]),
            activity_bq=float(row["activity_bq"]),
            emission_probability=float(row["emission_probability"]),
            geometry_factor=float(row.get("geometry_factor", 1.0) or 1.0),
            count_uncertainty=_optional_float(row.get("count_uncertainty")),
            activity_rel_unc=_optional_float(row.get("activity_rel_unc")),
            probability_uncertainty=_optional_float(row.get("probability_uncertainty")),
        )
        grouped.setdefault(position_mm, []).append(point)
        eff, unc = point.efficiency()
        point_payloads.append(
            {
                "position_mm": position_mm,
                "source_name": row.get("source_name", ""),
                "reference_energy_keV": point.energy_keV,
                "efficiency": eff,
                "efficiency_uncertainty": unc,
            }
        )

    reference_points = grouped.get(reference_position_mm)
    if not reference_points:
        raise ValueError("Reference-position calibration points are required for detector characterization.")
    effective_degree = max(0, min(int(degree), len(reference_points) - 2))
    fit = fit_efficiency_curve(reference_points, degree=effective_degree, detector_id=detector_id)

    conversions: Dict[str, Dict[str, Any]] = {}
    reference_by_energy = {round(point.energy_keV, 3): point.efficiency()[0] for point in reference_points}
    for position_mm, points in grouped.items():
        if abs(position_mm - reference_position_mm) < 1e-9:
            continue
        ratios: list[float] = []
        for point in points:
            key = round(point.energy_keV, 3)
            ref_eff = reference_by_energy.get(key)
            eff, _ = point.efficiency()
            if ref_eff and ref_eff > 0.0:
                ratios.append(eff / ref_eff)
        if ratios:
            conversions[f"{position_mm:.6g}"] = {
                "method": "empirical_reference_conversion",
                "reference_position_mm": reference_position_mm,
                "target_position_mm": position_mm,
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)) if len(ratios) > 1 else 0.0,
                "n_pairs": len(ratios),
            }

    return {
        "detector_id": detector_id,
        "reference_position_mm": reference_position_mm,
        "characterized_positions_mm": sorted(grouped),
        "calibration_points": point_payloads,
        "efficiency_model": {
            "model_type": "log_poly",
            "coefficients": list(fit.coefficients),
            "fit_degree": effective_degree,
            "energy_range_keV": list(fit.curve.energy_range),
            "residuals": [float(value) for value in np.asarray(fit.residuals, dtype=float)],
        },
        "geometry_conversions": {
            "method": "reference_position_models",
            "items": conversions,
        },
        "peak_to_total_model": {
            "method": "constant_ratio" if peak_to_total_ratio is not None else "not_characterized",
            "value": peak_to_total_ratio,
        },
        "coincidence_model": {
            "mode": coincidence_mode,
            "applied": False,
        },
        "capability_flags": CAPABILITY_FLAGS,
    }


def build_facility_characterization(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a governed facility-characterization artifact."""

    method = str(payload.get("method") or "bare_triple_monitor").strip().lower()
    irradiation = payload.get("irradiation", {}) or {}
    monitors = payload.get("monitors", []) or []
    irradiation_params = TRIGAIrradiationParams(
        irradiation_time_s=float(irradiation.get("irradiation_time_s", 0.0) or 0.0),
        decay_time_s=float(irradiation.get("decay_time_s", 0.0) or 0.0),
        counting_time_s=float(irradiation.get("counting_time_s", 0.0) or 0.0),
        live_time_s=_optional_float(irradiation.get("live_time_s")),
        dead_time_fraction=float(irradiation.get("dead_time_fraction", 0.0) or 0.0),
        reactor_power_kW=float(irradiation.get("reactor_power_kW", 1000.0) or 1000.0),
        position=str(irradiation.get("position") or ""),
    )
    temperature = payload.get("temperature", {}) or {}
    gradients = dict(payload.get("gradients") or {"supported": False})
    fast_flux = dict(payload.get("fast_flux") or {"status": "not_characterized"})

    if method == "bare_triple_monitor":
        activities = {str(row["monitor_id"]): float(row["activity"] or 0.0) for row in monitors}
        triple = triple_monitor_method(activities, irradiation_params)
        flux_parameters = {
            "f": triple.f,
            "f_uncertainty": triple.f_uncertainty,
            "alpha": triple.alpha,
            "alpha_uncertainty": triple.alpha_uncertainty,
            "phi_thermal": triple.phi_thermal,
            "phi_epithermal": triple.phi_epithermal,
            "phi_fast": triple.phi_fast,
            "monitors_used": list(triple.monitors_used),
            "iterations": triple.convergence_iterations,
        }
        facility_method = "bare_triple_monitor"
    elif method in {"cd_ratio_multi_monitor", "cadmium_ratio_multi_monitor", "cd_ratio"}:
        workflow = TRIGAk0Workflow(
            position=irradiation_params.position,
            reactor_power_kW=irradiation_params.reactor_power_kW,
        )
        workflow.set_irradiation_params(
            t_irr_s=irradiation_params.irradiation_time_s,
            t_decay_s=irradiation_params.decay_time_s,
            t_count_s=irradiation_params.counting_time_s,
            dead_time_fraction=irradiation_params.dead_time_fraction,
        )
        bare_activities = {
            str(row.get("element") or row.get("monitor_id") or row.get("monitor")): float(row.get("activity_bare") or row.get("activity") or 0.0)
            for row in monitors
            if row.get("activity_bare") is not None or row.get("activity") is not None
        }
        cd_activities = {
            str(row.get("element") or row.get("monitor_id") or row.get("monitor")): float(row.get("activity_cd") or 0.0)
            for row in monitors
            if row.get("activity_cd") is not None
        }
        uncertainties_bare = {
            str(row.get("element") or row.get("monitor_id") or row.get("monitor")): float(row.get("uncertainty_bare") or row.get("uncertainty") or 0.05)
            for row in monitors
        }
        uncertainties_cd = {
            str(row.get("element") or row.get("monitor_id") or row.get("monitor")): float(row.get("uncertainty_cd") or 0.07)
            for row in monitors
            if row.get("activity_cd") is not None
        }
        cd_flux = workflow.characterize_flux_cd_ratio(bare_activities, cd_activities, uncertainties_bare, uncertainties_cd)
        flux_parameters = {
            "f": cd_flux.f,
            "f_uncertainty": cd_flux.f_uncertainty,
            "alpha": cd_flux.alpha,
            "alpha_uncertainty": cd_flux.alpha_uncertainty,
            "phi_thermal": cd_flux.phi_thermal,
            "phi_epithermal": cd_flux.phi_epithermal,
            "phi_fast": float((payload.get("flux_parameters") or {}).get("phi_fast", 0.0) or 0.0),
            "monitors_used": sorted(set(bare_activities) & set(cd_activities)),
            "cd_covered": True,
        }
        fast_flux.setdefault("status", "tracked_not_solved")
        facility_method = "cadmium_ratio_multi_monitor"
    elif method in {"single_monitor_known_flux", "known_flux_parameters", "single_monitor"}:
        known = payload.get("known_flux_parameters") or payload.get("flux_parameters") or {}
        monitor_ids = [str(row.get("monitor_id") or row.get("element") or row.get("monitor") or "monitor") for row in monitors]
        flux_parameters = {
            "f": float(known.get("f", 0.0) or 0.0),
            "f_uncertainty": float(known.get("f_uncertainty", 0.0) or 0.0),
            "alpha": float(known.get("alpha", 0.0) or 0.0),
            "alpha_uncertainty": float(known.get("alpha_uncertainty", 0.0) or 0.0),
            "phi_thermal": _optional_float(known.get("phi_thermal")),
            "phi_epithermal": _optional_float(known.get("phi_epithermal")),
            "phi_fast": float(known.get("phi_fast", 0.0) or 0.0),
            "monitors_used": monitor_ids,
            "reference_characterization_id": known.get("reference_characterization_id"),
            "workflow_note": "Single-monitor mode uses previously characterized f and alpha rather than solving an underdetermined one-monitor field characterization.",
        }
        facility_method = "single_monitor_known_flux"
        fast_flux.setdefault("status", "tracked_not_solved")
    else:
        raise ValueError(f"Unsupported facility characterization method: {method}")

    return {
        "facility_id": str(payload.get("facility_id") or irradiation_params.position or "facility"),
        "method": facility_method,
        "monitor_definitions": [dict(row) for row in monitors],
        "irradiation": {
            "irradiation_time_s": irradiation_params.irradiation_time_s,
            "decay_time_s": irradiation_params.decay_time_s,
            "counting_time_s": irradiation_params.counting_time_s,
            "live_time_s": irradiation_params.live_time_s,
            "dead_time_fraction": irradiation_params.dead_time_fraction,
            "reactor_power_kW": irradiation_params.reactor_power_kW,
            "position": irradiation_params.position,
        },
        "flux_parameters": flux_parameters,
        "temperature": {
            "value_K": float(temperature.get("value_K", 300.0) or 300.0),
            "method": str(temperature.get("method") or "assumed_cooling_water"),
        },
        "gradients": gradients,
        "fast_flux": fast_flux,
        "capability_flags": CAPABILITY_FLAGS,
    }


def _observation_to_measurement(
    observation: PeakObservation,
    *,
    sample_mass_g: float,
) -> K0Measurement:
    peak_like = {
        "energy": observation.line_energy_keV,
        "net_area": observation.net_peak_area,
        "area_unc": observation.area_uncertainty,
    }
    measurement = create_k0_measurement_from_peak(
        peak_like,
        efficiency=float(observation.efficiency or 0.0),
        t_irr=observation.irradiation_time_s,
        t_decay=observation.decay_time_s,
        t_count=observation.counting_time_s,
        sample_mass=sample_mass_g,
        g_th=observation.g_th,
        g_ep=observation.g_ep,
        cd_factor=observation.cd_factor,
        tolerance_keV=2.0,
    )
    if measurement is None and observation.assigned_radionuclide:
        measurement = K0Measurement(
            product_isotope=observation.assigned_radionuclide,
            net_peak_area=observation.net_peak_area,
            peak_area_unc=observation.area_uncertainty,
            efficiency=float(observation.efficiency or 0.0),
            efficiency_unc=float(observation.efficiency_uncertainty or 0.0),
            t_irr=observation.irradiation_time_s,
            t_decay=observation.decay_time_s,
            t_count=observation.counting_time_s,
            sample_mass=sample_mass_g,
            gamma_energy_keV=observation.line_energy_keV,
            g_th=observation.g_th,
            g_ep=observation.g_ep,
            cd_factor=observation.cd_factor,
        )
    if measurement is None:
        raise ValueError(f"Could not create K0Measurement for observation {observation.peak_id}")
    return measurement


def analyze_k0_observations(
    observation_payload: Dict[str, Any],
    facility_payload: Dict[str, Any],
    *,
    sample_mass_g: float,
    reference_isotope: str = "Au-198",
    reference_mass_g: float | None = None,
    standard_library: GovernedLibrary | None = None,
    auxiliary_library: GovernedLibrary | None = None,
) -> Dict[str, Any]:
    """Run a first-pass k0 analysis bundle from normalized observations."""

    standard_library = standard_library or get_active_k0_library()
    auxiliary_library = auxiliary_library or get_active_auxiliary_correction_library()
    try:
        with use_governed_libraries(standard_library=standard_library, auxiliary_library=auxiliary_library):
            set_k0_database_from_governed_library(get_active_k0_library())

            observations = [PeakObservation.from_dict(item) for item in (observation_payload.get("observations") or [])]
            flux = facility_payload.get("flux_parameters", {}) or {}
            temperature = float((facility_payload.get("temperature") or {}).get("value_K", 293.6) or 293.6)
            parameters = K0Parameters(
                f=float(flux.get("f", 0.0) or 0.0),
                alpha=float(flux.get("alpha", 0.0) or 0.0),
                f_uncertainty=float(flux.get("f_uncertainty", 0.0) or 0.0),
                alpha_uncertainty=float(flux.get("alpha_uncertainty", 0.0) or 0.0),
                phi_thermal=float(flux.get("phi_thermal", 0.0) or 0.0),
                phi_epithermal=float(flux.get("phi_epithermal", 0.0) or 0.0),
                phi_fast=float(flux.get("phi_fast", 0.0) or 0.0),
            )

            reference_candidates = [
                item
                for item in observations
                if item.eligibility_accepted and item.assigned_radionuclide == reference_isotope
            ]
            if not reference_candidates:
                raise ValueError(f"No eligible reference observation found for {reference_isotope}")
            reference = reference_candidates[0]
            reference_mass = float(reference_mass_g or reference.sample_mass_g or sample_mass_g)
            calculator = K0Calculator(parameters, _observation_to_measurement(reference, sample_mass_g=reference_mass))

            line_results: list[Dict[str, Any]] = []
            rejected: list[Dict[str, Any]] = []
            weighted_by_element: Dict[str, list[tuple[float, float, str, PeakObservation]]] = {}
            recognized_not_applied: list[str] = []
            applied_corrections = [
                "saturation_decay_counting",
                "q0_alpha",
                "relative_au_monitor_formalism",
            ]

            for observation in observations:
                if observation.peak_id == reference.peak_id:
                    continue
                if not observation.eligibility_accepted:
                    rejected.append(observation.to_dict())
                    continue
                if observation.efficiency is None or observation.efficiency <= 0.0:
                    rejected.append({**observation.to_dict(), "rejection_reasons": list(observation.rejection_reasons) + ["missing_efficiency"]})
                    continue
                measurement = _observation_to_measurement(observation, sample_mass_g=float(observation.sample_mass_g or sample_mass_g))
                result = calculator.calculate_concentration(measurement)
                record = get_k0_library_record(result.product_isotope)
                g_t = get_westcott_g(record.target_isotope if record else result.product_isotope, temperature)
                if abs(g_t - 1.0) > 1e-6:
                    recognized_not_applied.append(f"westcott_gT_not_applied:{record.target_isotope if record else result.product_isotope}")
                budget = create_k0_naa_budget(
                    concentration=result.concentration_ug_g,
                    counting_rel=(measurement.peak_area_unc / measurement.net_peak_area) if measurement.net_peak_area else 0.0,
                    efficiency_rel=float(observation.efficiency_uncertainty or 0.0),
                    k0_rel=float((record.k0_unc_percent if record else 0.0) / 100.0),
                    q0_rel=float((record.Q0_unc_percent if record else 0.0) / 100.0),
                    f_rel=(parameters.f_uncertainty / parameters.f) if parameters.f else 0.0,
                    alpha_abs=parameters.alpha_uncertainty,
                    timing_rel=0.005,
                    coincidence_rel=0.0,
                    units="ug/g",
                )
                line_row = {
                    "peak_id": observation.peak_id,
                    "line_id": observation.line_id,
                    "line_energy_keV": observation.line_energy_keV,
                    "assigned_radionuclide": observation.assigned_radionuclide,
                    "element": result.element,
                    "project_id": observation.project_id,
                    "sample_id": observation.sample_id,
                    "irradiation_id": observation.irradiation_id,
                    "measurement_id": observation.measurement_id,
                    "concentration_ug_g": result.concentration_ug_g,
                    "concentration_unc_ug_g": result.concentration_unc,
                    "detection_limit_ug_g": result.detection_limit_ug_g,
                    "k0_used": result.k0_used,
                    "Q0_alpha_used": result.Q0_alpha_used,
                    "sdc_factor": result.sdc_factor,
                    "specific_count_rate": result.specific_count_rate,
                    "eligibility_class": observation.eligibility_class,
                    "uncertainty_budget": budget.to_dict(),
                }
                line_results.append(line_row)
                weighted_by_element.setdefault(result.element, []).append((result.concentration_ug_g, max(result.concentration_unc, 1e-12), observation.line_id, observation))

            element_results: list[Dict[str, Any]] = []
            for element, rows in weighted_by_element.items():
                weights = np.array([1.0 / (unc ** 2) for _, unc, _, _ in rows], dtype=float)
                values = np.array([value for value, _, _, _ in rows], dtype=float)
                combined = float(np.sum(values * weights) / np.sum(weights)) if np.sum(weights) else 0.0
                combined_unc = float(1.0 / np.sqrt(np.sum(weights))) if np.sum(weights) else 0.0
                row_observations = [item[3] for item in rows]
                element_results.append(
                    {
                        "element": element,
                        "project_id": _first_nonempty(*(item.project_id for item in row_observations)),
                        "sample_id": _first_nonempty(*(item.sample_id for item in row_observations)),
                        "irradiation_ids": sorted({item.irradiation_id for item in row_observations if item.irradiation_id}),
                        "measurement_ids": sorted({item.measurement_id for item in row_observations if item.measurement_id}),
                        "concentration_ug_g": combined,
                        "concentration_unc_ug_g": combined_unc,
                        "line_ids": [line_id for _, _, line_id, _ in rows],
                        "combination_method": "inverse_variance_weighted",
                    }
                )

            seen = set()
            deduped_recognized = []
            for item in recognized_not_applied:
                if item not in seen:
                    seen.add(item)
                    deduped_recognized.append(item)

            default_assumptions = [
                "thermal_workflow_scope",
                f"facility_characterization_method:{facility_payload.get('method', 'unknown')}",
            ]
            if str(facility_payload.get("method") or "") == "single_monitor_known_flux":
                default_assumptions.append("single_monitor_prior_f_alpha_required")

            project_id = _first_nonempty(*(item.project_id for item in observations))
            sample_id = _first_nonempty(*(item.sample_id for item in observations))
            irradiation_ids = sorted({item.irradiation_id for item in observations if item.irradiation_id})
            measurement_ids = sorted({item.measurement_id for item in observations if item.measurement_id})

            return {
                "summary": {
                    "line_result_count": len(line_results),
                    "rejected_observation_count": len(rejected),
                    "element_count": len(element_results),
                    "reference_isotope": reference_isotope,
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "irradiation_count": len(irradiation_ids),
                    "measurement_count": len(measurement_ids),
                },
                "line_results": line_results,
                "element_results": element_results,
                "rejected_observations": rejected,
                "applied_corrections": applied_corrections,
                "recognized_but_not_applied": deduped_recognized,
                "user_supplied_corrections": [],
                "default_assumptions": default_assumptions,
                "capability_flags": CAPABILITY_FLAGS,
                "libraries": get_library_summary(),
                "inputs": {
                    "observation_bundle_schema": observation_payload.get("schema"),
                    "facility_characterization_schema": facility_payload.get("schema"),
                    "project_id": project_id,
                    "sample_id": sample_id,
                    "irradiation_ids": irradiation_ids,
                    "measurement_ids": measurement_ids,
                },
            }
    finally:
        reset_k0_database()


def aggregate_k0_analysis_bundles(
    analysis_payloads: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate k0 analysis bundles across measurements and irradiations."""

    groups: Dict[tuple[str | None, str | None, str], Dict[str, Any]] = {}
    by_irradiation: Dict[tuple[str | None, str | None, str, str], list[tuple[float, float]]] = defaultdict(list)
    bundle_count = 0
    for payload in analysis_payloads:
        bundle_count += 1
        for row in payload.get("element_results") or []:
            key = (_optional_text(row.get("project_id") or (payload.get("summary") or {}).get("project_id")), _optional_text(row.get("sample_id") or (payload.get("summary") or {}).get("sample_id")), str(row.get("element") or "unknown"))
            group = groups.setdefault(
                key,
                {
                    "project_id": key[0],
                    "sample_id": key[1],
                    "element": key[2],
                    "values": [],
                    "measurement_ids": set(),
                    "irradiation_ids": set(),
                    "line_ids": set(),
                    "bundle_count": 0,
                },
            )
            value = float(row.get("concentration_ug_g", 0.0) or 0.0)
            unc = max(float(row.get("concentration_unc_ug_g", 0.0) or 0.0), 1e-12)
            group["values"].append((value, unc))
            group["bundle_count"] += 1
            group["measurement_ids"].update(row.get("measurement_ids") or [])
            group["irradiation_ids"].update(row.get("irradiation_ids") or [])
            group["line_ids"].update(row.get("line_ids") or [])
            for irradiation_id in row.get("irradiation_ids") or ["unspecified"]:
                by_irradiation[(key[0], key[1], key[2], str(irradiation_id))].append((value, unc))

    aggregated_results: list[Dict[str, Any]] = []
    irradiation_summaries: list[Dict[str, Any]] = []
    for (project_id, sample_id, element), group in sorted(groups.items(), key=lambda item: (item[0][0] or "", item[0][1] or "", item[0][2])):
        values = np.array([value for value, _ in group["values"]], dtype=float)
        weights = np.array([1.0 / (unc ** 2) for _, unc in group["values"]], dtype=float)
        combined = float(np.sum(values * weights) / np.sum(weights)) if np.sum(weights) else 0.0
        combined_unc = float(1.0 / np.sqrt(np.sum(weights))) if np.sum(weights) else 0.0
        aggregated_results.append(
            {
                "project_id": project_id,
                "sample_id": sample_id,
                "element": element,
                "concentration_ug_g": combined,
                "concentration_unc_ug_g": combined_unc,
                "measurement_ids": sorted(group["measurement_ids"]),
                "irradiation_ids": sorted(group["irradiation_ids"]),
                "line_ids": sorted(group["line_ids"]),
                "bundle_count": group["bundle_count"],
                "combination_method": "inverse_variance_weighted",
            }
        )

    for (project_id, sample_id, element, irradiation_id), values in sorted(by_irradiation.items(), key=lambda item: (item[0][0] or "", item[0][1] or "", item[0][2], item[0][3])):
        weights = np.array([1.0 / (unc ** 2) for _, unc in values], dtype=float)
        concentrations = np.array([value for value, _ in values], dtype=float)
        irradiation_summaries.append(
            {
                "project_id": project_id,
                "sample_id": sample_id,
                "element": element,
                "irradiation_id": irradiation_id,
                "concentration_ug_g": float(np.sum(concentrations * weights) / np.sum(weights)) if np.sum(weights) else 0.0,
                "concentration_unc_ug_g": float(1.0 / np.sqrt(np.sum(weights))) if np.sum(weights) else 0.0,
            }
        )

    return {
        "summary": {
            "source_bundle_count": bundle_count,
            "aggregated_result_count": len(aggregated_results),
            "irradiation_summary_count": len(irradiation_summaries),
            "sample_count": len({(row.get('project_id'), row.get('sample_id')) for row in aggregated_results}),
        },
        "aggregated_results": aggregated_results,
        "irradiation_summaries": irradiation_summaries,
    }


def evaluate_k0_qaqc(
    records: Iterable[Dict[str, Any]],
    *,
    default_blank_limit_ug_g: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate blank and CRM QA/QC decisions from k0 analysis bundles."""

    result_records: list[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    for record in records:
        role = str(record.get("role") or "sample").strip().lower()
        analysis_payload = record.get("analysis_payload") or {}
        element_results = analysis_payload.get("element_results") or []
        sample_id = _first_nonempty(record.get("sample_id"), (analysis_payload.get("summary") or {}).get("sample_id"))

        if role == "blank":
            limits = dict(record.get("element_limits_ug_g") or {})
            exceedances = []
            for row in element_results:
                limit = float(limits.get(row.get("element"), record.get("default_limit_ug_g", default_blank_limit_ug_g)) or 0.0)
                concentration = float(row.get("concentration_ug_g", 0.0) or 0.0)
                if concentration > limit:
                    exceedances.append(
                        {
                            "element": row.get("element"),
                            "concentration_ug_g": concentration,
                            "limit_ug_g": limit,
                        }
                    )
            status = "pass" if not exceedances else "fail"
            result_records.append(
                {
                    "role": "blank",
                    "sample_id": sample_id,
                    "status": status,
                    "exceedances": exceedances,
                }
            )
        elif role == "crm":
            certified_values = dict(record.get("certified_values") or {})
            comparisons = []
            for row in element_results:
                if row.get("element") not in certified_values:
                    continue
                reference = certified_values[row["element"]]
                target_value = float(reference.get("value_ug_g", 0.0) or 0.0)
                target_unc = float(reference.get("unc_ug_g", 0.0) or 0.0)
                measured = float(row.get("concentration_ug_g", 0.0) or 0.0)
                measured_unc = float(row.get("concentration_unc_ug_g", 0.0) or 0.0)
                combined_unc = float(np.sqrt(measured_unc ** 2 + target_unc ** 2)) if (measured_unc or target_unc) else 0.0
                bias = measured - target_value
                rel_bias = (bias / target_value * 100.0) if target_value else 0.0
                en_score = (bias / combined_unc) if combined_unc else 0.0
                comparisons.append(
                    {
                        "element": row.get("element"),
                        "measured_ug_g": measured,
                        "measured_unc_ug_g": measured_unc,
                        "certified_ug_g": target_value,
                        "certified_unc_ug_g": target_unc,
                        "bias_ug_g": bias,
                        "relative_bias_percent": rel_bias,
                        "en_score": en_score,
                        "status": "pass" if abs(en_score) <= float(record.get("acceptance_en_limit", 2.0) or 2.0) else "fail",
                    }
                )
            status = "pass" if comparisons and all(item["status"] == "pass" for item in comparisons) else "fail"
            result_records.append(
                {
                    "role": "crm",
                    "sample_id": sample_id,
                    "status": status,
                    "comparisons": comparisons,
                }
            )
        else:
            continue

        if result_records[-1]["status"] == "pass":
            pass_count += 1
        else:
            fail_count += 1

    return {
        "summary": {
            "record_count": len(result_records),
            "pass_count": pass_count,
            "fail_count": fail_count,
        },
        "records": result_records,
    }


def build_k0_report_payload(
    analysis_payload: Dict[str, Any],
    *,
    aggregation_payload: Dict[str, Any] | None = None,
    qaqc_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build summary text and table payloads for a richer k0 report."""

    summary = dict(analysis_payload.get("summary") or {})
    summary["recognized_but_not_applied_count"] = len(analysis_payload.get("recognized_but_not_applied") or [])
    if aggregation_payload is not None:
        summary["aggregated_result_count"] = len(aggregation_payload.get("aggregated_results") or [])
    if qaqc_payload is not None:
        qaqc_summary = qaqc_payload.get("summary") or {}
        summary["qaqc_pass_count"] = int(qaqc_summary.get("pass_count", 0) or 0)
        summary["qaqc_fail_count"] = int(qaqc_summary.get("fail_count", 0) or 0)

    lines = ["FluxForge k0-NAA Report", "======================", ""]
    lines.append("Summary")
    lines.append("-------")
    for key in sorted(summary):
        lines.append(f"{key}: {summary[key]}")
    lines.append("")

    libraries = analysis_payload.get("libraries") or {}
    standard_library = libraries.get("standard_k0_library") or {}
    if standard_library:
        lines.append("Libraries")
        lines.append("---------")
        lines.append(f"standard_k0_library: {standard_library.get('library_id', 'unknown')} @ {standard_library.get('version', 'unknown')}")
        lines.append(f"status: {standard_library.get('status', 'unknown')}")
        lines.append("")

    lines.append("Element Results")
    lines.append("---------------")
    for row in analysis_payload.get("element_results") or []:
        lines.append(
            f"{row.get('element', 'unknown')}: {float(row.get('concentration_ug_g', 0.0) or 0.0):.6g} ± {float(row.get('concentration_unc_ug_g', 0.0) or 0.0):.3g} ug/g"
        )
    lines.append("")

    recognized = analysis_payload.get("recognized_but_not_applied") or []
    if recognized:
        lines.append("Recognized but not applied")
        lines.append("--------------------------")
        for item in recognized:
            lines.append(f"- {item}")
        lines.append("")

    if aggregation_payload is not None:
        lines.append("Aggregation")
        lines.append("-----------")
        for row in aggregation_payload.get("aggregated_results") or []:
            lines.append(
                f"{row.get('sample_id', 'sample')} / {row.get('element', 'unknown')}: {float(row.get('concentration_ug_g', 0.0) or 0.0):.6g} ± {float(row.get('concentration_unc_ug_g', 0.0) or 0.0):.3g} ug/g"
            )
        lines.append("")

    if qaqc_payload is not None:
        lines.append("QA/QC")
        lines.append("-----")
        for row in qaqc_payload.get("records") or []:
            lines.append(f"{row.get('role', 'record')} {row.get('sample_id', '')}: {row.get('status', 'unknown')}")
        lines.append("")

    return {
        "summary": summary,
        "text": "\n".join(lines) + "\n",
        "tables": {
            "element_results": analysis_payload.get("element_results") or [],
            "line_results": analysis_payload.get("line_results") or [],
            "aggregation": [] if aggregation_payload is None else aggregation_payload.get("aggregated_results") or [],
            "qaqc": [] if qaqc_payload is None else qaqc_payload.get("records") or [],
        },
    }
