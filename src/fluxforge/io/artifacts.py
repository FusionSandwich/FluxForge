"""Artifact read/write helpers for FluxForge JSON/YAML bundles."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fluxforge.core.provenance import build_provenance, hash_file
from fluxforge.core.schemas import _schema_id
from fluxforge.io.spe import GammaSpectrum


def _load_yaml_module():
    if importlib.util.find_spec("yaml") is None:
        return None
    import yaml

    return yaml


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def write_artifact(path: Path, payload: Dict[str, Any]) -> None:
    """Write artifact data as JSON or YAML depending on extension."""
    if path.suffix.lower() in {".yml", ".yaml"}:
        yaml = _load_yaml_module()
        if yaml is None:
            raise ImportError("PyYAML is required to write YAML artifacts.")
        _write_text(path, yaml.safe_dump(payload, sort_keys=False))
    else:
        _write_text(path, json.dumps(payload, indent=2))


def read_artifact(path: Path) -> Dict[str, Any]:
    """Read artifact data from JSON or YAML."""
    if path.suffix.lower() in {".yml", ".yaml"}:
        yaml = _load_yaml_module()
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML artifacts.")
        return yaml.safe_load(_read_text(path))
    return json.loads(_read_text(path))


def make_spectrum_file(
    spectrum: GammaSpectrum,
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {
        "counts": "counts",
        "counts_uncertainty": "counts",
        "channels": "index",
        "energies": "keV",
        "live_time": "s",
        "real_time": "s",
    }
    definitions = {
        "counts": "raw counts per channel",
        "counts_uncertainty": "1-sigma per-channel uncertainty",
        "channels": "adc channel index",
        "energies": "calibrated energy in keV (null if unknown)",
        "live_time": "detector live time",
        "real_time": "clock time",
        "start_time": "ISO-8601 start time when available",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"counts": "raw"},
        definitions=definitions,
        source_hashes=hashes,
    )
    return {
        "schema": _schema_id("spectrum_file"),
        "spectrum": spectrum.to_dict(),
        "provenance": provenance,
    }


def write_spectrum_file(
    path: Path,
    spectrum: GammaSpectrum,
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_spectrum_file(spectrum, source_path=source_path)
    write_artifact(path, payload)
    return payload


def read_spectrum_file(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_peak_report(
    *,
    spectrum_id: str,
    live_time_s: float,
    peaks: Iterable[Dict[str, Any]],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {
        "energy_keV": "keV",
        "amplitude": "counts",
        "raw_counts": "counts",
        "sigma_keV": "keV",
        "area": "counts",
        "live_time_s": "s",
    }
    definitions = {
        "channel": "peak centroid channel",
        "energy_keV": "peak centroid energy",
        "amplitude": "peak height",
        "raw_counts": "peak height in raw spectrum",
        "sigma_keV": "gaussian sigma",
        "area": "net peak area",
        "live_time_s": "spectrum live time",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"peaks": "raw"},
        definitions=definitions,
        source_hashes=hashes,
    )
    return {
        "schema": _schema_id("peak_report"),
        "spectrum_id": spectrum_id,
        "live_time_s": live_time_s,
        "peaks": list(peaks),
        "provenance": provenance,
    }


def write_peak_report(
    path: Path,
    *,
    spectrum_id: str,
    live_time_s: float,
    peaks: Iterable[Dict[str, Any]],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_peak_report(
        spectrum_id=spectrum_id,
        live_time_s=live_time_s,
        peaks=peaks,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_peak_report(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_line_activities(
    *,
    spectrum_id: str,
    lines: Iterable[Dict[str, Any]],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {
        "energy_keV": "keV",
        "net_counts": "counts",
        "activity_Bq": "Bq",
        "activity_unc_Bq": "Bq",
        "half_life_s": "s",
        "decay_constant_s": "1/s",
        "radioisotope_specific_activity_Bq_g": "Bq/g",
        "atoms": "atoms",
        "atoms_unc": "atoms",
        "radioactive_mass_g": "g",
        "radioactive_mass_unc_g": "g",
        "sample_mass_g": "g",
        "specific_activity_Bq_g": "Bq/g",
        "specific_activity_unc_Bq_g": "Bq/g",
        "radioactive_mass_fraction": "fraction",
    }
    definitions = {
        "energy_keV": "gamma line energy",
        "net_counts": "net peak counts",
        "activity_Bq": "activity at count time unless corrected",
        "activity_unc_Bq": "1-sigma uncertainty on activity_Bq",
        "efficiency": "full-energy peak efficiency at energy",
        "emission_probability": "gamma emission probability",
        "half_life_s": "half-life for decay correction",
        "decay_constant_s": "radioactive decay constant inferred from half-life",
        "radioisotope_specific_activity_Bq_g": "specific activity of the pure radioactive isotope inferred from half-life",
        "atoms": "radioactive atoms inferred from activity and half-life",
        "atoms_unc": "1-sigma uncertainty on radioactive atom count",
        "radioactive_mass_g": "radioactive product mass inferred from activity",
        "radioactive_mass_unc_g": "1-sigma uncertainty on radioactive product mass",
        "sample_mass_g": "total sample mass used for specific-activity normalization",
        "specific_activity_Bq_g": "activity normalized by sample mass",
        "specific_activity_unc_Bq_g": "1-sigma uncertainty on specific activity",
        "radioactive_mass_fraction": "radioactive product mass divided by total sample mass",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"activity": "per-line"},
        definitions=definitions,
        source_hashes=hashes,
    )
    return {
        "schema": _schema_id("line_activities"),
        "spectrum_id": spectrum_id,
        "lines": list(lines),
        "provenance": provenance,
    }


def read_line_activities(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def write_line_activities(
    path: Path,
    *,
    spectrum_id: str,
    lines: Iterable[Dict[str, Any]],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_line_activities(
        spectrum_id=spectrum_id, lines=lines, source_path=source_path
    )
    write_artifact(path, payload)
    return payload


def make_reaction_rates(
    *,
    rates: Iterable[Dict[str, Any]],
    segments: Optional[List[Dict[str, Any]]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"rate": "reactions/s", "uncertainty": "reactions/s", "half_life_s": "s"}
    definitions = {
        "rate": "reaction rate at EOI per reaction",
        "uncertainty": "1-sigma uncertainty on rate",
        "half_life_s": "half-life used for decay correction",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"rates": "per-reaction"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("reaction_rates"),
        "rates": list(rates),
        "provenance": provenance,
    }
    if segments is not None:
        payload["segments"] = segments
    return payload


def write_reaction_rates(
    path: Path,
    *,
    rates: Iterable[Dict[str, Any]],
    segments: Optional[List[Dict[str, Any]]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_reaction_rates(
        rates=rates, segments=segments, source_path=source_path
    )
    write_artifact(path, payload)
    return payload


def read_reaction_rates(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_response_bundle(
    *,
    matrix: List[List[float]],
    reactions: List[str],
    boundaries_eV: List[float],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"matrix": "barn", "boundaries_eV": "eV"}
    definitions = {
        "matrix": "response matrix with rows as reactions and columns as energy groups",
        "boundaries_eV": "energy group boundaries in eV",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"matrix": "number_density_applied"},
        definitions=definitions,
        source_hashes=hashes,
    )
    return {
        "schema": _schema_id("response_bundle"),
        "matrix": matrix,
        "reactions": reactions,
        "boundaries_eV": boundaries_eV,
        "provenance": provenance,
    }


def write_response_bundle(
    path: Path,
    *,
    matrix: List[List[float]],
    reactions: List[str],
    boundaries_eV: List[float],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_response_bundle(
        matrix=matrix,
        reactions=reactions,
        boundaries_eV=boundaries_eV,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_response_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_unfold_result(
    *,
    boundaries_eV: List[float],
    reactions: List[str],
    flux: List[float],
    covariance: List[List[float]],
    chi2: float,
    method: str,
    diagnostics: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"flux": "a.u.", "covariance": "a.u.^2", "boundaries_eV": "eV"}
    definitions = {
        "flux": "group-integrated flux per energy bin",
        "covariance": "covariance of group-integrated flux",
        "chi2": "chi^2 of measured vs predicted rates",
        "boundaries_eV": "energy group boundaries in eV",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"flux": "per-group"},
        definitions=definitions,
        source_hashes=hashes,
    )
    return {
        "schema": _schema_id("unfold_result"),
        "boundaries_eV": boundaries_eV,
        "reactions": reactions,
        "flux": flux,
        "covariance": covariance,
        "chi2": chi2,
        "method": method,
        "diagnostics": diagnostics or {},
        "provenance": provenance,
    }


def write_unfold_result(
    path: Path,
    *,
    boundaries_eV: List[float],
    reactions: List[str],
    flux: List[float],
    covariance: List[List[float]],
    chi2: float,
    method: str,
    diagnostics: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_unfold_result(
        boundaries_eV=boundaries_eV,
        reactions=reactions,
        flux=flux,
        covariance=covariance,
        chi2=chi2,
        method=method,
        diagnostics=diagnostics,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_unfold_result(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_validation_bundle(
    *,
    metrics: Dict[str, Any],
    truth_flux: List[float],
    predicted_flux: List[float],
    residuals: List[float],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"truth_flux": "a.u.", "predicted_flux": "a.u.", "residuals": "a.u."}
    definitions = {
        "truth_flux": "reference flux for comparison",
        "predicted_flux": "unfolded flux",
        "residuals": "predicted_flux - truth_flux",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"metrics": "comparison"},
        definitions=definitions,
        source_hashes=hashes,
    )
    return {
        "schema": _schema_id("validation_bundle"),
        "metrics": metrics,
        "truth_flux": truth_flux,
        "predicted_flux": predicted_flux,
        "residuals": residuals,
        "provenance": provenance,
    }


def write_validation_bundle(
    path: Path,
    *,
    metrics: Dict[str, Any],
    truth_flux: List[float],
    predicted_flux: List[float],
    residuals: List[float],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_validation_bundle(
        metrics=metrics,
        truth_flux=truth_flux,
        predicted_flux=predicted_flux,
        residuals=residuals,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_validation_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_peak_observation_bundle(
    *,
    spectrum_id: str,
    detector_id: str,
    geometry_id: str,
    observations: Iterable[Dict[str, Any]],
    summary: Dict[str, Any],
    capability_flags: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {
        "line_energy_keV": "keV",
        "net_peak_area": "counts",
        "area_uncertainty": "counts",
        "live_time_s": "s",
        "real_time_s": "s",
        "irradiation_time_s": "s",
        "decay_time_s": "s",
        "counting_time_s": "s",
    }
    definitions = {
        "observations": "normalized peak observations that separate spectrum interpretation from k0 interpretation",
        "eligibility_class": "k0 line applicability classification used to accept or reject automatic use",
        "summary": "counts of accepted and rejected observations for the bundle",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"observations": "per-peak"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("peak_observation_bundle"),
        "spectrum_id": spectrum_id,
        "detector_id": detector_id,
        "geometry_id": geometry_id,
        "observations": list(observations),
        "summary": summary,
        "provenance": provenance,
    }
    if capability_flags is not None:
        payload["capability_flags"] = capability_flags
    return payload


def write_peak_observation_bundle(
    path: Path,
    *,
    spectrum_id: str,
    detector_id: str,
    geometry_id: str,
    observations: Iterable[Dict[str, Any]],
    summary: Dict[str, Any],
    capability_flags: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_peak_observation_bundle(
        spectrum_id=spectrum_id,
        detector_id=detector_id,
        geometry_id=geometry_id,
        observations=observations,
        summary=summary,
        capability_flags=capability_flags,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_peak_observation_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_detector_characterization(
    *,
    detector_id: str,
    reference_position_mm: float,
    characterized_positions_mm: List[float],
    calibration_points: List[Dict[str, Any]],
    efficiency_model: Dict[str, Any],
    geometry_conversions: Optional[Dict[str, Any]] = None,
    peak_to_total_model: Optional[Dict[str, Any]] = None,
    coincidence_model: Optional[Dict[str, Any]] = None,
    capability_flags: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"reference_position_mm": "mm", "characterized_positions_mm": "mm"}
    definitions = {
        "efficiency_model": "reference-position full-energy peak efficiency model",
        "geometry_conversions": "explicit geometry-conversion models relative to the reference position",
        "peak_to_total_model": "peak-to-total characterization used for coincidence-aware workflows when available",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"efficiency_model": "reference_position"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("detector_characterization"),
        "detector_id": detector_id,
        "reference_position_mm": reference_position_mm,
        "characterized_positions_mm": characterized_positions_mm,
        "calibration_points": calibration_points,
        "efficiency_model": efficiency_model,
        "provenance": provenance,
    }
    if geometry_conversions is not None:
        payload["geometry_conversions"] = geometry_conversions
    if peak_to_total_model is not None:
        payload["peak_to_total_model"] = peak_to_total_model
    if coincidence_model is not None:
        payload["coincidence_model"] = coincidence_model
    if capability_flags is not None:
        payload["capability_flags"] = capability_flags
    return payload


def write_detector_characterization(
    path: Path,
    *,
    detector_id: str,
    reference_position_mm: float,
    characterized_positions_mm: List[float],
    calibration_points: List[Dict[str, Any]],
    efficiency_model: Dict[str, Any],
    geometry_conversions: Optional[Dict[str, Any]] = None,
    peak_to_total_model: Optional[Dict[str, Any]] = None,
    coincidence_model: Optional[Dict[str, Any]] = None,
    capability_flags: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_detector_characterization(
        detector_id=detector_id,
        reference_position_mm=reference_position_mm,
        characterized_positions_mm=characterized_positions_mm,
        calibration_points=calibration_points,
        efficiency_model=efficiency_model,
        geometry_conversions=geometry_conversions,
        peak_to_total_model=peak_to_total_model,
        coincidence_model=coincidence_model,
        capability_flags=capability_flags,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_detector_characterization(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_facility_characterization(
    *,
    facility_id: str,
    method: str,
    monitor_definitions: List[Dict[str, Any]],
    flux_parameters: Dict[str, Any],
    irradiation: Optional[Dict[str, Any]] = None,
    temperature: Optional[Dict[str, Any]] = None,
    gradients: Optional[Dict[str, Any]] = None,
    fast_flux: Optional[Dict[str, Any]] = None,
    capability_flags: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {
        "f": "ratio",
        "alpha": "dimensionless",
        "phi_thermal": "n/cm^2/s",
        "phi_epithermal": "n/cm^2/s",
    }
    definitions = {
        "monitor_definitions": "traceable monitor metadata used for facility characterization",
        "flux_parameters": "facility neutron-spectrum parameters assigned to downstream k0 analyses",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"flux_parameters": "facility"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("facility_characterization"),
        "facility_id": facility_id,
        "method": method,
        "monitor_definitions": monitor_definitions,
        "flux_parameters": flux_parameters,
        "provenance": provenance,
    }
    if irradiation is not None:
        payload["irradiation"] = irradiation
    if temperature is not None:
        payload["temperature"] = temperature
    if gradients is not None:
        payload["gradients"] = gradients
    if fast_flux is not None:
        payload["fast_flux"] = fast_flux
    if capability_flags is not None:
        payload["capability_flags"] = capability_flags
    return payload


def write_facility_characterization(
    path: Path,
    *,
    facility_id: str,
    method: str,
    monitor_definitions: List[Dict[str, Any]],
    flux_parameters: Dict[str, Any],
    irradiation: Optional[Dict[str, Any]] = None,
    temperature: Optional[Dict[str, Any]] = None,
    gradients: Optional[Dict[str, Any]] = None,
    fast_flux: Optional[Dict[str, Any]] = None,
    capability_flags: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_facility_characterization(
        facility_id=facility_id,
        method=method,
        monitor_definitions=monitor_definitions,
        flux_parameters=flux_parameters,
        irradiation=irradiation,
        temperature=temperature,
        gradients=gradients,
        fast_flux=fast_flux,
        capability_flags=capability_flags,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_facility_characterization(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_k0_analysis_bundle(
    *,
    summary: Dict[str, Any],
    line_results: List[Dict[str, Any]],
    element_results: List[Dict[str, Any]],
    rejected_observations: Optional[List[Dict[str, Any]]] = None,
    applied_corrections: Optional[List[str]] = None,
    recognized_but_not_applied: Optional[List[str]] = None,
    user_supplied_corrections: Optional[List[str]] = None,
    default_assumptions: Optional[List[str]] = None,
    capability_flags: Optional[Dict[str, Any]] = None,
    libraries: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"concentration_ug_g": "ug/g", "concentration_unc_ug_g": "ug/g"}
    definitions = {
        "line_results": "per-line k0 concentration results after eligibility filtering",
        "element_results": "aggregated elemental k0 concentration results",
        "recognized_but_not_applied": "workflow-relevant corrections acknowledged but not yet applied by this first-pass implementation",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"line_results": "per-line", "element_results": "per-element"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("k0_analysis_bundle"),
        "summary": summary,
        "line_results": line_results,
        "element_results": element_results,
        "provenance": provenance,
    }
    if rejected_observations is not None:
        payload["rejected_observations"] = rejected_observations
    if applied_corrections is not None:
        payload["applied_corrections"] = applied_corrections
    if recognized_but_not_applied is not None:
        payload["recognized_but_not_applied"] = recognized_but_not_applied
    if user_supplied_corrections is not None:
        payload["user_supplied_corrections"] = user_supplied_corrections
    if default_assumptions is not None:
        payload["default_assumptions"] = default_assumptions
    if capability_flags is not None:
        payload["capability_flags"] = capability_flags
    if libraries is not None:
        payload["libraries"] = libraries
    if inputs is not None:
        payload["inputs"] = inputs
    return payload


def write_k0_analysis_bundle(
    path: Path,
    *,
    summary: Dict[str, Any],
    line_results: List[Dict[str, Any]],
    element_results: List[Dict[str, Any]],
    rejected_observations: Optional[List[Dict[str, Any]]] = None,
    applied_corrections: Optional[List[str]] = None,
    recognized_but_not_applied: Optional[List[str]] = None,
    user_supplied_corrections: Optional[List[str]] = None,
    default_assumptions: Optional[List[str]] = None,
    capability_flags: Optional[Dict[str, Any]] = None,
    libraries: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_k0_analysis_bundle(
        summary=summary,
        line_results=line_results,
        element_results=element_results,
        rejected_observations=rejected_observations,
        applied_corrections=applied_corrections,
        recognized_but_not_applied=recognized_but_not_applied,
        user_supplied_corrections=user_supplied_corrections,
        default_assumptions=default_assumptions,
        capability_flags=capability_flags,
        libraries=libraries,
        inputs=inputs,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_k0_analysis_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_k0_aggregation_bundle(
    *,
    summary: Dict[str, Any],
    aggregated_results: List[Dict[str, Any]],
    irradiation_summaries: Optional[List[Dict[str, Any]]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"concentration_ug_g": "ug/g", "concentration_unc_ug_g": "ug/g"}
    definitions = {
        "aggregated_results": "combined k0 results across measurements and irradiations",
        "irradiation_summaries": "per-irradiation aggregated summaries",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={
            "aggregated_results": "cross-measurement",
            "irradiation_summaries": "per-irradiation",
        },
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("k0_aggregation_bundle"),
        "summary": summary,
        "aggregated_results": aggregated_results,
        "provenance": provenance,
    }
    if irradiation_summaries is not None:
        payload["irradiation_summaries"] = irradiation_summaries
    if inputs is not None:
        payload["inputs"] = inputs
    return payload


def write_k0_aggregation_bundle(
    path: Path,
    *,
    summary: Dict[str, Any],
    aggregated_results: List[Dict[str, Any]],
    irradiation_summaries: Optional[List[Dict[str, Any]]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_k0_aggregation_bundle(
        summary=summary,
        aggregated_results=aggregated_results,
        irradiation_summaries=irradiation_summaries,
        inputs=inputs,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_k0_aggregation_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_k0_qaqc_bundle(
    *,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"bias_ug_g": "ug/g", "relative_bias_percent": "%"}
    definitions = {
        "records": "blank and CRM QA/QC evaluations derived from k0 analysis bundles",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"records": "per-qaqc-record"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("k0_qaqc_bundle"),
        "summary": summary,
        "records": records,
        "provenance": provenance,
    }
    if inputs is not None:
        payload["inputs"] = inputs
    return payload


def write_k0_qaqc_bundle(
    path: Path,
    *,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_k0_qaqc_bundle(
        summary=summary,
        records=records,
        inputs=inputs,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_k0_qaqc_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_report_bundle(
    *,
    summary: Dict[str, Any],
    inputs: Optional[Dict[str, Any]] = None,
    figures: Optional[Dict[str, Any]] = None,
    tables: Optional[Dict[str, Any]] = None,
    text_report: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"summary": "mixed"}
    definitions = {"summary": "aggregate summary across artifacts"}
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"report": "aggregate"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("report_bundle"),
        "summary": summary,
        "provenance": provenance,
    }
    if inputs is not None:
        payload["inputs"] = inputs
    if figures is not None:
        payload["figures"] = figures
    if tables is not None:
        payload["tables"] = tables
    if text_report is not None:
        payload["text_report"] = text_report
    return payload


def write_report_bundle(
    path: Path,
    *,
    summary: Dict[str, Any],
    inputs: Optional[Dict[str, Any]] = None,
    figures: Optional[Dict[str, Any]] = None,
    tables: Optional[Dict[str, Any]] = None,
    text_report: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_report_bundle(
        summary=summary,
        inputs=inputs,
        figures=figures,
        tables=tables,
        text_report=text_report,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_report_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


def make_ffexp_bundle(
    *,
    summary: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    activities: Optional[Dict[str, Any]] = None,
    inventory: Optional[Dict[str, Any]] = None,
    masking: Optional[Dict[str, Any]] = None,
    optimization: Optional[Dict[str, Any]] = None,
    second_irradiation: Optional[Dict[str, Any]] = None,
    plot_manifest: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    units = {"summary": "mixed", "metadata": "mixed"}
    definitions = {
        "summary": "aggregate readiness and workflow summary for benchmark export",
        "metadata": "sample, detector, method, and provenance metadata included in the ffexp bundle",
    }
    hashes = {"source": hash_file(source_path)} if source_path else None
    provenance = build_provenance(
        units=units,
        normalization={"bundle": "experimental"},
        definitions=definitions,
        source_hashes=hashes,
    )
    payload = {
        "schema": _schema_id("ffexp_bundle"),
        "format": ".ffexp",
        "summary": summary,
        "provenance": provenance,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    if activities is not None:
        payload["activities"] = activities
    if inventory is not None:
        payload["inventory"] = inventory
    if masking is not None:
        payload["masking"] = masking
    if optimization is not None:
        payload["optimization"] = optimization
    if second_irradiation is not None:
        payload["second_irradiation"] = second_irradiation
    if plot_manifest is not None:
        payload["plot_manifest"] = plot_manifest
    return payload


def write_ffexp_bundle(
    path: Path,
    *,
    summary: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    activities: Optional[Dict[str, Any]] = None,
    inventory: Optional[Dict[str, Any]] = None,
    masking: Optional[Dict[str, Any]] = None,
    optimization: Optional[Dict[str, Any]] = None,
    second_irradiation: Optional[Dict[str, Any]] = None,
    plot_manifest: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_ffexp_bundle(
        summary=summary,
        metadata=metadata,
        activities=activities,
        inventory=inventory,
        masking=masking,
        optimization=optimization,
        second_irradiation=second_irradiation,
        plot_manifest=plot_manifest,
        source_path=source_path,
    )
    write_artifact(path, payload)
    return payload


def read_ffexp_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)
