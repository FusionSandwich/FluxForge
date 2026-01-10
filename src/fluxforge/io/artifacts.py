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
        "channels": "index",
        "energies": "keV",
        "live_time": "s",
        "real_time": "s",
    }
    definitions = {
        "counts": "raw counts per channel",
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
    }
    definitions = {
        "energy_keV": "gamma line energy",
        "net_counts": "net peak counts",
        "activity_Bq": "activity at count time unless corrected",
        "activity_unc_Bq": "1-sigma uncertainty on activity_Bq",
        "efficiency": "full-energy peak efficiency at energy",
        "emission_probability": "gamma emission probability",
        "half_life_s": "half-life for decay correction",
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


def write_line_activities(
    path: Path,
    *,
    spectrum_id: str,
    lines: Iterable[Dict[str, Any]],
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_line_activities(spectrum_id=spectrum_id, lines=lines, source_path=source_path)
    write_artifact(path, payload)
    return payload


def read_line_activities(path: Path) -> Dict[str, Any]:
    return read_artifact(path)


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
    payload = make_reaction_rates(rates=rates, segments=segments, source_path=source_path)
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
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_unfold_result(
        boundaries_eV=boundaries_eV,
        reactions=reactions,
        flux=flux,
        covariance=covariance,
        chi2=chi2,
        method=method,
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


def make_report_bundle(
    *,
    summary: Dict[str, Any],
    inputs: Optional[Dict[str, Any]] = None,
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
    return payload


def write_report_bundle(
    path: Path,
    *,
    summary: Dict[str, Any],
    inputs: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload = make_report_bundle(summary=summary, inputs=inputs, source_path=source_path)
    write_artifact(path, payload)
    return payload


def read_report_bundle(path: Path) -> Dict[str, Any]:
    return read_artifact(path)
