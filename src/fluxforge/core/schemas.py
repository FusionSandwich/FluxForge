"""Artifact schema definitions for FluxForge outputs."""

from __future__ import annotations

import json
from typing import Any, Dict


def _schema_id(name: str, version: str = "v1") -> str:
    return f"fluxforge.{name}.{version}"


SPECTRUM_FILE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SpectrumFile",
    "type": "object",
    "required": ["schema", "spectrum", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("spectrum_file")},
        "spectrum": {
            "type": "object",
            "required": ["counts", "channels", "live_time", "real_time"],
            "properties": {
                "counts": {"type": "array", "items": {"type": "number"}},
                "counts_uncertainty": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "channels": {"type": "array", "items": {"type": "integer"}},
                "energies": {"type": ["array", "null"], "items": {"type": "number"}},
                "live_time": {"type": "number"},
                "real_time": {"type": "number"},
                "start_time": {"type": ["string", "null"]},
                "spectrum_id": {"type": "string"},
                "detector_id": {"type": "string"},
                "calibration": {"type": "object"},
                "metadata": {"type": "object"},
            },
        },
        "provenance": {"type": "object"},
    },
}

PEAK_REPORT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PeakReport",
    "type": "object",
    "required": ["schema", "spectrum_id", "peaks", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("peak_report")},
        "spectrum_id": {"type": "string"},
        "live_time_s": {"type": "number"},
        "peaks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["channel", "energy_keV", "amplitude", "raw_counts"],
                "properties": {
                    "channel": {"type": "integer"},
                    "energy_keV": {"type": "number"},
                    "amplitude": {"type": "number"},
                    "raw_counts": {"type": "number"},
                    "sigma_keV": {"type": "number"},
                    "area": {"type": "number"},
                    "region": {"type": "string"},
                    "is_report": {"type": "boolean"},
                    "report_isotope": {"type": "string"},
                    "report_file": {"type": "string"},
                    "label": {"type": "string"},
                    "manual": {"type": "boolean"},
                    "left_channel": {"type": "integer"},
                    "right_channel": {"type": "integer"},
                    "left_energy_keV": {"type": "number"},
                    "right_energy_keV": {"type": "number"},
                    "gross_counts": {"type": "number"},
                    "gross_counts_unc": {"type": "number"},
                    "net_counts": {"type": "number"},
                    "net_counts_unc": {"type": "number"},
                    "background_counts": {"type": "number"},
                    "background_subtracted": {"type": "boolean"},
                    "analysis_peak_energy_keV": {"type": "number"},
                },
            },
        },
        "provenance": {"type": "object"},
    },
}

LINE_ACTIVITIES_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "LineActivities",
    "type": "object",
    "required": ["schema", "lines", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("line_activities")},
        "spectrum_id": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "energy_keV",
                    "net_counts",
                    "activity_Bq",
                    "activity_unc_Bq",
                ],
                "properties": {
                    "energy_keV": {"type": "number"},
                    "isotope": {"type": "string"},
                    "reaction_id": {"type": "string"},
                    "net_counts": {"type": "number"},
                    "activity_Bq": {"type": "number"},
                    "activity_unc_Bq": {"type": "number"},
                    "efficiency": {"type": "number"},
                    "emission_probability": {"type": "number"},
                    "half_life_s": {"type": "number"},
                    "decay_constant_s": {"type": "number"},
                    "radioisotope_specific_activity_Bq_g": {"type": "number"},
                    "atoms": {"type": "number"},
                    "atoms_unc": {"type": "number"},
                    "radioactive_mass_g": {"type": "number"},
                    "radioactive_mass_unc_g": {"type": "number"},
                    "sample_mass_g": {"type": "number"},
                    "specific_activity_Bq_g": {"type": "number"},
                    "specific_activity_unc_Bq_g": {"type": "number"},
                    "radioactive_mass_fraction": {"type": "number"},
                },
            },
        },
        "provenance": {"type": "object"},
    },
}

REACTION_RATES_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ReactionRates",
    "type": "object",
    "required": ["schema", "rates", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("reaction_rates")},
        "segments": {"type": "array"},
        "rates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["reaction_id", "rate", "uncertainty"],
                "properties": {
                    "reaction_id": {"type": "string"},
                    "rate": {"type": "number"},
                    "uncertainty": {"type": "number"},
                    "half_life_s": {"type": "number"},
                },
            },
        },
        "provenance": {"type": "object"},
    },
}

RESPONSE_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ResponseBundle",
    "type": "object",
    "required": ["schema", "matrix", "reactions", "boundaries_eV", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("response_bundle")},
        "matrix": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "reactions": {"type": "array", "items": {"type": "string"}},
        "boundaries_eV": {"type": "array", "items": {"type": "number"}},
        "provenance": {"type": "object"},
    },
}

UNFOLD_RESULT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "UnfoldResult",
    "type": "object",
    "required": ["schema", "flux", "boundaries_eV", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("unfold_result")},
        "flux": {"type": "array", "items": {"type": "number"}},
        "covariance": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "chi2": {"type": "number"},
        "method": {"type": "string"},
        "diagnostics": {"type": "object"},
        "reactions": {"type": "array", "items": {"type": "string"}},
        "boundaries_eV": {"type": "array", "items": {"type": "number"}},
        "provenance": {"type": "object"},
    },
}

VALIDATION_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ValidationBundle",
    "type": "object",
    "required": ["schema", "metrics", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("validation_bundle")},
        "metrics": {"type": "object"},
        "truth_flux": {"type": "array", "items": {"type": "number"}},
        "predicted_flux": {"type": "array", "items": {"type": "number"}},
        "residuals": {"type": "array", "items": {"type": "number"}},
        "provenance": {"type": "object"},
    },
}

REPORT_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ReportBundle",
    "type": "object",
    "required": ["schema", "summary", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("report_bundle")},
        "summary": {"type": "object"},
        "inputs": {"type": "object"},
        "figures": {"type": "object"},
        "tables": {"type": "object"},
        "text_report": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

PEAK_OBSERVATION_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PeakObservationBundle",
    "type": "object",
    "required": ["schema", "observations", "summary", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("peak_observation_bundle")},
        "spectrum_id": {"type": "string"},
        "detector_id": {"type": "string"},
        "geometry_id": {"type": "string"},
        "summary": {"type": "object"},
        "observations": {"type": "array", "items": {"type": "object"}},
        "capability_flags": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

DETECTOR_CHARACTERIZATION_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DetectorCharacterization",
    "type": "object",
    "required": [
        "schema",
        "detector_id",
        "reference_position_mm",
        "efficiency_model",
        "provenance",
    ],
    "properties": {
        "schema": {"const": _schema_id("detector_characterization")},
        "detector_id": {"type": "string"},
        "reference_position_mm": {"type": "number"},
        "characterized_positions_mm": {"type": "array", "items": {"type": "number"}},
        "calibration_points": {"type": "array", "items": {"type": "object"}},
        "efficiency_model": {"type": "object"},
        "geometry_conversions": {"type": "object"},
        "peak_to_total_model": {"type": "object"},
        "coincidence_model": {"type": "object"},
        "capability_flags": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

FACILITY_CHARACTERIZATION_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FacilityCharacterization",
    "type": "object",
    "required": ["schema", "facility_id", "method", "flux_parameters", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("facility_characterization")},
        "facility_id": {"type": "string"},
        "method": {"type": "string"},
        "monitor_definitions": {"type": "array", "items": {"type": "object"}},
        "irradiation": {"type": "object"},
        "flux_parameters": {"type": "object"},
        "temperature": {"type": "object"},
        "gradients": {"type": "object"},
        "fast_flux": {"type": "object"},
        "capability_flags": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

K0_ANALYSIS_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "K0AnalysisBundle",
    "type": "object",
    "required": ["schema", "summary", "line_results", "element_results", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("k0_analysis_bundle")},
        "summary": {"type": "object"},
        "line_results": {"type": "array", "items": {"type": "object"}},
        "element_results": {"type": "array", "items": {"type": "object"}},
        "rejected_observations": {"type": "array", "items": {"type": "object"}},
        "applied_corrections": {"type": "array", "items": {"type": "string"}},
        "recognized_but_not_applied": {"type": "array", "items": {"type": "string"}},
        "user_supplied_corrections": {"type": "array", "items": {"type": "string"}},
        "default_assumptions": {"type": "array", "items": {"type": "string"}},
        "capability_flags": {"type": "object"},
        "libraries": {"type": "object"},
        "inputs": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

K0_AGGREGATION_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "K0AggregationBundle",
    "type": "object",
    "required": ["schema", "summary", "aggregated_results", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("k0_aggregation_bundle")},
        "summary": {"type": "object"},
        "aggregated_results": {"type": "array", "items": {"type": "object"}},
        "irradiation_summaries": {"type": "array", "items": {"type": "object"}},
        "inputs": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

K0_QAQC_BUNDLE_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "K0QaQcBundle",
    "type": "object",
    "required": ["schema", "summary", "records", "provenance"],
    "properties": {
        "schema": {"const": _schema_id("k0_qaqc_bundle")},
        "summary": {"type": "object"},
        "records": {"type": "array", "items": {"type": "object"}},
        "inputs": {"type": "object"},
        "provenance": {"type": "object"},
    },
}

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "SpectrumFile": SPECTRUM_FILE_SCHEMA,
    "PeakReport": PEAK_REPORT_SCHEMA,
    "LineActivities": LINE_ACTIVITIES_SCHEMA,
    "ReactionRates": REACTION_RATES_SCHEMA,
    "ResponseBundle": RESPONSE_BUNDLE_SCHEMA,
    "UnfoldResult": UNFOLD_RESULT_SCHEMA,
    "ValidationBundle": VALIDATION_BUNDLE_SCHEMA,
    "ReportBundle": REPORT_BUNDLE_SCHEMA,
    "PeakObservationBundle": PEAK_OBSERVATION_BUNDLE_SCHEMA,
    "DetectorCharacterization": DETECTOR_CHARACTERIZATION_SCHEMA,
    "FacilityCharacterization": FACILITY_CHARACTERIZATION_SCHEMA,
    "K0AnalysisBundle": K0_ANALYSIS_BUNDLE_SCHEMA,
    "K0AggregationBundle": K0_AGGREGATION_BUNDLE_SCHEMA,
    "K0QaQcBundle": K0_QAQC_BUNDLE_SCHEMA,
}


def schema_as_yaml(schema: Dict[str, Any]) -> str:
    """Return a YAML representation (JSON is valid YAML)."""
    return json.dumps(schema, indent=2)


def validate_artifact(
    payload: Dict[str, Any],
    *,
    require_definitions: bool = True,
    require_normalization: bool = True,
) -> list[str]:
    """Return a list of validation errors for a FluxForge artifact payload."""
    errors: list[str] = []
    schema_id = payload.get("schema")
    if not schema_id:
        return ["Missing schema identifier."]

    schema = next(
        (
            s
            for s in SCHEMAS.values()
            if s.get("properties", {}).get("schema", {}).get("const") == schema_id
        ),
        None,
    )
    if schema is None:
        return [f"Unknown schema identifier: {schema_id}."]

    required = schema.get("required", [])
    for key in required:
        if key not in payload:
            errors.append(f"Missing required field: {key}.")

    provenance = payload.get("provenance")
    if provenance is None:
        errors.append("Missing provenance block.")
    else:
        if "units" not in provenance:
            errors.append("Missing provenance units.")
        if require_normalization and "normalization" not in provenance:
            errors.append("Missing provenance normalization.")
        if require_definitions and "definitions" not in provenance:
            errors.append("Missing provenance definitions.")

    return errors


def validate_or_raise(
    payload: Dict[str, Any],
    *,
    require_definitions: bool = True,
    require_normalization: bool = True,
) -> None:
    """Raise ValueError if the artifact payload fails validation."""
    errors = validate_artifact(
        payload,
        require_definitions=require_definitions,
        require_normalization=require_normalization,
    )
    if errors:
        raise ValueError("Artifact validation failed: " + "; ".join(errors))
