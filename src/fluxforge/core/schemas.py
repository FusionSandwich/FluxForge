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
                "required": ["energy_keV", "net_counts", "activity_Bq", "activity_unc_Bq"],
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
        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
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
        "covariance": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "chi2": {"type": "number"},
        "method": {"type": "string"},
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
}


def schema_as_yaml(schema: Dict[str, Any]) -> str:
    """Return a YAML representation (JSON is valid YAML)."""
    return json.dumps(schema, indent=2)
