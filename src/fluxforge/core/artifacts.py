"""Artifact schemas, validation, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Iterable, List, Mapping

SchemaDict = Dict[str, Any]

SCHEMA_VERSION = "0.1.0"

RESPONSE_SCHEMA: SchemaDict = {
    "schema_name": "fluxforge.response_matrix",
    "schema_version": SCHEMA_VERSION,
    "required_data_fields": ["matrix", "reactions", "boundaries_eV"],
    "required_units": ["matrix", "boundaries_eV"],
    "required_normalization": ["basis", "description"],
}

MEASUREMENTS_SCHEMA: SchemaDict = {
    "schema_name": "fluxforge.activation_measurements",
    "schema_version": SCHEMA_VERSION,
    "required_data_fields": ["segments", "reactions"],
    "required_units": [
        "duration_s",
        "relative_power",
        "net_counts",
        "live_time_s",
        "efficiency",
        "gamma_intensity",
        "half_life_s",
        "cooling_time_s",
        "dead_time_fraction",
    ],
    "required_normalization": ["basis", "description"],
}

GLS_SPECTRUM_SCHEMA: SchemaDict = {
    "schema_name": "fluxforge.gls_spectrum",
    "schema_version": SCHEMA_VERSION,
    "required_data_fields": [
        "boundaries_eV",
        "reactions",
        "flux",
        "covariance",
        "correlation",
        "chi2",
        "covariance_storage",
    ],
    "required_units": ["boundaries_eV", "flux", "covariance", "correlation", "chi2"],
    "required_normalization": ["basis", "description"],
    "covariance_storage": {
        "style": "staysl",
        "format": "full",
        "ordering": "energy_groups",
        "symmetry": "symmetric",
        "correlation_expectation": "stored",
    },
}


def _fluxforge_version() -> str:
    try:
        return version("fluxforge")
    except PackageNotFoundError:
        return "0.1.0"
    except Exception:
        return "0.1.0"


def _require_keys(container: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in container]
    if missing:
        raise ValueError(f"Missing required {context} keys: {', '.join(missing)}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_sha256(payload: Any) -> str:
    serialized = _json_dumps(payload).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_metadata(
    schema: SchemaDict,
    payload: Any,
    units: Mapping[str, str],
    normalization: Mapping[str, str],
    extra_versions: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    metadata = {
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "units": dict(units),
        "normalization": dict(normalization),
        "library_versions": {
            "fluxforge": _fluxforge_version(),
            "python": platform.python_version(),
            **(dict(extra_versions) if extra_versions else {}),
        },
        "checksums": {"payload_sha256": compute_sha256(payload)},
    }
    return metadata


def build_artifact(
    schema: SchemaDict,
    payload: Any,
    units: Mapping[str, str],
    normalization: Mapping[str, str],
    extra_versions: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    metadata = build_metadata(schema, payload, units, normalization, extra_versions)
    return {"metadata": metadata, "data": payload}


def validate_artifact(artifact: Mapping[str, Any], schema: SchemaDict) -> None:
    if "metadata" not in artifact or "data" not in artifact:
        raise ValueError("Artifact must include 'metadata' and 'data' sections.")
    metadata = artifact["metadata"]
    data = artifact["data"]
    _require_keys(metadata, ["schema_name", "schema_version", "units", "normalization", "library_versions", "checksums"], "metadata")
    if metadata["schema_name"] != schema["schema_name"]:
        raise ValueError(
            f"Artifact schema mismatch: expected {schema['schema_name']} but got {metadata['schema_name']}"
        )
    if metadata["schema_version"] != schema["schema_version"]:
        raise ValueError(
            f"Artifact schema version mismatch: expected {schema['schema_version']} but got {metadata['schema_version']}"
        )
    _require_keys(metadata["units"], schema["required_units"], "units")
    _require_keys(metadata["normalization"], schema["required_normalization"], "normalization")
    _require_keys(data, schema["required_data_fields"], "data")

    checksum = metadata.get("checksums", {}).get("payload_sha256")
    if checksum and checksum != compute_sha256(data):
        raise ValueError("Artifact checksum does not match payload.")


def extract_payload(obj: Any) -> Any:
    if isinstance(obj, dict) and "metadata" in obj and "data" in obj:
        return obj["data"]
    return obj


def ensure_square_matrix(matrix: List[List[float]], name: str) -> None:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError(f"{name} matrix must be square.")


def compute_correlation(covariance: List[List[float]]) -> List[List[float]]:
    ensure_square_matrix(covariance, "covariance")
    size = len(covariance)
    corr: List[List[float]] = []
    for i in range(size):
        row: List[float] = []
        denom_i = math.sqrt(max(covariance[i][i], 0.0))
        for j in range(size):
            denom_j = math.sqrt(max(covariance[j][j], 0.0))
            if denom_i == 0.0 or denom_j == 0.0:
                row.append(0.0)
            else:
                row.append(covariance[i][j] / (denom_i * denom_j))
        corr.append(row)
    return corr
