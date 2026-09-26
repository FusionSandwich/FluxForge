"""Local-only run-readiness manifest validation.

The preflight verifies that a manifest is structurally complete and that every
listed local file matches its declared SHA-256 digest. It does not execute any
scientific model or mutate the checked inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

RUN_READINESS_MANIFEST_SCHEMA = "fluxforge.run_readiness_manifest.v1"
RUN_READINESS_REPORT_SCHEMA = "fluxforge.run_readiness_check.v1"
_REQUIRED_METADATA_FIELDS = ("manifest_id", "purpose")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\\\]")


class RunReadinessManifestError(ValueError):
    """Raised when a run-readiness manifest or one of its bindings is invalid."""


def _require_nonempty_string(mapping: dict[str, Any], key: str, *, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RunReadinessManifestError(
            f"Missing or invalid required metadata '{where}.{key}': expected a "
            "non-empty string."
        )
    return value.strip()


def _validate_relative_path(value: Any, *, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunReadinessManifestError(
            f"Missing or invalid required metadata 'artifacts[{index}].path': "
            "expected a non-empty relative path."
        )
    raw = value.strip()
    normalized = raw.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        pure.is_absolute()
        or _WINDOWS_DRIVE_PATTERN.match(raw) is not None
        or ".." in pure.parts
    ):
        raise RunReadinessManifestError(
            f"Invalid artifact path at artifacts[{index}].path: {raw!r}. "
            "Paths must stay within the manifest directory."
        )
    return pure.as_posix()


def _validate_sha256(value: Any, *, index: int) -> str:
    if value is None or value == "":
        raise RunReadinessManifestError(
            f"Missing required SHA-256 hash 'artifacts[{index}].sha256'."
        )
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RunReadinessManifestError(
            f"Malformed SHA-256 hash at artifacts[{index}].sha256: "
            "expected exactly 64 hexadecimal characters."
        )
    return value.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_run_readiness_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate one local manifest and its file/hash bindings.

    Paths listed in the manifest are resolved relative to the manifest's parent
    directory. The function is read-only and deterministic: it performs no model
    execution, network access, directory creation, or input mutation.
    """

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RunReadinessManifestError(
            f"Run-readiness manifest not found: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise RunReadinessManifestError(
            f"Run-readiness manifest is not a file: {manifest_path}"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunReadinessManifestError(
            f"Invalid JSON in run-readiness manifest {manifest_path}: "
            f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise RunReadinessManifestError(
            "Run-readiness manifest root must be a JSON object."
        )

    schema = payload.get("schema")
    if schema != RUN_READINESS_MANIFEST_SCHEMA:
        raise RunReadinessManifestError(
            "Missing or unsupported required metadata 'schema': expected "
            f"{RUN_READINESS_MANIFEST_SCHEMA!r}."
        )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RunReadinessManifestError(
            "Missing or invalid required metadata 'metadata': expected a JSON object."
        )
    for field in _REQUIRED_METADATA_FIELDS:
        _require_nonempty_string(metadata, field, where="metadata")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RunReadinessManifestError(
            "Missing or invalid required metadata 'artifacts': expected a "
            "non-empty list."
        )

    manifest_root = manifest_path.parent.resolve()
    seen_paths: set[str] = set()
    checked: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise RunReadinessManifestError(
                f"Invalid metadata at artifacts[{index}]: expected a JSON object."
            )

        relative_path = _validate_relative_path(item.get("path"), index=index)
        role = _require_nonempty_string(item, "role", where=f"artifacts[{index}]")
        expected_sha256 = _validate_sha256(item.get("sha256"), index=index)

        if relative_path in seen_paths:
            raise RunReadinessManifestError(
                f"Duplicate artifact path in run-readiness manifest: {relative_path}"
            )
        seen_paths.add(relative_path)

        candidate = (manifest_root / relative_path).resolve()
        try:
            candidate.relative_to(manifest_root)
        except ValueError as exc:
            raise RunReadinessManifestError(
                f"Artifact path escapes manifest directory: {relative_path}"
            ) from exc

        if not candidate.exists():
            failures.append(f"missing file: {relative_path}")
            checked.append(
                {
                    "path": relative_path,
                    "role": role,
                    "expected_sha256": expected_sha256,
                    "status": "missing",
                }
            )
            continue
        if not candidate.is_file():
            failures.append(f"not a regular file: {relative_path}")
            checked.append(
                {
                    "path": relative_path,
                    "role": role,
                    "expected_sha256": expected_sha256,
                    "status": "not_file",
                }
            )
            continue

        actual_sha256 = _sha256_file(candidate)
        status = "pass" if actual_sha256 == expected_sha256 else "hash_mismatch"
        checked.append(
            {
                "path": relative_path,
                "role": role,
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "size_bytes": candidate.stat().st_size,
                "status": status,
            }
        )
        if status != "pass":
            failures.append(
                f"SHA-256 mismatch: {relative_path} "
                f"(expected {expected_sha256}, got {actual_sha256})"
            )

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RunReadinessManifestError(
            f"Run-readiness preflight failed for {metadata['manifest_id']}:\n{details}"
        )

    return {
        "schema": RUN_READINESS_REPORT_SCHEMA,
        "manifest_schema": RUN_READINESS_MANIFEST_SCHEMA,
        "manifest_id": metadata["manifest_id"].strip(),
        "purpose": metadata["purpose"].strip(),
        "status": "pass",
        "checked_count": len(checked),
        "artifacts": checked,
    }


def write_run_readiness_report(path: Path, report: dict[str, Any]) -> Path:
    """Write a new JSON report without replacing an existing file."""

    target = Path(path)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing run-readiness report: {target}"
        ) from exc
    return target


__all__ = [
    "RUN_READINESS_MANIFEST_SCHEMA",
    "RUN_READINESS_REPORT_SCHEMA",
    "RunReadinessManifestError",
    "check_run_readiness_manifest",
    "write_run_readiness_report",
]
