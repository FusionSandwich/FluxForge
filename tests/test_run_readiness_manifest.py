from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fluxforge.validation.run_readiness import (
    RUN_READINESS_MANIFEST_SCHEMA,
    RunReadinessManifestError,
    check_run_readiness_manifest,
    write_run_readiness_report,
)


def _write_manifest(
    root: Path, *, artifacts: list[dict], metadata: dict | None = None
) -> Path:
    manifest = root / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": RUN_READINESS_MANIFEST_SCHEMA,
                "metadata": metadata
                or {"manifest_id": "local-smoke", "purpose": "run readiness"},
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_valid_manifest_passes(tmp_path: Path) -> None:
    data = b"deterministic input\n"
    (tmp_path / "input.dat").write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    manifest = _write_manifest(
        tmp_path,
        artifacts=[{"path": "input.dat", "role": "model_input", "sha256": expected}],
    )

    report = check_run_readiness_manifest(manifest)

    assert report["status"] == "pass"
    assert report["checked_count"] == 1
    assert report["artifacts"][0]["actual_sha256"] == expected


def test_missing_required_metadata_fails_clearly(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        metadata={"manifest_id": "local-smoke"},
        artifacts=[{"path": "input.dat", "role": "model_input", "sha256": "0" * 64}],
    )

    with pytest.raises(RunReadinessManifestError, match=r"metadata\.purpose"):
        check_run_readiness_manifest(manifest)


def test_missing_hash_fails_clearly(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        artifacts=[{"path": "input.dat", "role": "model_input"}],
    )

    with pytest.raises(RunReadinessManifestError, match="Missing required SHA-256"):
        check_run_readiness_manifest(manifest)


def test_malformed_hash_fails_clearly(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        artifacts=[{"path": "input.dat", "role": "model_input", "sha256": "abc123"}],
    )

    with pytest.raises(RunReadinessManifestError, match="Malformed SHA-256"):
        check_run_readiness_manifest(manifest)


def test_missing_file_fails_clearly(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        artifacts=[{"path": "missing.dat", "role": "model_input", "sha256": "0" * 64}],
    )

    with pytest.raises(RunReadinessManifestError, match=r"missing file: missing\.dat"):
        check_run_readiness_manifest(manifest)


def test_hash_mismatch_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "input.dat").write_text("actual\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        artifacts=[{"path": "input.dat", "role": "model_input", "sha256": "0" * 64}],
    )

    with pytest.raises(RunReadinessManifestError, match="SHA-256 mismatch: input.dat"):
        check_run_readiness_manifest(manifest)


def test_report_writer_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_run_readiness_report(target, {"status": "pass"})

    assert target.read_text(encoding="utf-8") == "sentinel\n"
