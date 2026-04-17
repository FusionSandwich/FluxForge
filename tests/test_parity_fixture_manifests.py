from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PARITY_ROOT = REPO_ROOT / "tests" / "spectra" / "reference_parity"
ACTIVATION_FIXTURES_ROOT = REPO_ROOT / "tests" / "activation_inventory" / "fixtures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_common_manifest_shape(manifest: dict) -> None:
    required = {
        "schema",
        "fixture_id",
        "parity_scope",
        "source_repo",
        "source_ref",
        "source_paths",
        "source_case",
        "workflow",
        "input_files",
        "expected_outputs",
        "tolerances",
        "provenance_notes",
    }
    assert required.issubset(manifest.keys())
    assert manifest["parity_scope"] in {"algorithm", "workflow"}
    assert isinstance(manifest["source_paths"], list) and manifest["source_paths"]
    assert isinstance(manifest["input_files"], list) and manifest["input_files"]
    assert isinstance(manifest["expected_outputs"], list) and manifest["expected_outputs"]
    assert isinstance(manifest["provenance_notes"], list) and manifest["provenance_notes"]
    assert isinstance(manifest["tolerances"], dict) and manifest["tolerances"]
    for value in manifest["tolerances"].values():
        assert float(value) >= 0.0


def test_reference_parity_manifests_follow_contract():
    schema_path = REFERENCE_PARITY_ROOT / "manifest.schema.json"
    assert schema_path.exists()

    manifests = sorted(REFERENCE_PARITY_ROOT.glob("cases/**/manifest.json"))
    assert manifests

    for path in manifests:
        payload = _load_json(path)
        _assert_common_manifest_shape(payload)
        assert payload["schema"] == "fluxforge.reference_parity_manifest.v1"
        for file_name in payload["input_files"]:
            assert (path.parent / file_name).exists()
        for file_name in payload["expected_outputs"]:
            assert (path.parent / file_name).exists()


def test_activation_inventory_fixture_manifests_follow_contract():
    schema_path = ACTIVATION_FIXTURES_ROOT / "manifest.schema.json"
    assert schema_path.exists()

    manifests = sorted(ACTIVATION_FIXTURES_ROOT.glob("**/manifest.json"))
    assert manifests

    for path in manifests:
        payload = _load_json(path)
        _assert_common_manifest_shape(payload)
        assert payload["schema"] == "fluxforge.activation_inventory_fixture.v1"
        for file_name in payload["input_files"]:
            assert (path.parent / file_name).exists()
        for file_name in payload["expected_outputs"]:
            assert (path.parent / file_name).exists()
