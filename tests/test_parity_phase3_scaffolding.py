from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PARITY_ROOT = REPO_ROOT / "tests" / "spectra" / "reference_parity" / "cases"
ACTIVATION_FIXTURES_ROOT = REPO_ROOT / "tests" / "activation_inventory" / "fixtures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase3_reference_parity_scaffolding_has_algorithm_and_workflow_cases() -> None:
    manifests = sorted(REFERENCE_PARITY_ROOT.glob("**/manifest.json"))
    assert manifests

    scopes = set()
    for path in manifests:
        payload = _load_json(path)
        scopes.add(str(payload.get("parity_scope") or "workflow"))
        assert payload["schema"] == "fluxforge.reference_parity_manifest.v1"
        assert payload.get("fixture_id")
        assert payload.get("workflow")
        assert payload.get("source_case")
        assert payload.get("source_ref")
        assert payload.get("source_paths")

    assert "algorithm" in scopes
    assert "workflow" in scopes


def test_phase3_activation_scaffolding_includes_second_irradiation_case() -> None:
    manifests = sorted(ACTIVATION_FIXTURES_ROOT.glob("**/manifest.json"))
    assert manifests

    fixture_ids = set()
    scopes = set()
    for path in manifests:
        payload = _load_json(path)
        fixture_ids.add(str(payload.get("fixture_id") or ""))
        scopes.add(str(payload.get("parity_scope") or "workflow"))
        assert payload["schema"] == "fluxforge.activation_inventory_fixture.v1"
        assert payload.get("source_ref")
        assert payload.get("source_paths")

    assert "minimal_second_irradiation_case" in fixture_ids
    assert "workflow" in scopes
