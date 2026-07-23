from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / ".github" / "project-management" / "full_parity_ledger.json"

STATUS_ENUM = {
    "validated",
    "implemented",
    "prototype",
    "scaffolded",
    "planned",
    "conditional-hardware",
    "reference-only",
    "not-applicable",
}
LAYERS = ("backend", "cli", "gui", "fixture", "test")


def _expected_ids() -> set[str]:
    return {
        *(f"3.{number}" for number in range(18, 28)),
        *(f"3N.{number}" for number in range(1, 17)),
        *(f"4.{number}" for number in range(1, 5)),
        *(f"5.{number}" for number in range(1, 7)),
        *(f"BG.{number}" for number in range(1, 9)),
    }


def _load_ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_full_parity_ledger_has_complete_unique_coverage() -> None:
    ledger = _load_ledger()
    assert ledger["schema"] == "fluxforge.full-parity-ledger.v1"
    assert ledger["version"] == 1
    assert set(ledger["status_enum"]) == STATUS_ENUM

    entries = ledger["entries"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    assert set(ids) == _expected_ids()


def test_full_parity_ledger_entries_follow_the_evidence_contract() -> None:
    ledger = _load_ledger()
    sources = ledger["sources"]
    assert {
        "fluxforge-roadmap",
        "interspec",
        "peakeasy",
        "gammavision",
        "genie",
        "hyperlab",
        "umg",
        "staysl",
    } == set(sources)
    assert all(url.startswith("https://") for url in sources.values())

    referenced_sources: set[str] = set()
    for entry in ledger["entries"]:
        assert isinstance(entry["title"], str) and entry["title"].strip()
        assert isinstance(entry["roadmap"], str) and entry["roadmap"].strip()
        assert entry["status"] in STATUS_ENUM
        assert entry["source_urls"]
        assert set(entry["source_urls"]) <= set(sources)
        referenced_sources.update(entry["source_urls"])
        assert isinstance(entry["notes"], str) and entry["notes"].strip()
        assert isinstance(entry["evidence"], list) and entry["evidence"]

        for layer in LAYERS:
            assert entry[layer] in STATUS_ENUM

        for evidence_path in entry["evidence"]:
            assert not Path(evidence_path).is_absolute()
            assert (
                ROOT / evidence_path
            ).exists(), f"{entry['id']} references missing evidence {evidence_path}"

    assert referenced_sources == set(sources)


def test_optimization_and_inverse_work_remains_honestly_labeled() -> None:
    entries = {entry["id"]: entry for entry in _load_ledger()["entries"]}
    prototype_ids = {"3.19", "3.25", "3N.11", "3N.12", "3N.13", "5.5"}
    assert {
        entry_id
        for entry_id in prototype_ids
        if entries[entry_id]["status"] != "prototype"
    } == set()

    optimization_text = " ".join(
        entries[entry_id]["title"] + " " + entries[entry_id]["notes"]
        for entry_id in ("3N.11", "3N.12", "3N.13")
    )
    for method in ("DI-FOM", "FIM", "MWDCS", "Pareto", "posterior"):
        assert method.casefold() in optimization_text.casefold()

    status_doc = (ROOT / "docs" / "ROADMAP_EXECUTION_STATUS.md").read_text(
        encoding="utf-8"
    )
    assert "Scientific Maturity of Irradiation Optimization" in status_doc
    assert "prototype" in status_doc.casefold()
    assert "smoke tests do not establish scientific validation" in status_doc.casefold()
