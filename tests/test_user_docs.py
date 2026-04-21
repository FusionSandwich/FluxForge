from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

USER_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "INSTALLATION.md",
    ROOT / "docs" / "CLI_REFERENCE.md",
    ROOT / "docs" / "USER_GUIDE.md",
    ROOT / "docs" / "EXAMPLE_WORKFLOWS.md",
    ROOT / "docs" / "tutorials" / "0_quick_start.md",
    ROOT / "docs" / "tutorials" / "1_getting_started.md",
    ROOT / "examples" / "README.md",
    ROOT / "examples" / "RAFM_irradiation" / "README.md",
    ROOT / "examples" / "manual_peak_inspection" / "README.md",
    ROOT / "examples" / "speckit_benchmark" / "README.md",
    ROOT / "examples" / "spectroscopy_data" / "README.md",
    ROOT / "examples" / "unfolding_benchmark" / "README.md",
    ROOT / "examples" / "validation" / "README.md",
]


def test_user_docs_use_installed_entrypoints():
    for path in USER_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "PYTHONPATH=src" not in text, path
        assert "python -m fluxforge.cli.app" not in text, path


def test_primary_docs_point_to_command_discovery():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
    cli_reference = (ROOT / "docs" / "CLI_REFERENCE.md").read_text(encoding="utf-8")

    assert "fluxforge commands" in readme
    assert "fluxforge-gui" in readme
    assert "fluxforge commands" in install
    assert "No module named 'fluxforge.gui'" in install
    assert "fluxforge-gui" in install
    assert "# FluxForge CLI Reference" in cli_reference


def _discover_example_entry_paths() -> set[str]:
    discovered: set[str] = set()
    for path in (ROOT / "examples").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("reference_"):
            continue
        discovered.add(path.relative_to(ROOT).as_posix())
    discovered.add("examples/manual_peak_inspection/README.md")
    discovered.add("examples/speckit_benchmark")
    return discovered


def test_example_inventory_covers_every_entrypoint():
    inventory = json.loads(
        (ROOT / "examples" / "example_inventory.json").read_text(encoding="utf-8")
    )
    entries = inventory["entries"]
    assert entries
    manifest_paths = {entry["entry_path"] for entry in entries}
    assert manifest_paths == _discover_example_entry_paths()
    for entry in entries:
        assert entry["id"]
        assert entry["title"]
        assert entry["tier"]
        assert entry["install_profile"] is not None
        assert entry["command"]


def test_example_inventory_dataset_paths_exist():
    inventory = json.loads(
        (ROOT / "examples" / "example_inventory.json").read_text(encoding="utf-8")
    )
    dataset_paths = inventory["datasets"]
    assert dataset_paths
    for dataset in dataset_paths:
        assert (ROOT / dataset["path"]).exists(), dataset["path"]
