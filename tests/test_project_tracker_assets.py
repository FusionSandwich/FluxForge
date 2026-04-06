import json
from pathlib import Path

from fluxforge.gui import GUIMode, MainWindowScaffold, ModeManager, SelectionBus, SelectionState
from fluxforge.hal import AcquisitionState, MockMCADevice


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_stage0_tracker_counts_and_integrity():
    milestones = load_json(".github/project-management/milestones.json")
    labels = load_json(".github/project-management/labels.json")
    issues = load_json(".github/project-management/issues.json")

    milestone_titles = {item["title"] for item in milestones}
    label_names = {item["name"] for item in labels}
    epic_ids = {item["id"] for item in issues["epics"]}

    # The tracker is additive: later roadmap modules can append milestones and
    # planning items, but the Stage 0 baseline must remain present.
    assert len(milestones) >= 7
    assert len(issues["epics"]) >= 10
    assert len(issues["issues"]) >= 25
    assert "M3B — Offline Spectroscopy Parity" in milestone_titles
    assert "area/adr" in label_names
    assert "type/adr" in label_names
    assert "priority/p0-blocking" in label_names

    for item in issues["epics"]:
        assert item["milestone"] in milestone_titles

    for item in issues["issues"]:
        assert item["milestone"] in milestone_titles
        assert item["area"] in label_names
        assert item["type"] in label_names
        assert item["priority"] in label_names
        if "epic" in item:
            assert item["epic"] in epic_ids


def test_stage0_templates_and_adrs_exist():
    expected_templates = {
        "adr.yml",
        "bug.yml",
        "epic.yml",
        "feature.yml",
        "gui_ux.yml",
        "performance.yml",
        "standards_validation.yml",
    }
    template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    adr_dir = ROOT / "docs" / "adr"

    assert expected_templates.issubset({path.name for path in template_dir.iterdir()})

    adr_files = sorted(path for path in adr_dir.iterdir() if path.name.startswith("ADR-"))
    assert len(adr_files) == 7
    for path in adr_files:
        assert "**Status:** Accepted" in path.read_text(encoding="utf-8")


def test_gui_scaffold_has_six_dock_zones():
    scaffold = MainWindowScaffold()

    assert [zone.code for zone in scaffold.zones] == ["A", "B", "C", "D", "E", "F"]
    assert scaffold.zone_map()["C"].title == "Central Canvas"


def test_mode_manager_requires_standard_for_standards_mode():
    manager = ModeManager()

    try:
        manager.set_mode(GUIMode.STANDARDS)
    except ValueError as exc:
        assert "requires an active standard" in str(exc)
    else:  # pragma: no cover - defensive fallback
        raise AssertionError("standards mode should require a standard")

    state = manager.set_standard("ASTM E261")
    assert state.mode is GUIMode.STANDARDS
    assert state.standard == "ASTM E261"


def test_selection_bus_notifies_subscribers():
    bus = SelectionBus()
    observed = []
    bus.subscribe(observed.append)

    state = SelectionState(peak_energy_keV=661.7, nuclide="Cs-137")
    bus.publish(state)

    assert observed == [state]
    assert bus.state == state


def test_mock_mca_device_transitions():
    device = MockMCADevice()

    assert device.status().state is AcquisitionState.DISCONNECTED
    assert device.connect().state is AcquisitionState.IDLE
    assert device.start_acquisition().state is AcquisitionState.ACQUIRING
    assert device.stop_acquisition().state is AcquisitionState.IDLE
    assert len(device.read_counts()) == 4096
