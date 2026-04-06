import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_steps():
    path = ROOT / ".github" / "project-management" / "implementation_steps.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_ordered_tracker_has_single_next_step():
    payload = load_steps()
    steps = payload["steps"]
    next_steps = [step for step in steps if step["sequence_status"] == "next"]

    assert len(next_steps) == 1
    assert next_steps[0]["id"] == "3.17"


def test_stage0_gate_is_closed_live():
    steps = load_steps()["steps"]
    index = {step["id"]: step for step in steps}

    assert index["S0.1"]["sequence_status"] == "complete"
    assert index["S0.2"]["sequence_status"] == "complete"
    assert index["S0.3"]["sequence_status"] == "complete"
    assert index["S0.4"]["sequence_status"] == "complete"
    assert index["S0.5"]["sequence_status"] == "complete"


def test_foundation_steps_are_complete_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    assert steps["1.1"]["repo_status"] == "complete"
    assert steps["1.1"]["sequence_status"] == "complete"
    assert steps["1.2"]["repo_status"] == "complete"
    assert steps["1.2"]["sequence_status"] == "complete"


def test_gui_foundation_completion_is_recorded_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    for step_id in ("1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.16", "1.17", "1.18"):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"


def test_foundation_repo_completion_is_explicitly_tracked():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    for step_id in ("1.9", "1.10", "1.11", "1.12", "1.13", "1.14"):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"

    assert steps["1.15"]["repo_status"] == "complete"
    assert steps["1.19"]["repo_status"] == "complete"


def test_analysis_workspace_steps_are_complete_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    assert steps["2.1"]["repo_status"] == "complete"
    assert steps["2.1"]["sequence_status"] == "complete"
    for step_id in (
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "2.10",
        "2.11",
        "2.12",
        "2.13",
        "2.14",
        "2.15",
        "2.16",
        "2.17",
        "2.18",
        "2.19",
        "2.20",
    ):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"

    for step_id in ("2.21", "2.22", "2.23", "2.24"):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"


def test_advanced_analysis_module_completion_is_recorded_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    for step_id in (
        "3.1",
        "3.2",
        "3.3",
        "3.4",
        "3.5",
        "3.6",
        "3.7",
        "3.8",
        "3.9",
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
        "3.15",
    ):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"


def test_offline_parity_block_is_explicit_before_hal_work():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    assert steps["3.16"]["repo_status"] == "complete"
    assert steps["3.16"]["sequence_status"] == "complete"
    assert steps["3.17"]["repo_status"] == "not-started"
    assert steps["3.17"]["sequence_status"] == "next"
    for step_id in ("3.18", "3.19", "3.20", "3.21", "3.22", "3.23", "3.24", "3.25", "3.26", "3.27"):
        assert steps[step_id]["sequence_status"] == "pending"

    assert steps["4.1"]["repo_status"] == "not-started"
    assert steps["4.1"]["sequence_status"] == "pending"
    for step_id in ("4.2", "4.3", "4.4"):
        assert steps[step_id]["sequence_status"] == "pending"


def test_predictive_detour_is_recorded_as_complete():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    for step_id in ("4P.1", "4P.2", "4P.3", "4P.4", "4P.5", "4P.6", "4P.7"):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"
