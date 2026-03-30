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
    assert next_steps[0]["id"] == "2.2"


def test_stage0_gate_is_closed_live():
    steps = load_steps()["steps"]
    index = {step["id"]: step for step in steps}

    assert index["S0.1"]["sequence_status"] == "complete"
    assert index["S0.2"]["sequence_status"] == "complete"
    assert index["S0.3"]["sequence_status"] == "complete"
    assert index["S0.4"]["sequence_status"] == "complete"
    assert index["S0.5"]["sequence_status"] == "complete"


def test_phase1_steps_are_complete_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    assert steps["1.1"]["repo_status"] == "complete"
    assert steps["1.1"]["sequence_status"] == "complete"
    assert steps["1.2"]["repo_status"] == "complete"
    assert steps["1.2"]["sequence_status"] == "complete"


def test_gui_phase1_completion_is_recorded_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    for step_id in ("1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.16", "1.17", "1.18"):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"


def test_phase1_foundation_repo_completion_is_explicitly_tracked():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    for step_id in ("1.9", "1.10", "1.11", "1.12", "1.13", "1.14"):
        assert steps[step_id]["repo_status"] == "complete"
        assert steps[step_id]["sequence_status"] == "complete"

    assert steps["1.15"]["repo_status"] == "complete"
    assert steps["1.19"]["repo_status"] == "complete"


def test_phase2_calibration_workspace_is_complete_in_sequence():
    steps = {step["id"]: step for step in load_steps()["steps"]}

    assert steps["2.1"]["repo_status"] == "complete"
    assert steps["2.1"]["sequence_status"] == "complete"
    assert steps["2.2"]["sequence_status"] == "next"
