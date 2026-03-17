from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _desktop_dependencies_available() -> bool:
    if sys.platform == "win32":
        return (
            importlib.util.find_spec("pywinauto") is not None
            and importlib.util.find_spec("PIL") is not None
        )
    return importlib.util.find_spec("pyautogui") is not None


def _run_native_desktop_driver(output_dir: Path) -> dict[str, object]:
    if not _desktop_dependencies_available():
        pytest.skip("Native desktop GUI test dependencies are not installed.")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["FLUXFORGE_OFFLINE"] = "1"

    command = [
        sys.executable,
        str(REPO_ROOT / "tests" / "gui_desktop_driver.py"),
        str(output_dir),
    ]
    if sys.platform != "win32" and not env.get("DISPLAY"):
        xvfb_run = shutil.which("xvfb-run")
        if xvfb_run is None:
            pytest.skip("No DISPLAY available and xvfb-run is not installed.")
        command = [xvfb_run, "-a", *command]

    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CI diagnostics
        message = [
            "Native desktop GUI driver failed.",
            f"command: {exc.cmd}",
            f"returncode: {exc.returncode}",
            "stdout:",
            exc.stdout or "<empty>",
            "stderr:",
            exc.stderr or "<empty>",
        ]
        pytest.fail("\n".join(message))
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_native_desktop_gui_workflow_generates_evidence(tmp_path):
    result = _run_native_desktop_driver(tmp_path / "native_desktop_run")

    assert result["status"] == "ok"
    assert result["offline_mode"] is True
    assert len(result["screenshots"]) >= 9
    for screenshot in result["screenshots"]:
        assert Path(screenshot).exists()

    artifact_paths = [Path(item) for item in result["artifacts"]]
    for artifact in artifact_paths:
        assert artifact.exists()
    assert any(path.name == "gui_preview_native.png" for path in artifact_paths)
    assert any(path.name == "cli_spectrum_plot.png" for path in artifact_paths)
    assert any(path.name == "peaks.json" for path in artifact_paths)
    assert any(path.name == "activities.json" for path in artifact_paths)
    assert any(path.name == "rates.json" for path in artifact_paths)
    assert any(path.name == "response.json" for path in artifact_paths)
    assert any(path.name == "unfold.json" for path in artifact_paths)
    assert any(path.name == "validation.json" for path in artifact_paths)
    assert any(path.name == "k0_report.json" for path in artifact_paths)
    assert any(path.name == "plot_suite" and path.is_dir() for path in artifact_paths)

    spectrum_cli = result["cli_commands"]["spectrum_plot"]
    plot_suite_cli = result["cli_commands"]["plot_suite"]
    assert "fluxforge spectrum-plot" in spectrum_cli
    assert "--input" in spectrum_cli
    assert "fluxforge plots" in plot_suite_cli
    assert "--example" in plot_suite_cli

    steps = {entry["step"]: entry for entry in result["steps"]}
    assert "launch" in steps
    assert steps["ingest_raw"]["input"].endswith("Ti-RAFM-1a_25cm.ASC")
    assert steps["load_preview"]["spectrum_label"]
    assert steps["peak_selection"]["peak_count"] >= 1
    assert steps["peak_selection"]["selected_peak_energy_keV"] is not None
    assert "ROI channels" in steps["peak_selection"]["count_summary"]
    assert steps["roi_edit"]["roi_count"] == 1
    assert steps["calibration"]["point_count"] == 2
    assert steps["run_activity"]["summary"]
    assert steps["run_rates"]["summary"]
    assert steps["standards_preset"]["data_source"] == "k0_naa_monitors"
    assert "Method: GLS" in steps["unfold_preview"]["unfold_summary"]
    assert "MAE" in steps["compare_summary"]["summary"]
    assert steps["run_plot_suite"]["plot_count"] >= 1
