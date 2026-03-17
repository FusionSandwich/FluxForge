from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_gui_probe(*args: str, env_overrides: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if env_overrides:
        env.update(env_overrides)

    command = [sys.executable, str(REPO_ROOT / "tests" / "gui_probe.py"), *args]
    if sys.platform != "win32" and not env.get("DISPLAY"):
        xvfb_run = shutil.which("xvfb-run")
        if xvfb_run is None:
            pytest.skip("No DISPLAY available and xvfb-run is not installed.")
        command = [xvfb_run, "-a", *command]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_native_gui_cli_surfaces():
    snapshot = _run_gui_probe("snapshot")

    assert snapshot["title"] == "FluxForge GUI (Spectrum + Workflow MVP)"
    assert len(snapshot["tabs"]) == 10
    for button_text in (
        "Run Batch Ingest",
        "Run CLI Plot Export",
        "Build Response",
        "Import Kayzero",
        "Run ASTM E3376",
        "Run RAFM Validation",
        "Run Plot Suite",
    ):
        assert button_text in snapshot["buttons"]


@pytest.mark.parametrize(
    ("button_text", "expected_subcommand"),
    (
        ("Run Batch Ingest", "ingest-batch"),
        ("Run CLI Plot Export", "spectrum-plot"),
        ("Build Response", "response"),
        ("Import Kayzero", "k0-import-kayzero"),
        ("Run Plot Suite", "plots"),
    ),
)
def test_new_gui_buttons_copy_last_cli(button_text: str, expected_subcommand: str):
    result = _run_gui_probe("button-cli", button_text)

    assert result["copy_state"] == "normal"
    assert result["clipboard"] == result["command_line"]
    assert result["cli_tokens"][0] == expected_subcommand


def test_native_gui_offline_mode_banner_state():
    snapshot = _run_gui_probe("snapshot", env_overrides={"FLUXFORGE_OFFLINE": "1"})

    assert snapshot["offline_mode"] is True
