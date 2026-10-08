"""Record whether the pre-repair GUI also fails with a newly released Qt runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

import PySide6

BASELINE = "57eac3d801e2ac00fcaa6e16e8cedef36e9929aa"
root = Path(__file__).resolve().parents[3]
subprocess.run(["git", "fetch", "--depth=1", "origin", BASELINE], cwd=root, check=True)
with tempfile.TemporaryDirectory(prefix="fluxforge-qt-baseline-") as directory:
    temporary = Path(directory).resolve()
    assert temporary.parent == Path(tempfile.gettempdir()).resolve()
    archive = temporary / "baseline.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", "--output", str(archive), BASELINE],
        cwd=root,
        check=True,
    )
    with ZipFile(archive) as contents:
        contents.extractall(temporary)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(temporary / "src")
    env["XDG_CONFIG_HOME"] = str(temporary / "settings")
    env["XDG_DATA_HOME"] = str(temporary / "data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_modern_gui_shell.py",
            "tests/test_project_tracker_assets.py",
        ],
        cwd=temporary,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    print(result.stdout, flush=True)
    print(
        json.dumps(
            {
                "diagnostic_only": True,
                "baseline": BASELINE,
                "pyside6": PySide6.__version__,
                "python": sys.version,
                "exit_code": result.returncode,
            }
        ),
        flush=True,
    )
