from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cleanup_inventory_tool_writes_expected_hotspots(tmp_path):
    csv_output = tmp_path / "cleanup.csv"
    md_output = tmp_path / "cleanup.md"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "qa" / "build_cleanup_inventory.py"),
            "--csv-output",
            str(csv_output),
            "--md-output",
            str(md_output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    assert csv_output.exists()
    assert md_output.exists()

    with csv_output.open(encoding="utf-8", newline="") as handle:
        rows = {row["path"]: row for row in csv.DictReader(handle)}

    assert rows["src/fluxforge_gui/app.py"]["suggested_action"] == "split"
    assert rows["src/fluxforge_gui/split_app.py"]["suggested_action"] == "delete"
    assert rows["src/fluxforge/cli/app.py"]["accuracy_risk"] == "medium"
    txt_rows = {path for path in rows if path.endswith(".txt")}
    assert txt_rows == {
        "src/fluxforge_gui/_sync_probe.txt",
        "src/fluxforge_gui/xyz.txt",
    }
    assert "Repo Cleanup Workstream" in md_output.read_text(encoding="utf-8")
