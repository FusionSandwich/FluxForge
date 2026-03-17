from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_gui_review_gallery_builder_creates_html_and_summary(tmp_path):
    current_dir = tmp_path / "current"
    baseline_root = tmp_path / "baselines"
    gallery_dir = tmp_path / "gallery"
    manifest_path = tmp_path / "manifest.json"

    payload = {"platform": "linux", "screenshots": [], "steps": [], "status": "ok"}
    (current_dir / "run.json").parent.mkdir(parents=True, exist_ok=True)
    (current_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    _write_png(current_dir / "01-launch.png", (10, 20, 30))
    _write_png(current_dir / "02-spectrum-loaded.png", (30, 40, 50))
    _write_png(baseline_root / "linux" / "01-launch.png", (10, 20, 30))
    _write_png(baseline_root / "linux" / "02-spectrum-loaded.png", (35, 45, 55))

    manifest_path.write_text(
        json.dumps(
            {
                "platforms": {
                    "linux": {
                        "window_geometry": "1400x900",
                        "checkpoints": [
                            {
                                "name": "01-launch.png",
                                "label": "Launch",
                                "scenario": "shell",
                                "review_note": "check launch",
                            },
                            {
                                "name": "02-spectrum-loaded.png",
                                "label": "Spectrum",
                                "scenario": "preview",
                                "review_note": "check preview",
                            },
                        ],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "qa" / "build_gui_review_gallery.py"),
            "--current-dir",
            str(current_dir),
            "--output-dir",
            str(gallery_dir),
            "--baseline-root",
            str(baseline_root),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    summary = json.loads((gallery_dir / "summary.json").read_text(encoding="utf-8"))
    statuses = {record["name"]: record["status"] for record in summary["records"]}
    assert statuses["01-launch.png"] == "match"
    assert statuses["02-spectrum-loaded.png"] == "changed"
    assert (gallery_dir / "index.html").exists()
