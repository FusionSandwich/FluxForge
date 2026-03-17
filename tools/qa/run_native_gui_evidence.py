#!/usr/bin/env python3
"""Run native GUI evidence capture and build a review gallery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_QA = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_QA))

from build_gui_review_gallery import build_gallery  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture native FluxForge GUI evidence and build a review gallery."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "tests" / "data" / "gui_review_baselines",
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit gallery output directory. "
            "Defaults to <output-dir>/review_gallery."
        ),
    )
    return parser.parse_args()


def build_command(output_dir: Path, env: dict[str, str]) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tests" / "gui_desktop_driver.py"),
        str(output_dir),
    ]
    if sys.platform != "win32" and not env.get("DISPLAY"):
        xvfb_run = shutil.which("xvfb-run")
        if xvfb_run:
            command = [xvfb_run, "-a", *command]
    return command


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    gallery_dir = (args.gallery_dir or (output_dir / "review_gallery")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["FLUXFORGE_OFFLINE"] = "1"

    command = build_command(output_dir, env)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    run_json = output_dir / "run.json"
    run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = build_gallery(
        current_dir=output_dir,
        baseline_root=args.baseline_root.resolve(),
        output_dir=gallery_dir,
    )
    print(
        json.dumps(
            {
                "run_json": str(run_json),
                "gallery_dir": str(gallery_dir),
                "summary": summary["platform"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
