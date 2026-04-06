"""Build native FluxForge CLI/GUI bundles with PyInstaller."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
SRC_DIR = REPO_ROOT / "src"


def _add_data_arg(source: Path, destination: str) -> str:
    separator = ";" if os.name == "nt" else ":"
    return f"{source}{separator}{destination}"


def _build_target(
    *,
    name: str,
    launcher: Path,
    dist_dir: Path,
    work_dir: Path,
    spec_dir: Path,
    windowed: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--paths",
        str(SRC_DIR),
        "--add-data",
        _add_data_arg(SRC_DIR / "fluxforge" / "data", "fluxforge/data"),
        str(launcher),
    ]
    if windowed:
        command.append("--windowed")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build native FluxForge CLI/GUI bundles with PyInstaller."
    )
    parser.add_argument(
        "--target",
        choices=["cli", "gui", "both"],
        default="both",
        help="Bundle target to build.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist" / "native",
        help="Directory where PyInstaller bundles should be written.",
    )
    args = parser.parse_args(argv)

    dist_dir = args.output_dir
    work_dir = REPO_ROOT / "build" / "pyinstaller"
    spec_dir = REPO_ROOT / "build" / "pyinstaller_specs"
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.target in {"cli", "both"}:
        targets.append(("FluxForgeCLI", TOOLS_DIR / "pyinstaller_launch_cli.py", False))
    if args.target in {"gui", "both"}:
        targets.append(("FluxForgeGUI", TOOLS_DIR / "pyinstaller_launch_gui.py", True))

    for name, launcher, windowed in targets:
        _build_target(
            name=name,
            launcher=launcher,
            dist_dir=dist_dir,
            work_dir=work_dir,
            spec_dir=spec_dir,
            windowed=windowed,
        )

    print(f"Wrote native bundle(s) to {dist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
