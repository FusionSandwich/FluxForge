"""Build an offline wheelhouse for FluxForge and selected extras."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _project_target(extras: list[str]) -> str:
    if not extras:
        return str(REPO_ROOT)
    return f"{REPO_ROOT}[{','.join(extras)}]"


def build_wheelhouse(output_dir: Path, extras: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        _project_target(extras),
        "--wheel-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline wheelhouse for FluxForge."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist" / "wheelhouse",
        help="Directory where wheels should be written.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        choices=["dev", "ml", "bundle"],
        help="Optional dependency extra to include. May be passed more than once.",
    )
    args = parser.parse_args(argv)
    build_wheelhouse(args.output_dir, list(args.extra))
    print(f"Wrote offline wheelhouse to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
