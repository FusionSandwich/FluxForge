#!/usr/bin/env python3
"""Generate the folder-by-folder cleanup workstream inventory."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "docs" / "REPO_CLEANUP_WORKSTREAM.csv"
DEFAULT_MD = REPO_ROOT / "docs" / "REPO_CLEANUP_WORKSTREAM.md"
REVIEWABLE_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".cfg",
    ".yml",
    ".yaml",
    ".sh",
}
HOTSPOT_ACTIONS = {
    "src/fluxforge_gui/app.py": "split",
    "src/fluxforge_gui/ui_builder.py": "split",
    "src/fluxforge_gui/commands.py": "split",
    "src/fluxforge/cli/app.py": "split",
    "src/fluxforge_gui/split_app.py": "delete",
    "src/fluxforge_gui/_sync_probe.txt": "delete",
    "src/fluxforge_gui/xyz.txt": "delete",
}
HOTSPOT_NOTES = {
    "src/fluxforge_gui/app.py": (
        "mechanical split left a duplicated import header and a 2k+ line "
        "shell module"
    ),
    "src/fluxforge_gui/ui_builder.py": (
        "large mixed-responsibility widget builder with copied imports"
    ),
    "src/fluxforge_gui/commands.py": (
        "large command dispatch module with copied imports and handler density"
    ),
    "src/fluxforge/cli/app.py": (
        "3.5k-line CLI monolith that should be grouped by subcommand family"
    ),
    "tools/github_issues/create_issues.py": (
        "contains workstation-specific absolute paths"
    ),
    "tools/github_issues/generate_detailed_standards_issues.py": (
        "contains workstation-specific absolute paths"
    ),
    "tools/github_issues/generate_granular_gui_issues.py": (
        "contains workstation-specific absolute paths"
    ),
    "tools/github_issues/generate_gui_issues.py": (
        "contains workstation-specific absolute paths"
    ),
    "tools/github_issues/generate_standards_issues.py": (
        "contains workstation-specific absolute paths"
    ),
    "tools/github_issues/rename_issues.py": (
        "contains workstation-specific absolute paths"
    ),
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files: list[Path] = []
    for raw in result.stdout.splitlines():
        if not raw:
            continue
        path = Path(raw)
        if path.parts and path.parts[0] == "artifacts":
            continue
        if path.parts[:2] == ("tests", "data"):
            continue
        if path.suffix.lower() in REVIEWABLE_SUFFIXES:
            files.append(path)
        elif path.as_posix() in HOTSPOT_ACTIONS:
            files.append(path)
    return sorted(files)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return path.read_text(encoding="utf-8").count("\n") + 1
    except UnicodeDecodeError:
        return 0


def _top_folder(path: Path) -> str:
    if len(path.parts) < 2:
        return path.parts[0]
    return "/".join(path.parts[:2]) if path.parts[0] == "src" else path.parts[0]


def _accuracy_risk(path: Path) -> str:
    parts = set(path.parts)
    if {"analysis", "data", "physics", "solvers", "validation"} & parts:
        return "high"
    if {
        "fluxforge_gui",
        "cli",
        "core",
        "io",
        "plots",
        "reporting",
        "workflows",
    } & parts:
        return "medium"
    return "low"


def _suggested_action(path: Path, line_count: int) -> str:
    path_key = path.as_posix()
    if path_key in HOTSPOT_ACTIONS:
        return HOTSPOT_ACTIONS[path_key]
    if path.parts[:2] == ("tools", "github_issues"):
        return "refactor"
    if line_count >= 1500:
        return "split"
    if line_count >= 600:
        return "refactor"
    return "keep"


def _duplicate_findings(path: Path, line_count: int) -> str:
    path_key = path.as_posix()
    if path_key in HOTSPOT_NOTES:
        return HOTSPOT_NOTES[path_key]
    if line_count >= 1500:
        return "hotspot-size"
    if "tests" in path.parts and line_count >= 400:
        return "fixture/setup consolidation candidate"
    if path.parts[:2] == ("tools", "github_issues"):
        return "absolute-path cleanup"
    return "none flagged yet"


def _test_gate(path: Path) -> tuple[str, str]:
    parts = set(path.parts)
    if "fluxforge_gui" in parts:
        gates = (
            "tests/test_gui_app.py; tests/test_gui_native_app.py; "
            "tests/test_gui_desktop_native.py"
        )
        return gates, gates
    if "cli" in parts:
        return "tests/test_cli_app.py", "tests/test_cli_app.py"
    if {"analysis", "data", "physics", "solvers", "validation"} & parts:
        gate = "targeted domain tests + full regression subset"
        return gate, gate
    if parts & {"tools", "examples", "docs"}:
        return "lint + path/reference checks", "lint + path/reference checks"
    return "relevant targeted tests", "relevant targeted tests"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for rel_path in _tracked_files():
        if not (REPO_ROOT / rel_path).exists():
            continue
        line_count = _line_count(REPO_ROOT / rel_path)
        pre_tests, post_tests = _test_gate(rel_path)
        path_key = rel_path.as_posix()
        rows.append(
            {
                "folder_bucket": _top_folder(rel_path),
                "path": path_key,
                "line_count": str(line_count),
                "review_status": "baseline-captured",
                "suggested_action": _suggested_action(rel_path, line_count),
                "accuracy_risk": _accuracy_risk(rel_path),
                "duplicate_findings": _duplicate_findings(rel_path, line_count),
                "required_tests_before": pre_tests,
                "required_tests_after": post_tests,
            }
        )
        seen_paths.add(path_key)
    for path_key, action in HOTSPOT_ACTIONS.items():
        if action != "delete" or path_key in seen_paths:
            continue
        rel_path = Path(path_key)
        pre_tests, post_tests = _test_gate(rel_path)
        rows.append(
            {
                "folder_bucket": _top_folder(rel_path),
                "path": path_key,
                "line_count": "0",
                "review_status": "removed-from-runtime",
                "suggested_action": action,
                "accuracy_risk": _accuracy_risk(rel_path),
                "duplicate_findings": HOTSPOT_NOTES.get(
                    path_key, "removed as dev-only runtime clutter"
                ),
                "required_tests_before": pre_tests,
                "required_tests_after": post_tests,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    rows: list[dict[str, str]],
    output_path: Path,
    csv_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folder_counts = Counter(row["folder_bucket"] for row in rows)
    action_counts = Counter(row["suggested_action"] for row in rows)
    hotspot_rows = sorted(
        rows,
        key=lambda row: (row["suggested_action"] != "split", -int(row["line_count"])),
    )[:15]

    try:
        csv_label = csv_path.relative_to(REPO_ROOT)
    except ValueError:
        csv_label = csv_path

    lines = [
        "# Repo Cleanup Workstream",
        "",
        (
            "This document is generated by "
            "`tools/qa/build_cleanup_inventory.py` and tracks the "
            "folder-by-folder cleanup queue."
        ),
        "",
        f"- Total reviewed files: `{len(rows)}`",
        f"- CSV inventory: `{csv_label}`",
        "",
        "## Folder Buckets",
    ]
    for bucket, count in sorted(folder_counts.items()):
        lines.append(f"- `{bucket}`: `{count}` files")
    lines.extend(["", "## Suggested Actions"])
    for action, count in sorted(action_counts.items()):
        lines.append(f"- `{action}`: `{count}` files")
    lines.extend(
        [
            "",
            "## Highest-Priority Hotspots",
            "",
            "| Path | Lines | Action | Accuracy Risk | Notes |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for row in hotspot_rows:
        lines.append(
            (
                f"| `{row['path']}` | {row['line_count']} | "
                f"`{row['suggested_action']}` | "
                f"`{row['accuracy_risk']}` | {row['duplicate_findings']} |"
            )
        )
    lines.extend(
        [
            "",
            "## Review Policy",
            "",
            (
                "- `baseline-captured` means the file is in scope for the "
                "cleanup campaign and has an initial suggested action."
            ),
            (
                "- Accuracy-sensitive folders (`analysis`, `data`, "
                "`physics`, `solvers`, `validation`) require "
                "characterization tests before logic changes."
            ),
            (
                "- GUI and CLI hotspots should be split in staged passes "
                "rather than one large formatting diff."
            ),
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the FluxForge folder-by-folder cleanup inventory."
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV output path for the per-file inventory.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD,
        help="Markdown summary output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows()
    write_csv(rows, args.csv_output)
    write_markdown(rows, args.md_output, args.csv_output)
    print(f"Wrote cleanup inventory to {args.csv_output}")
    print(f"Wrote cleanup summary to {args.md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
