"""Utilities for the Phase 5 testing-catalog crosswalk tracker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PHASE5_CROSSWALK_SCHEMA = "fluxforge.phase5.crosswalk.v1"
REPLAY_STATES = {"replay-now", "adapter-required", "reference-only"}

_REQUIRED_ENTRY_FIELDS = {
    "writeup_section",
    "source_family",
    "replay_state",
    "data_class",
    "source_paths",
    "backend_paths",
    "cli_paths",
    "gui_paths",
    "fixture_paths",
    "test_paths",
    "probe_paths",
    "notes",
}

_PATH_LIST_FIELDS = (
    "source_paths",
    "backend_paths",
    "cli_paths",
    "gui_paths",
    "fixture_paths",
    "test_paths",
    "probe_paths",
)


def load_phase5_crosswalk(path: Path) -> dict[str, Any]:
    """Load and validate the machine-readable Phase 5 crosswalk payload."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 5 crosswalk must be a JSON object.")

    schema = str(payload.get("schema") or "").strip()
    if schema != PHASE5_CROSSWALK_SCHEMA:
        raise ValueError(
            f"Unexpected crosswalk schema {schema!r}; expected {PHASE5_CROSSWALK_SCHEMA!r}."
        )

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Crosswalk payload must include a non-empty 'entries' list.")

    seen_sections: set[int] = set()
    seen_families: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry #{index} must be an object.")
        missing = sorted(_REQUIRED_ENTRY_FIELDS - set(entry.keys()))
        if missing:
            raise ValueError(f"Entry #{index} is missing required fields: {', '.join(missing)}")

        section = int(entry["writeup_section"])
        if section <= 0:
            raise ValueError(f"Entry #{index} has invalid writeup_section={section}.")
        if section in seen_sections:
            raise ValueError(f"Duplicate writeup_section found: {section}")
        seen_sections.add(section)

        family = str(entry["source_family"] or "").strip()
        if not family:
            raise ValueError(f"Entry #{index} has empty source_family.")
        family_key = family.lower()
        if family_key in seen_families:
            raise ValueError(f"Duplicate source_family found: {family}")
        seen_families.add(family_key)

        replay_state = str(entry["replay_state"] or "").strip()
        if replay_state not in REPLAY_STATES:
            raise ValueError(
                f"Entry #{index} has invalid replay_state={replay_state!r}; "
                f"expected one of {sorted(REPLAY_STATES)}"
            )

        for field in _PATH_LIST_FIELDS:
            value = entry.get(field)
            if not isinstance(value, list) or not value:
                raise ValueError(f"Entry #{index} field {field!r} must be a non-empty list.")
            if any(not str(item).strip() for item in value):
                raise ValueError(f"Entry #{index} field {field!r} contains an empty path.")

        notes = str(entry.get("notes") or "").strip()
        if not notes:
            raise ValueError(f"Entry #{index} must include non-empty notes.")

    return payload


def summarize_phase5_crosswalk(
    payload: dict[str, Any], *, workspace_root: Path | None = None
) -> dict[str, Any]:
    """Summarize coverage and surface readiness for a validated crosswalk payload."""

    entries = list(payload.get("entries") or [])
    by_replay_state = {state: 0 for state in sorted(REPLAY_STATES)}
    by_data_class: dict[str, int] = {}

    backend_complete = 0
    cli_complete = 0
    gui_complete = 0
    fixture_targets = 0
    fixture_existing = 0
    test_targets = 0
    test_existing = 0
    probe_targets = 0
    probe_existing = 0

    for entry in entries:
        replay_state = str(entry.get("replay_state") or "")
        by_replay_state[replay_state] = by_replay_state.get(replay_state, 0) + 1

        data_class = str(entry.get("data_class") or "")
        by_data_class[data_class] = by_data_class.get(data_class, 0) + 1

        if entry.get("backend_paths"):
            backend_complete += 1
        if entry.get("cli_paths"):
            cli_complete += 1
        if entry.get("gui_paths"):
            gui_complete += 1

        fixture_paths = [str(item) for item in entry.get("fixture_paths") or []]
        test_paths = [str(item) for item in entry.get("test_paths") or []]
        probe_paths = [str(item) for item in entry.get("probe_paths") or []]

        fixture_targets += len(fixture_paths)
        test_targets += len(test_paths)
        probe_targets += len(probe_paths)

        if workspace_root is not None:
            fixture_existing += sum(1 for path in fixture_paths if (workspace_root / path).exists())
            test_existing += sum(1 for path in test_paths if (workspace_root / path).exists())
            probe_existing += sum(1 for path in probe_paths if (workspace_root / path).exists())

    return {
        "schema": "fluxforge.phase5.crosswalk.summary.v1",
        "crosswalk_schema": payload.get("schema"),
        "source_document": payload.get("source_document"),
        "total_entries": len(entries),
        "entry_sections": sorted(int(entry["writeup_section"]) for entry in entries),
        "by_replay_state": by_replay_state,
        "by_data_class": dict(sorted(by_data_class.items())),
        "surface_coverage": {
            "backend_entries": backend_complete,
            "cli_entries": cli_complete,
            "gui_entries": gui_complete,
        },
        "path_targets": {
            "fixture": {
                "targets": fixture_targets,
                "existing": fixture_existing,
            },
            "tests": {
                "targets": test_targets,
                "existing": test_existing,
            },
            "probes": {
                "targets": probe_targets,
                "existing": probe_existing,
            },
        },
    }


def render_phase5_crosswalk_markdown(
    payload: dict[str, Any], summary: dict[str, Any]
) -> str:
    """Render a concise markdown report for human review and release evidence."""

    entries = sorted(
        (dict(item) for item in payload.get("entries") or []),
        key=lambda item: int(item.get("writeup_section") or 0),
    )

    lines = [
        "# Phase 5 Crosswalk Report",
        "",
        f"- Source document: {payload.get('source_document')}",
        f"- Total entries: {summary.get('total_entries')}",
        "",
        "## Replay-State Summary",
    ]
    for state, count in (summary.get("by_replay_state") or {}).items():
        lines.append(f"- {state}: {count}")

    lines.extend(
        [
            "",
            "## Data-Class Summary",
        ]
    )
    for label, count in (summary.get("by_data_class") or {}).items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Entry Matrix",
            "",
            "| Section | Source family | Replay state | Backend | CLI | GUI |",
            "|---|---|---|---|---|---|",
        ]
    )
    for entry in entries:
        backend = "yes" if entry.get("backend_paths") else "no"
        cli = "yes" if entry.get("cli_paths") else "no"
        gui = "yes" if entry.get("gui_paths") else "no"
        lines.append(
            "| "
            f"{entry.get('writeup_section')} | "
            f"{entry.get('source_family')} | "
            f"{entry.get('replay_state')} | "
            f"{backend} | {cli} | {gui} |"
        )

    return "\n".join(lines) + "\n"


__all__ = [
    "PHASE5_CROSSWALK_SCHEMA",
    "REPLAY_STATES",
    "load_phase5_crosswalk",
    "summarize_phase5_crosswalk",
    "render_phase5_crosswalk_markdown",
]
