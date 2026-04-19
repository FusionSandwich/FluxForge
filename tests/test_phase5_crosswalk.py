from __future__ import annotations

from pathlib import Path

from fluxforge.validation.phase5_crosswalk import (
    load_phase5_crosswalk,
    render_phase5_crosswalk_markdown,
    summarize_phase5_crosswalk,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CROSSWALK_PATH = REPO_ROOT / ".github" / "project-management" / "phase5_crosswalk.json"


def test_phase5_crosswalk_loads_all_writeup_sections() -> None:
    payload = load_phase5_crosswalk(CROSSWALK_PATH)
    entries = payload["entries"]
    assert len(entries) == 27

    sections = sorted(int(item["writeup_section"]) for item in entries)
    assert sections == list(range(1, 28))


def test_phase5_crosswalk_summary_counts_replay_states_and_surfaces() -> None:
    payload = load_phase5_crosswalk(CROSSWALK_PATH)
    summary = summarize_phase5_crosswalk(payload, workspace_root=REPO_ROOT)

    assert summary["total_entries"] == 27
    by_replay = summary["by_replay_state"]
    assert by_replay["replay-now"] > 0
    assert by_replay["adapter-required"] > 0
    assert by_replay["reference-only"] > 0

    surface = summary["surface_coverage"]
    assert surface["backend_entries"] == 27
    assert surface["cli_entries"] == 27
    assert surface["gui_entries"] == 27


def test_phase5_crosswalk_markdown_render_contains_matrix_rows() -> None:
    payload = load_phase5_crosswalk(CROSSWALK_PATH)
    summary = summarize_phase5_crosswalk(payload)
    markdown = render_phase5_crosswalk_markdown(payload, summary)

    assert "# Phase 5 Crosswalk Report" in markdown
    assert "| Section | Source family | Replay state | Backend | CLI | GUI |" in markdown
    assert "| 1 | actigamma | replay-now | yes | yes | yes |" in markdown
    assert "| 27 | INAA-INRIM 3.1 | adapter-required | yes | yes | yes |" in markdown
