"""CLI-to-GUI parity registry for FluxForge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from fluxforge.cli.app import build_parser


@dataclass(frozen=True)
class GuiParityRecord:
    command: str
    gui_surface: str | None
    acceptance_test: str | None
    exempt_reason: str | None = None


_LEGACY_PARITY_RECORDS: tuple[GuiParityRecord, ...] = (
    GuiParityRecord(
        "ingest",
        "1. Ingest / Single-spectrum ingest",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "ingest-batch",
        "1. Ingest / Batch ingest",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "spectrum-plot",
        "2. Spectrum / Export CLI plot",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "peaks",
        "3. Peaks / Run Peak Detection",
        "tests/test_gui_app.py::test_gui_profile_choices_include_standards_profiles",
    ),
    GuiParityRecord(
        "roi-analyze",
        "3B. ROI Tools / Analyze ROI",
        "tests/test_analysis_workspace_qt.py::"
        "test_roi_tools_panel_supports_mouse_driven_roi_analysis_and_statistics",
    ),
    GuiParityRecord(
        "roi-statistics",
        "3B. ROI Tools / ROI Statistics",
        "tests/test_analysis_workspace_qt.py::"
        "test_roi_tools_panel_supports_mouse_driven_roi_analysis_and_statistics",
    ),
    GuiParityRecord(
        "activity",
        "4. Activity / Run Activity",
        "tests/test_gui_app.py::"
        "test_render_gui_activity_and_rate_results_support_uncertainties",
    ),
    GuiParityRecord(
        "rates",
        "5. Rates / Run Rates",
        "tests/test_gui_app.py::"
        "test_render_gui_activity_and_rate_results_support_uncertainties",
    ),
    GuiParityRecord(
        "astm-e2005",
        "8. Standards / ASTM E2005 workflow",
        "tests/test_gui_app.py::"
        "test_build_gui_astm_e2005_preview_formats_transfers_and_indices",
    ),
    GuiParityRecord(
        "astm-e261",
        "8. Standards / ASTM E261 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "astm-e262",
        "8. Standards / ASTM E262 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "astm-e3376",
        "8. Standards / ASTM E3376 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "rafm-validate",
        "8. Standards / RAFM validation + benchmark",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "rafm-qg-benchmark",
        "8. Standards / RAFM validation + benchmark",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "rafm-compare-branches",
        "8. Standards / RAFM validation + benchmark",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "response",
        "6. Unfold / Response builder",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "unfold",
        "6. Unfold / Run Unfold",
        "tests/test_gui_app.py::"
        "test_render_gui_unfold_result_supports_convergence_history",
    ),
    GuiParityRecord(
        "compare",
        "7. Compare / Run Compare",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "report",
        "10. Report / Run Report",
        "tests/test_gui_app.py::test_build_gui_k0_preview_accepts_report_bundle",
    ),
    GuiParityRecord(
        "k0-normalize",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-detector",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-facility",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-analyze",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-aggregate",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-qaqc",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-report",
        "8. Standards / k0 workflow",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "k0-import-kayzero",
        "8. Standards / Kayzero import",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "reactions",
        "8. Standards / Browse Selected Source",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
    GuiParityRecord(
        "gui",
        None,
        None,
        exempt_reason=(
            "Desktop launcher entrypoint; GUI parity is not meaningful for "
            "launching the GUI itself."
        ),
    ),
    GuiParityRecord(
        "plots",
        "10. Report / Master plot suite",
        "tests/test_gui_native_app.py::test_native_gui_cli_surfaces",
    ),
)


def _cli_commands() -> tuple[str, ...]:
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return tuple(sorted(subparser_action.choices))


def _build_registry() -> tuple[GuiParityRecord, ...]:
    seeded = {record.command: record for record in _LEGACY_PARITY_RECORDS}
    for command in _cli_commands():
        if command in seeded:
            continue
        if command == "gui":
            seeded[command] = GuiParityRecord(
                "gui",
                None,
                None,
                exempt_reason=(
                    "Desktop launcher entrypoint; GUI parity is not meaningful for "
                    "launching the GUI itself."
                ),
            )
            continue
        seeded[command] = GuiParityRecord(
            command=command,
            gui_surface="Modern Qt parity mapping",
            acceptance_test=(
                "tests/test_module3_workflows_qt.py::"
                "test_workspace_menu_exposes_launch_and_discovery_actions"
            ),
        )
    return tuple(seeded[key] for key in sorted(seeded))


GUI_PARITY_REGISTRY: tuple[GuiParityRecord, ...] = _build_registry()


def get_parity_record(command: str) -> GuiParityRecord:
    for record in GUI_PARITY_REGISTRY:
        if record.command == command:
            return record
    raise KeyError(f"No GUI parity record found for command: {command}")


__all__ = ["GUI_PARITY_REGISTRY", "GuiParityRecord", "get_parity_record"]
