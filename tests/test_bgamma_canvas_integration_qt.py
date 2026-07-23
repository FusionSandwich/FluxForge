from __future__ import annotations

import os
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fluxforge.core.analysis_workspace import detect_peak_candidates  # noqa: E402
from fluxforge.core.workspace_document import AnalysisROI  # noqa: E402
from fluxforge.gui import QT_AVAILABLE, SelectionBus  # noqa: E402
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE  # noqa: E402
from fluxforge.gui.main_window import FluxForgeMainWindow  # noqa: E402
from fluxforge.gui.mode_manager import ModeManager  # noqa: E402
from fluxforge.gui.panels.modern_shell import build_demo_spectrum  # noqa: E402
from fluxforge.gui.qt_compat import QApplication  # noqa: E402
from fluxforge.io.session import read_ffs_session  # noqa: E402


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _window_with_duplicate_energy_peaks():
    app = _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.resize(1200, 800)
    window.show()
    app.processEvents()

    detected = detect_peak_candidates(build_demo_spectrum())
    assert detected
    source = detected[0]
    peaks = (
        replace(
            source,
            peak_id="duplicate-a",
            energy_keV=500.0,
            roi_bounds_keV=(480.0, 490.0),
        ),
        replace(
            source,
            peak_id="duplicate-b",
            energy_keV=500.0,
            roi_bounds_keV=(700.0, 720.0),
        ),
    )
    window.analysis_workspace.replace_peaks(peaks)
    spectrum_id = window.analysis_workspace.document.active_spectrum_id
    duplicate_b = window.analysis_workspace.document.peak_by_id(
        spectrum_id,
        "duplicate-b",
    )
    assert duplicate_b is not None
    canonical_roi = AnalysisROI(
        roi_id="canonical-duplicate-b-roi",
        spectrum_id=spectrum_id,
        signal_range=(730.0, 740.0),
        left_background_range=(720.0, 728.0),
        right_background_range=(742.0, 750.0),
        associated_peak_ids=("duplicate-b",),
        label="Canonical duplicate B",
    )
    window.analysis_workspace.upsert_roi(canonical_roi)
    window.analysis_workspace.upsert_peak_model(
        replace(duplicate_b, roi_id=canonical_roi.roi_id)
    )
    app.processEvents()
    return window, app


def test_peak_table_uses_exact_id_and_zooms_to_that_peaks_roi() -> None:
    window, app = _window_with_duplicate_energy_peaks()
    try:
        peak_panel = window.bottom_dock.widget().peak_table_panel
        peak_panel.table.setCurrentCell(1, 0)
        peak_panel.table.selectRow(1)
        peak_panel._publish_selected_peak()
        app.processEvents()

        assert window.analysis_workspace.selected_peak().peak_id == "duplicate-b"
        assert window.selection_bus.state.peak_id == "duplicate-b"
        assert window.central_tabs.canvas._selected_peak_id == "duplicate-b"
        assert window.central_tabs.canvas.plot_item.viewRange()[0] == pytest.approx(
            (728.8, 741.2), abs=0.1
        )
        assert peak_panel.table.item(1, 2).text() == "730.0-740.0"

        roi_bundle = window.central_tabs.canvas._roi_items_by_id[
            "canonical-duplicate-b-roi"
        ]
        roi_bundle.signal.setRegion((732.0, 742.0))
        app.processEvents()
        assert peak_panel.table.item(1, 2).text() == "732.0-742.0"
        window.undo_stack.undo()
        app.processEvents()
        assert peak_panel.table.item(1, 2).text() == "730.0-740.0"
    finally:
        window.close()


def test_canvas_exact_id_selection_updates_the_peak_table() -> None:
    window, app = _window_with_duplicate_energy_peaks()
    try:
        canvas = window.central_tabs.canvas
        point = next(
            item
            for item in canvas._peak_scatter.points()
            if item.data() == "duplicate-a"
        )
        canvas._peak_scatter_clicked(canvas._peak_scatter, [point])
        app.processEvents()

        peak_panel = window.bottom_dock.widget().peak_table_panel
        assert window.analysis_workspace.selected_peak().peak_id == "duplicate-a"
        assert window.selection_bus.state.peak_id == "duplicate-a"
        assert peak_panel.table.currentRow() == 0
    finally:
        window.close()


def test_context_edits_persist_and_undo_as_focused_operations(tmp_path) -> None:
    app = _qapp()
    window = FluxForgeMainWindow(
        mode_manager=ModeManager(),
        selection_bus=SelectionBus(),
        load_example=True,
    )
    window.show()
    app.processEvents()
    try:
        original_peak_ids = {
            peak.peak_id for peak in window.analysis_workspace.document.peaks
        }
        original_roi_ids = {
            roi.roi_id for roi in window.analysis_workspace.document.rois
        }
        canvas = window.central_tabs.canvas

        canvas._prepare_context_menu(500.0)
        canvas.add_peak_action.trigger()
        app.processEvents()
        added_peak = next(
            peak
            for peak in window.analysis_workspace.document.peaks
            if peak.peak_id not in original_peak_ids
        )
        assert "re-run required" in window.right_dock.widget().analysis_summary.text()

        canvas._prepare_context_menu(added_peak.centroid_energy_keV)
        canvas.add_component_action.trigger()
        app.processEvents()
        assert (
            len(
                window.analysis_workspace.document.peak_by_id(
                    added_peak.spectrum_id, added_peak.peak_id
                ).components
            )
            == 1
        )

        canvas._prepare_context_menu(540.0)
        canvas.create_roi_action.trigger()
        app.processEvents()
        added_roi = next(
            roi
            for roi in window.analysis_workspace.document.rois
            if roi.roi_id not in original_roi_ids
        )
        assert window.undo_stack.count() == 3

        target = tmp_path / "direct-manipulation.ffs"
        assert window.save_session(target)
        persisted = read_ffs_session(target).document
        assert persisted.roi_by_id(added_roi.roi_id) == added_roi
        assert (
            len(
                persisted.peak_by_id(
                    added_peak.spectrum_id, added_peak.peak_id
                ).components
            )
            == 1
        )

        window.undo_stack.undo()
        window.undo_stack.undo()
        window.undo_stack.undo()
        app.processEvents()
        assert {
            peak.peak_id for peak in window.analysis_workspace.document.peaks
        } == original_peak_ids
        assert {
            roi.roi_id for roi in window.analysis_workspace.document.rois
        } == original_roi_ids
        assert window.selection_bus.state.peak_id is None
        assert window.selection_bus.state.roi_id is None
    finally:
        window.close()


def test_invalid_background_drag_reverts_overlay_and_shows_clear_error() -> None:
    window, app = _window_with_duplicate_energy_peaks()
    try:
        canvas = window.central_tabs.canvas
        roi = window.analysis_workspace.document.roi_by_id("canonical-duplicate-b-roi")
        assert roi is not None
        window.central_tabs._sync_analysis_overlays()
        bundle = canvas._roi_items_by_id[roi.roi_id]
        original = tuple(bundle.left_background.getRegion())
        undo_count = window.undo_stack.count()

        bundle.left_background.setRegion((731.0, 735.0))
        app.processEvents()

        persisted = window.analysis_workspace.document.roi_by_id(roi.roi_id)
        restored = canvas._roi_items_by_id[roi.roi_id]
        assert persisted.left_background_range == pytest.approx(original)
        assert tuple(restored.left_background.getRegion()) == pytest.approx(original)
        assert window.undo_stack.count() == undo_count
        assert canvas.status_label.text().startswith("Edit rejected:")
    finally:
        window.close()


def test_view_actions_and_crosshair_are_canonical_undoable_and_persisted(
    tmp_path,
) -> None:
    window, app = _window_with_duplicate_energy_peaks()
    try:
        assert window.undo_stack.count() == 0
        window._log_scale_action.trigger()
        app.processEvents()
        viewport = window.analysis_workspace.document.viewport_by_id("primary-spectrum")
        assert viewport.log_y
        assert window.central_tabs.canvas._log_scale
        assert window._log_scale_action.isChecked()

        window.undo_stack.undo()
        app.processEvents()
        assert (
            window.analysis_workspace.document.viewport_by_id("primary-spectrum")
            is None
        )
        assert not window.central_tabs.canvas._log_scale
        assert not window._log_scale_action.isChecked()
        window.undo_stack.redo()
        app.processEvents()
        assert window.central_tabs.canvas._log_scale
        assert window._log_scale_action.isChecked()

        window._peak_labels_action.trigger()
        app.processEvents()
        assert not window.central_tabs.canvas._peak_labels_visible
        window.undo_stack.undo()
        app.processEvents()
        assert window.central_tabs.canvas._peak_labels_visible
        assert window._peak_labels_action.isChecked()

        window.central_tabs.canvas.crosshair_button.click()
        app.processEvents()
        assert window.central_tabs.canvas._crosshair_enabled
        window.undo_stack.undo()
        app.processEvents()
        assert not window.central_tabs.canvas._crosshair_enabled
        window.undo_stack.redo()
        app.processEvents()
        assert window.central_tabs.canvas._crosshair_enabled

        target = tmp_path / "viewport.ffs"
        assert window.save_session(target)
        persisted = read_ffs_session(target).document.viewport_by_id("primary-spectrum")
        assert persisted.log_y
        assert persisted.crosshair_enabled
    finally:
        window.close()


def test_switching_spectrum_clears_exact_selection_and_retargets_viewport() -> None:
    window, app = _window_with_duplicate_energy_peaks()
    try:
        peak_panel = window.bottom_dock.widget().peak_table_panel
        peak_panel.table.selectRow(1)
        peak_panel._publish_selected_peak()
        assert window.selection_bus.state.peak_id == "duplicate-b"

        window.analysis_workspace.select_spectrum("background")
        app.processEvents()
        assert window.selection_bus.state.peak_id is None
        assert window.selection_bus.state.roi_id is None
        assert window.central_tabs.canvas._selected_peak_id is None

        window.central_tabs.canvas.crosshair_button.click()
        app.processEvents()
        viewport = window.analysis_workspace.document.viewport_by_id("primary-spectrum")
        assert (
            viewport.spectrum_id
            == window.analysis_workspace.document.active_spectrum_id
        )
    finally:
        window.close()
