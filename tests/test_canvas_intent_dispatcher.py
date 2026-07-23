from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QUndoStack

from fluxforge.core.analysis_workspace import ActivityCalculationResult
from fluxforge.core.workspace_document import (
    FitDiagnostics,
    SpectrumRoleAssignment,
    WorkspaceDocument,
    WorkspaceSpectrum,
)
from fluxforge.gui.analysis_workspace import AnalysisWorkspaceController
from fluxforge.gui.canvas_controller import CanvasIntentDispatcher
from fluxforge.gui.canvas_intents import CanvasIntent, CanvasIntentKind
from fluxforge.gui.selection_bus import SelectionBus
from fluxforge.io.spe import GammaSpectrum


def _document() -> WorkspaceDocument:
    spectrum = GammaSpectrum(
        counts=np.asarray([2.0, 8.0, 5.0, 1.0]),
        channels=np.arange(4, dtype=float),
        energies=np.asarray([100.0, 101.0, 102.0, 103.0]),
        live_time=10.0,
        real_time=11.0,
        spectrum_id="spec-1",
    )
    return WorkspaceDocument(
        spectra=(
            WorkspaceSpectrum(
                spectrum_id="spec-1",
                spectrum=spectrum,
                label="Direct manipulation",
            ),
        ),
        spectrum_roles=(
            SpectrumRoleAssignment(role="foreground", spectrum_ids=("spec-1",)),
        ),
        active_spectrum_id="spec-1",
    )


def _dispatcher():
    controller = AnalysisWorkspaceController(_document())
    selection = SelectionBus()
    stack = QUndoStack()
    counts: dict[str, int] = {}

    def next_id(prefix: str) -> str:
        counts[prefix] = counts.get(prefix, 0) + 1
        return f"{prefix}-{counts[prefix]}"

    return (
        CanvasIntentDispatcher(
            controller,
            selection,
            undo_stack=stack,
            id_factory=next_id,
        ),
        controller,
        selection,
        stack,
    )


def _create_roi(dispatcher: CanvasIntentDispatcher) -> None:
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.CREATE_ROI,
            spectrum_id="spec-1",
            bounds=(100.8, 102.2),
            left_background=(99.8, 100.6),
            right_background=(102.4, 103.2),
        )
    )


def _add_peak(dispatcher: CanvasIntentDispatcher) -> None:
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.ADD_PEAK,
            spectrum_id="spec-1",
            position=101.1,
        )
    )


def _activity_result() -> ActivityCalculationResult:
    return ActivityCalculationResult(
        nuclide="Co60",
        line_energy_keV=1173.2,
        activity_bq=5.0,
        uncertainty_bq=0.5,
        age_corrected_activity_bq=6.0,
        mda_bq=0.1,
        half_life_s=1.0e8,
        source_age_s=100.0,
        chain_summary="test fixture",
    )


def _diagnostic(
    controller: AnalysisWorkspaceController,
    diagnostic_id: str,
) -> FitDiagnostics:
    return next(
        item
        for item in controller.document.fit_diagnostics
        if item.diagnostic_id == diagnostic_id
    )


def test_shift_drag_roi_intent_creates_one_undoable_canonical_edit() -> None:
    dispatcher, controller, selection, stack = _dispatcher()

    _create_roi(dispatcher)

    assert stack.count() == 1
    roi = controller.document.roi_by_id("roi-1")
    assert roi is not None
    assert roi.signal_range == pytest.approx((100.8, 102.2))
    assert roi.left_background_range == pytest.approx((99.8, 100.6))
    assert roi.right_background_range == pytest.approx((102.4, 103.2))
    assert (
        controller.document.viewport_by_id("primary-spectrum").selected_roi_id
        == "roi-1"
    )
    assert selection.state.roi_id == "roi-1"

    stack.undo()
    assert controller.document.rois == ()
    assert controller.document.viewport_by_id("primary-spectrum") is None
    stack.redo()
    assert controller.document.roi_by_id("roi-1") is not None


def test_roi_and_background_moves_commit_once_and_restore_exactly() -> None:
    dispatcher, controller, _selection, stack = _dispatcher()
    _create_roi(dispatcher)
    original = controller.document.roi_by_id("roi-1")
    stack.clear()

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.MOVE_ROI,
            spectrum_id="spec-1",
            roi_id="roi-1",
            bounds=(101.0, 102.5),
            drag_token="signal-drag",
        )
    )
    moved = controller.document.roi_by_id("roi-1")
    assert stack.count() == 1
    assert moved.signal_range == pytest.approx((101.0, 102.5))
    stack.undo()
    assert controller.document.roi_by_id("roi-1") == original
    stack.redo()

    before_background = controller.document.roi_by_id("roi-1")
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.MOVE_BACKGROUND,
            spectrum_id="spec-1",
            roi_id="roi-1",
            left_background=(99.5, 100.5),
            right_background=(102.8, 103.8),
            drag_token="background-drag",
        )
    )
    assert controller.document.roi_by_id(
        "roi-1"
    ).left_background_range == pytest.approx((99.5, 100.5))
    stack.undo()
    assert controller.document.roi_by_id("roi-1") == before_background


def test_peak_add_select_move_delete_and_undo_use_exact_ids() -> None:
    dispatcher, controller, selection, stack = _dispatcher()
    controller.set_activity_results((_activity_result(),))
    original_workflow = controller.document.workflow_state
    _add_peak(dispatcher)
    peak = controller.document.peak_by_id("spec-1", "peak-1")
    assert peak is not None
    assert selection.state.peak_id == "peak-1"
    assert (
        controller.document.workflow_state["analysis_invalidation"][
            "requires_reanalysis"
        ]
        is True
    )
    assert controller.state.activity_results == ()
    stack.undo()
    assert controller.document.workflow_state == original_workflow
    assert len(controller.state.activity_results) == 1
    stack.redo()

    peak = controller.document.peak_by_id("spec-1", "peak-1")
    fitted_peak = replace(
        peak,
        status="accepted",
        fit_quality=0.99,
        normalized_residuals=(0.25,),
        residual_channels=(1.0,),
    )
    unrelated_peak = replace(
        peak,
        peak_id="peak-unrelated",
        centroid_channel=2.0,
        centroid_energy_keV=102.0,
        fit_quality=0.88,
        normalized_residuals=(0.5,),
        residual_channels=(2.0,),
    )
    controller.upsert_peak_model(fitted_peak)
    controller.upsert_peak_model(unrelated_peak)
    fitted_diagnostic = FitDiagnostics(
        diagnostic_id="diag-peak-1",
        spectrum_id="spec-1",
        peak_id="peak-1",
        fit_revision=1,
        status="valid",
        x=(1.0,),
        observed=(8.0,),
        model=(7.8,),
        uncertainty=(2.0,),
        normalized_residuals=(0.1,),
        goodness_of_fit={"chi_squared": 0.01},
        method="fixture",
    )
    unrelated_diagnostic = replace(
        fitted_diagnostic,
        diagnostic_id="diag-unrelated",
        peak_id="peak-unrelated",
        x=(2.0,),
    )
    controller.upsert_fit_diagnostic(fitted_diagnostic)
    controller.upsert_fit_diagnostic(unrelated_diagnostic)
    peak = fitted_peak

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.MOVE_PEAK,
            spectrum_id="spec-1",
            peak_id="peak-1",
            position=102.4,
            drag_token="peak-drag",
        )
    )
    assert controller.document.peak_by_id(
        "spec-1", "peak-1"
    ).centroid_energy_keV == pytest.approx(102.4)
    moved_peak = controller.document.peak_by_id("spec-1", "peak-1")
    assert moved_peak.fit_quality == 0.0
    assert moved_peak.normalized_residuals == ()
    assert _diagnostic(controller, "diag-peak-1").status == "invalid"
    assert _diagnostic(controller, "diag-unrelated").status == "valid"
    assert controller.document.peak_by_id(
        "spec-1", "peak-unrelated"
    ).fit_quality == pytest.approx(0.88)
    stack.undo()
    assert controller.document.peak_by_id("spec-1", "peak-1") == peak
    assert _diagnostic(controller, "diag-peak-1") == fitted_diagnostic
    assert _diagnostic(controller, "diag-unrelated") == unrelated_diagnostic
    stack.redo()

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.DELETE_PEAK,
            spectrum_id="spec-1",
            peak_id="peak-1",
        )
    )
    assert controller.document.peak_by_id("spec-1", "peak-1") is None
    stack.undo()
    assert controller.document.peak_by_id("spec-1", "peak-1") is not None


def test_roi_association_and_shared_fit_invalidation_are_reciprocal_and_undoable() -> (
    None
):
    dispatcher, controller, _selection, stack = _dispatcher()
    _add_peak(dispatcher)
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.ADD_PEAK,
            spectrum_id="spec-1",
            position=101.8,
        )
    )
    for peak_id, residual in (("peak-1", 0.2), ("peak-2", -0.4)):
        peak = controller.document.peak_by_id("spec-1", peak_id)
        controller.upsert_peak_model(
            replace(
                peak,
                status="accepted",
                fit_quality=0.95,
                normalized_residuals=(residual,),
                residual_channels=(peak.centroid_channel,),
            )
        )
        controller.upsert_fit_diagnostic(
            FitDiagnostics(
                diagnostic_id=f"diag-{peak_id}",
                spectrum_id="spec-1",
                peak_id=peak_id,
                fit_revision=1,
                status="valid",
                x=(peak.centroid_channel,),
                observed=(8.0,),
                model=(7.5,),
                uncertainty=(2.0,),
                normalized_residuals=(residual,),
                method="fixture",
            )
        )

    _create_roi(dispatcher)
    roi = controller.document.roi_by_id("roi-1")
    assert roi.associated_peak_ids == ("peak-1", "peak-2")
    assert {
        controller.document.peak_by_id("spec-1", peak_id).roi_id
        for peak_id in roi.associated_peak_ids
    } == {"roi-1"}
    assert all(
        controller.document.peak_by_id("spec-1", peak_id).normalized_residuals == ()
        for peak_id in roi.associated_peak_ids
    )
    assert all(
        _diagnostic(controller, f"diag-{peak_id}").status == "invalid"
        for peak_id in roi.associated_peak_ids
    )

    stack.undo()
    assert controller.document.roi_by_id("roi-1") is None
    assert all(
        controller.document.peak_by_id("spec-1", peak_id).roi_id is None
        for peak_id in ("peak-1", "peak-2")
    )
    assert all(
        _diagnostic(controller, f"diag-{peak_id}").status == "valid"
        for peak_id in ("peak-1", "peak-2")
    )
    stack.redo()

    for peak_id, residual in (("peak-1", 0.2), ("peak-2", -0.4)):
        peak = controller.document.peak_by_id("spec-1", peak_id)
        controller.upsert_peak_model(
            replace(
                peak,
                status="accepted",
                fit_quality=0.97,
                normalized_residuals=(residual,),
                residual_channels=(peak.centroid_channel,),
            )
        )
        diagnostic = _diagnostic(controller, f"diag-{peak_id}")
        controller.upsert_fit_diagnostic(
            replace(
                diagnostic,
                roi_id="roi-1",
                status="valid",
                x=(peak.centroid_channel,),
                observed=(8.0,),
                model=(7.5,),
                uncertainty=(2.0,),
                normalized_residuals=(residual,),
                goodness_of_fit={"chi_squared": 0.02},
                warning_flags=(),
            )
        )

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.MOVE_ROI,
            spectrum_id="spec-1",
            roi_id="roi-1",
            bounds=(100.9, 102.4),
            drag_token="shared-roi",
        )
    )
    assert all(
        controller.document.peak_by_id("spec-1", peak_id).status == "invalidated"
        for peak_id in ("peak-1", "peak-2")
    )
    assert all(
        _diagnostic(controller, f"diag-{peak_id}").status == "invalid"
        for peak_id in ("peak-1", "peak-2")
    )
    stack.undo()
    assert all(
        controller.document.peak_by_id("spec-1", peak_id).status == "accepted"
        for peak_id in ("peak-1", "peak-2")
    )


def test_completed_reanalysis_clears_the_stale_result_marker() -> None:
    dispatcher, controller, _selection, _stack = _dispatcher()
    _add_peak(dispatcher)
    assert "analysis_invalidation" in controller.document.workflow_state

    controller.set_activity_results((_activity_result(),))

    assert "analysis_invalidation" not in controller.document.workflow_state
    assert len(controller.state.activity_results) == 1


def test_peak_context_edits_preserve_assignments_tags_components_and_pins() -> None:
    dispatcher, controller, _selection, stack = _dispatcher()
    _add_peak(dispatcher)

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.ASSIGN_NUCLIDE,
            spectrum_id="spec-1",
            peak_id="peak-1",
            nuclide="Cs-137",
        )
    )
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.TAG_PEAK,
            spectrum_id="spec-1",
            peak_id="peak-1",
            tag="reviewed",
        )
    )
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.TAG_NUCLIDE,
            nuclide="Cs-137",
            tag="benchmark",
        )
    )
    assert controller.document.nuclide_tags["Cs-137"] == ("benchmark",)
    stack.undo()
    assert "Cs-137" not in controller.document.nuclide_tags
    stack.redo()
    assert controller.document.nuclide_tags["Cs-137"] == ("benchmark",)

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.ADD_COMPONENT,
            spectrum_id="spec-1",
            peak_id="peak-1",
            position=101.0,
        )
    )
    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.SPLIT_PEAK,
            spectrum_id="spec-1",
            peak_id="peak-1",
            component_ids=("component-1",),
        )
    )
    peak = controller.document.peak_by_id("spec-1", "peak-1")
    assert peak.assignments[0].nuclide == "Cs-137"
    assert peak.assignments[0].provenance["source"] == "canvas_context_menu"
    assert peak.tags == ("reviewed",)
    assert controller.document.nuclide_tags["Cs-137"] == ("benchmark",)
    assert len(peak.components) == 2

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.MERGE_PEAKS,
            spectrum_id="spec-1",
            peak_id="peak-1",
            component_ids=tuple(item.component_id for item in peak.components),
        )
    )
    assert len(controller.document.peak_by_id("spec-1", "peak-1").components) == 1

    dispatcher.handle(CanvasIntent(CanvasIntentKind.PIN_NUCLIDE, nuclide="Co60"))
    assert controller.document.pinned_nuclides == ("Co60",)
    pinned_cascade = controller.state.cascade_sum_lines_keV
    assert pinned_cascade
    dispatcher.handle(CanvasIntent(CanvasIntentKind.UNPIN_NUCLIDE, nuclide="Co60"))
    assert controller.document.pinned_nuclides == ()
    assert controller.state.cascade_sum_lines_keV == ()
    stack.undo()
    assert controller.document.pinned_nuclides == ("Co60",)
    assert controller.state.cascade_sum_lines_keV == pinned_cascade


def test_role_and_viewport_intents_are_focused_and_undoable() -> None:
    dispatcher, controller, _selection, stack = _dispatcher()

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.ASSIGN_SPECTRUM_ROLE,
            spectrum_id="spec-1",
            role="foreground",
        )
    )
    assert stack.count() == 0
    assert "analysis_invalidation" not in controller.document.workflow_state

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.ASSIGN_SPECTRUM_ROLE,
            spectrum_id="spec-1",
            role="background",
        )
    )
    assert next(
        item for item in controller.document.spectrum_roles if item.role == "background"
    ).spectrum_ids == ("spec-1",)
    stack.undo()
    assert all(item.role != "background" for item in controller.document.spectrum_roles)

    dispatcher.handle(
        CanvasIntent(
            CanvasIntentKind.TOGGLE_CROSSHAIR,
            spectrum_id="spec-1",
            viewport_id="primary-spectrum",
            enabled=True,
        )
    )
    assert controller.document.viewport_by_id("primary-spectrum").crosshair_enabled
    stack.undo()
    assert controller.document.viewport_by_id("primary-spectrum") is None
