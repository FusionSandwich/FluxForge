from __future__ import annotations

import os
from dataclasses import replace

import pytest

from fluxforge.gui import QT_AVAILABLE, SelectionBus
from fluxforge.gui.backends import PYQTGRAPH_AVAILABLE, PyQtGraphSpectrumCanvas
from fluxforge.gui.backends import pyqtgraph_backend as backend_module
from fluxforge.gui.canvas_intents import CanvasIntentKind
from fluxforge.gui.qt_compat import QApplication, Qt
from fluxforge.gui.spectrum_canvas import (
    CanvasPeakOverlay,
    CanvasROIOverlay,
    SpectrumTrace,
)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


pytestmark = pytest.mark.skipif(
    not (QT_AVAILABLE and PYQTGRAPH_AVAILABLE),
    reason="Qt spectrum renderer dependencies are unavailable.",
)


if QT_AVAILABLE:
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QAction
    from PySide6.QtTest import QTest


def _qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _canvas() -> PyQtGraphSpectrumCanvas:
    app = _qapp()
    canvas = PyQtGraphSpectrumCanvas(selection_bus=SelectionBus())
    canvas.resize(1000, 640)
    canvas.show()
    canvas.set_interaction_context("spectrum-1")
    canvas.set_traces(
        (
            SpectrumTrace(
                label="Foreground",
                counts=tuple(float((index % 19) + 1) for index in range(101)),
                channels=tuple(float(index) for index in range(101)),
                x_axis_label="Energy (keV)",
            ),
        )
    )
    canvas.plot_item.setXRange(0.0, 100.0, padding=0.0)
    canvas.plot_item.setYRange(0.0, 25.0, padding=0.0)
    app.processEvents()
    return canvas


def _viewport_point(canvas: PyQtGraphSpectrumCanvas, x: float, y: float = 12.0):
    scene_position = canvas.plot_item.getViewBox().mapViewToScene(QPointF(x, y))
    return canvas.plot.mapFromScene(scene_position)


def _overlays():
    return (
        CanvasROIOverlay(
            roi_id="roi-1",
            spectrum_id="spectrum-1",
            signal_range=(40.0, 50.0),
            left_background_range=(30.0, 38.0),
            right_background_range=(52.0, 60.0),
            selected=True,
        ),
    ), (
        CanvasPeakOverlay(
            peak_id="peak-1",
            spectrum_id="spectrum-1",
            position=45.0,
            y_value=18.0,
            component_ids=("component-a", "component-b"),
            nuclide="Co60",
            tags=("reviewed",),
            pinned=False,
            selected=True,
        ),
    )


def test_shift_drag_previews_and_emits_one_complete_create_roi_intent() -> None:
    app = _qapp()
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)
    viewport = canvas.plot.viewport()

    start = _viewport_point(canvas, 25.0)
    middle = _viewport_point(canvas, 30.0)
    finish = _viewport_point(canvas, 35.0)
    QTest.mousePress(viewport, Qt.LeftButton, Qt.ShiftModifier, start)
    QTest.mouseMove(viewport, middle, delay=20)
    app.processEvents()
    assert canvas._shift_drag_preview is not None
    assert canvas._shift_drag_preview.isVisible()
    QTest.mouseMove(viewport, finish, delay=20)
    QTest.mouseRelease(viewport, Qt.LeftButton, Qt.ShiftModifier, finish)
    app.processEvents()

    assert canvas._shift_drag_preview is None
    assert len(intents) == 1
    intent = intents[0]
    assert intent.kind is CanvasIntentKind.CREATE_ROI
    assert intent.spectrum_id == "spectrum-1"
    # Integer viewport coordinates map back to slightly different values across
    # Windows and Linux Qt platform plugins.
    assert intent.bounds == pytest.approx((25.0, 35.0), abs=2.0)
    assert intent.left_background is not None
    assert intent.right_background is not None
    assert intent.left_background[1] == pytest.approx(intent.bounds[0])
    assert intent.right_background[0] == pytest.approx(intent.bounds[1])
    assert intent.drag_token.startswith("roi-create-")
    canvas.close()


def test_canonical_roi_regions_emit_one_finished_signal_or_background_move() -> None:
    canvas = _canvas()
    intents = []
    canvas.subscribe_intents(intents.append)
    rois, peaks = _overlays()
    canvas.set_analysis_overlays(
        rois,
        peaks,
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )

    bundle = canvas._roi_items_by_id["roi-1"]
    bundle.signal.setRegion((41.0, 51.0))
    assert len(intents) == 1
    assert intents[0].kind is CanvasIntentKind.MOVE_ROI
    assert intents[0].roi_id == "roi-1"
    assert intents[0].bounds == pytest.approx((41.0, 51.0))
    assert intents[0].left_background == pytest.approx((30.0, 38.0))
    assert intents[0].right_background == pytest.approx((52.0, 60.0))

    bundle.left_background.setRegion((29.0, 37.0))
    assert len(intents) == 2
    assert intents[1].kind is CanvasIntentKind.MOVE_BACKGROUND
    assert intents[1].roi_id == "roi-1"
    assert intents[1].left_background == pytest.approx((29.0, 37.0))
    assert intents[1].right_background == pytest.approx((52.0, 60.0))
    canvas.close()


def test_exact_peak_selection_and_finished_centroid_move_emit_exact_ids() -> None:
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)
    rois, peaks = _overlays()
    duplicate = CanvasPeakOverlay(
        peak_id="peak-duplicate",
        spectrum_id="spectrum-1",
        position=45.0,
        y_value=10.0,
    )
    canvas.set_analysis_overlays(
        rois,
        peaks + (duplicate,),
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )

    point = next(
        point
        for point in canvas._peak_scatter.points()
        if point.data() == "peak-duplicate"
    )
    canvas._peak_scatter_clicked(canvas._peak_scatter, [point])
    assert len(intents) == 1
    assert intents[0].kind is CanvasIntentKind.SELECT_PEAK
    assert intents[0].peak_id == "peak-duplicate"

    duplicate_roi = replace(rois[0], roi_id="roi-duplicate", selected=True)
    canvas.set_analysis_overlays(
        rois + (duplicate_roi,),
        peaks + (duplicate,),
        selected_roi_id="roi-duplicate",
        selected_peak_id="peak-duplicate",
    )
    intents.clear()
    canvas._prepare_context_menu(45.0)
    assert canvas._context_peak_id == "peak-duplicate"
    assert canvas._context_roi_id == "roi-duplicate"
    canvas.move_peak_action.trigger()
    canvas.delete_roi_action.trigger()
    assert intents[0].peak_id == "peak-duplicate"
    assert intents[1].roi_id == "roi-duplicate"

    canvas.set_analysis_overlays(
        rois,
        peaks,
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )
    intents.clear()
    canvas._selected_peak_line.setPos(47.25)
    canvas._selected_peak_line.sigPositionChangeFinished.emit(
        canvas._selected_peak_line
    )
    assert len(intents) == 1
    assert intents[0].kind is CanvasIntentKind.MOVE_PEAK
    assert intents[0].spectrum_id == "spectrum-1"
    assert intents[0].peak_id == "peak-1"
    assert intents[0].position == pytest.approx(47.25)
    assert intents[0].drag_token.startswith("peak-")
    canvas.close()


def test_real_mouse_drag_of_selected_centroid_emits_once_on_release() -> None:
    app = _qapp()
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)
    rois, peaks = _overlays()
    canvas.set_analysis_overlays(
        rois,
        peaks,
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )
    viewport = canvas.plot.viewport()
    start = _viewport_point(canvas, 45.0, 12.0)
    finish = _viewport_point(canvas, 48.0, 12.0)

    QTest.mousePress(viewport, Qt.LeftButton, pos=start)
    QTest.mouseMove(viewport, finish, delay=30)
    app.processEvents()
    assert intents == []
    QTest.mouseRelease(viewport, Qt.LeftButton, pos=finish)
    app.processEvents()

    assert len(intents) == 1
    assert intents[0].kind is CanvasIntentKind.MOVE_PEAK
    assert intents[0].peak_id == "peak-1"
    assert intents[0].position == pytest.approx(48.0, abs=1.2)
    canvas.close()


def test_real_mouse_drag_of_roi_signal_handle_emits_once_on_release() -> None:
    app = _qapp()
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)
    rois, peaks = _overlays()
    canvas.set_analysis_overlays(
        rois,
        peaks,
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )
    viewport = canvas.plot.viewport()
    start = _viewport_point(canvas, 40.0, 12.0)
    finish = _viewport_point(canvas, 42.0, 12.0)

    QTest.mousePress(viewport, Qt.LeftButton, pos=start)
    QTest.mouseMove(viewport, finish, delay=30)
    app.processEvents()
    assert intents == []
    QTest.mouseRelease(viewport, Qt.LeftButton, pos=finish)
    app.processEvents()

    assert len(intents) == 1
    assert intents[0].kind is CanvasIntentKind.MOVE_ROI
    assert intents[0].roi_id == "roi-1"
    assert intents[0].bounds == pytest.approx((42.0, 50.0), abs=1.2)
    canvas.close()


@pytest.mark.parametrize(
    ("part", "start_x", "finish_x", "expected"),
    (
        ("left_background", 30.0, 28.0, (28.0, 38.0)),
        ("right_background", 60.0, 62.0, (52.0, 62.0)),
    ),
)
def test_real_mouse_drag_of_each_background_handle_emits_once_on_release(
    part: str,
    start_x: float,
    finish_x: float,
    expected: tuple[float, float],
) -> None:
    app = _qapp()
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)
    rois, peaks = _overlays()
    canvas.set_analysis_overlays(
        rois,
        peaks,
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )
    viewport = canvas.plot.viewport()
    start = _viewport_point(canvas, start_x, 12.0)
    finish = _viewport_point(canvas, finish_x, 12.0)

    QTest.mousePress(viewport, Qt.LeftButton, pos=start)
    QTest.mouseMove(viewport, finish, delay=30)
    app.processEvents()
    assert intents == []
    QTest.mouseRelease(viewport, Qt.LeftButton, pos=finish)
    app.processEvents()

    assert len(intents) == 1
    assert intents[0].kind is CanvasIntentKind.MOVE_BACKGROUND
    assert intents[0].roi_id == "roi-1"
    moved = getattr(intents[0], part)
    if part == "left_background":
        assert moved[0] < start_x
        assert moved[1] == pytest.approx(expected[1], abs=0.25)
    else:
        assert moved[0] == pytest.approx(expected[0], abs=0.25)
        assert moved[1] > start_x
    canvas.close()


def test_context_menu_actions_are_persistent_and_emit_scientific_intents(
    monkeypatch,
) -> None:
    app = _qapp()
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)
    rois, peaks = _overlays()
    canvas.set_analysis_overlays(
        rois,
        peaks,
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )
    QTest.mouseClick(
        canvas.plot.viewport(),
        Qt.RightButton,
        pos=_viewport_point(canvas, 45.0),
    )
    app.processEvents()
    assert canvas.context_menu.isVisible()
    canvas.context_menu.hide()
    canvas._prepare_context_menu(45.0)
    responses = iter((("Cs-137", True), ("reviewed-again", True), ("benchmark", True)))
    monkeypatch.setattr(
        backend_module.QInputDialog,
        "getText",
        staticmethod(lambda *_args, **_kwargs: next(responses)),
    )

    object_names = {
        action.objectName()
        for action in canvas.context_menu.actions()
        if action.objectName()
    }
    assert {
        "CanvasContextAddPeakAction",
        "CanvasContextSelectPeakAction",
        "CanvasContextMovePeakAction",
        "CanvasContextDeletePeakAction",
        "CanvasContextAssignNuclideAction",
        "CanvasContextClearNuclideAction",
        "CanvasContextPinNuclideAction",
        "CanvasContextUnpinNuclideAction",
        "CanvasContextTagPeakAction",
        "CanvasContextTagNuclideAction",
        "CanvasContextAddComponentAction",
        "CanvasContextSplitPeakAction",
        "CanvasContextMergePeaksAction",
        "CanvasContextSelectRoiAction",
        "CanvasContextDeleteRoiAction",
        "CanvasContextCreateRoiAction",
        "CanvasContextResetViewAction",
        "CanvasContextCrosshairAction",
        "ExportSpectrumPlotAction",
    }.issubset(object_names)

    export_action = next(
        (
            action
            for action in canvas.context_menu.actions()
            if action.objectName() == "ExportSpectrumPlotAction"
        ),
        None,
    )
    assert export_action is not None
    export_action.trigger()
    app.processEvents()
    assert canvas.plot.scene().exportDialog is not None
    assert canvas.plot.scene().exportDialog.isVisible()
    canvas.plot.scene().exportDialog.close()

    for action in (
        canvas.add_peak_action,
        canvas.select_peak_action,
        canvas.move_peak_action,
        canvas.delete_peak_action,
        canvas.assign_nuclide_action,
        canvas.clear_nuclide_action,
    ):
        action.trigger()
    assert canvas.pin_nuclide_action.isEnabled()
    assert not canvas.unpin_nuclide_action.isEnabled()
    canvas.pin_nuclide_action.trigger()
    canvas.set_analysis_overlays(
        rois,
        (replace(peaks[0], pinned=True),),
        selected_roi_id="roi-1",
        selected_peak_id="peak-1",
    )
    canvas._prepare_context_menu(45.0)
    assert not canvas.pin_nuclide_action.isEnabled()
    assert canvas.unpin_nuclide_action.isEnabled()
    canvas.unpin_nuclide_action.trigger()
    for action in (
        canvas.tag_peak_action,
        canvas.tag_nuclide_action,
        canvas.add_component_action,
        canvas.split_peak_action,
        canvas.merge_peaks_action,
        canvas.select_roi_action,
        canvas.delete_roi_action,
        canvas.create_roi_action,
    ):
        action.trigger()
    for object_name in (
        "CanvasContextForegroundRoleAction",
        "CanvasContextBackgroundRoleAction",
        "CanvasContextSecondaryRoleAction",
    ):
        action = canvas.context_menu.findChild(QAction, object_name)
        assert action is not None
        action.trigger()
    emitted_kinds = [intent.kind for intent in intents]
    assert emitted_kinds == [
        CanvasIntentKind.ADD_PEAK,
        CanvasIntentKind.SELECT_PEAK,
        CanvasIntentKind.MOVE_PEAK,
        CanvasIntentKind.DELETE_PEAK,
        CanvasIntentKind.ASSIGN_NUCLIDE,
        CanvasIntentKind.CLEAR_NUCLIDE,
        CanvasIntentKind.PIN_NUCLIDE,
        CanvasIntentKind.UNPIN_NUCLIDE,
        CanvasIntentKind.TAG_PEAK,
        CanvasIntentKind.TAG_NUCLIDE,
        CanvasIntentKind.ADD_COMPONENT,
        CanvasIntentKind.SPLIT_PEAK,
        CanvasIntentKind.MERGE_PEAKS,
        CanvasIntentKind.SELECT_ROI,
        CanvasIntentKind.DELETE_ROI,
        CanvasIntentKind.CREATE_ROI,
        CanvasIntentKind.ASSIGN_SPECTRUM_ROLE,
        CanvasIntentKind.ASSIGN_SPECTRUM_ROLE,
        CanvasIntentKind.ASSIGN_SPECTRUM_ROLE,
    ]
    assert [intent.role for intent in intents[-3:]] == [
        "foreground",
        "background",
        "secondary",
    ]
    canvas.close()


def test_crosshair_button_emits_toggle_intent_updates_readout_and_zoom_api() -> None:
    app = _qapp()
    canvas = _canvas()
    intents = []
    canvas.set_intent_sink(intents.append)

    QTest.mouseClick(canvas.crosshair_button, Qt.LeftButton)
    app.processEvents()
    assert canvas._crosshair_enabled
    assert canvas.crosshair_readout.isVisible()
    assert intents[-1].kind is CanvasIntentKind.TOGGLE_CROSSHAIR
    assert intents[-1].enabled is True

    QTest.mouseMove(canvas.plot.viewport(), _viewport_point(canvas, 55.0, 10.0))
    app.processEvents()
    if "--" in canvas.crosshair_readout.text():
        # Linux's offscreen platform does not always synthesize hover delivery;
        # exercise the same scene-coordinate callback deterministically.
        canvas._move_crosshair(
            canvas.plot.mapToScene(_viewport_point(canvas, 55.0, 10.0))
        )
    assert "--" not in canvas.crosshair_readout.text()

    canvas.zoom_to_range(40.0, 50.0, padding_fraction=0.1)
    x_range = canvas.plot_item.viewRange()[0]
    assert x_range == pytest.approx((39.0, 51.0), abs=0.05)

    intents.clear()
    QTest.mouseClick(canvas.zoom_in_button, Qt.LeftButton)
    QTest.qWait(180)
    app.processEvents()
    assert [intent.kind for intent in intents] == [CanvasIntentKind.CHANGE_VIEWPORT]
    assert intents[0].viewport_id == "primary-spectrum"
    assert intents[0].x_range is not None
    assert intents[0].y_range is not None
    canvas.close()
