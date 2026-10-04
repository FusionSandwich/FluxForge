"""Capture the supported Qt workspace without changing its analysis or viewport."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import base64
import json


def capture_report_snapshot(window) -> dict[str, object]:
    """Copy current inputs, table text, canonical state and visible plot pixels.

    This is a presentation snapshot, not a scientific admission or a session file.
    Capture runs synchronously on the GUI thread before any export takes place.
    """
    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QTableWidget,
        QWidget,
        QDialog,
    )
    import pyqtgraph as pg

    report_dialog = getattr(window, "_report_dialog", None)
    if report_dialog is None:
        report_dialog = window.findChild(QDialog, "ReportExportDialog")
    bottom = window.bottom_dock.widget()
    excluded = [
        getattr(bottom, name, None)
        for name in (
            "inventory_timeline_panel",
            "masking_review_panel",
            "optimization_workspace_panel",
            "second_irradiation_panel",
            "phase5_parity_panel",
        )
    ]
    if report_dialog is not None:
        excluded.append(report_dialog)

    def owned(widget):
        return not any(
            root is not None and (root is widget or root.isAncestorOf(widget))
            for root in excluded
        )

    inputs = []
    tables = []
    views = []
    for widget in window.findChildren(QWidget):
        name = widget.objectName()
        if not name or not owned(widget):
            continue
        value = None
        if isinstance(widget, QComboBox):
            value = widget.currentText()
        elif isinstance(widget, QCheckBox):
            value = widget.isChecked()
        elif isinstance(widget, (QDoubleSpinBox, QSpinBox, QSlider)):
            value = widget.value()
        elif isinstance(widget, QLineEdit) and not widget.isReadOnly():
            # Exclude the implementation's spin-box editors; their typed value
            # is already captured by the named spin box above.
            if name != "qt_spinbox_lineedit":
                value = widget.text()
        elif isinstance(widget, QPlainTextEdit) and not widget.isReadOnly():
            value = widget.toPlainText()
        if value is not None:
            inputs.append({"control": name, "value": value})
        if isinstance(widget, QTableWidget):

            def cell_text(row, column):
                editor = widget.cellWidget(row, column)
                if isinstance(editor, QComboBox):
                    return editor.currentText()
                if isinstance(editor, QLineEdit):
                    return editor.text()
                item = widget.item(row, column)
                return item.text() if item is not None else ""

            tables.append(
                {
                    "control": name,
                    "headers": [
                        (
                            widget.horizontalHeaderItem(column).text()
                            if widget.horizontalHeaderItem(column) is not None
                            else str(column + 1)
                        )
                        for column in range(widget.columnCount())
                    ],
                    "rows": [
                        [
                            cell_text(row, column)
                            for column in range(widget.columnCount())
                        ]
                        for row in range(widget.rowCount())
                    ],
                }
            )
        if isinstance(widget, pg.PlotWidget) and widget.isVisibleTo(window):
            image = widget.grab()
            buffer = QBuffer()
            buffer.open(QIODevice.WriteOnly)
            if image.isNull() or not image.save(buffer, "PNG"):
                raise RuntimeError(f"Could not capture the current plot {name}.")
            pixels = bytes(buffer.data())
            plot = widget.getPlotItem()
            views.append(
                {
                    "control": name,
                    "range": plot.viewRange(),
                    "log_x": bool(plot.ctrl.logXCheck.isChecked()),
                    "log_y": bool(plot.ctrl.logYCheck.isChecked()),
                    "png_base64": base64.b64encode(pixels).decode("ascii"),
                    "sha256": sha256(pixels).hexdigest(),
                    "asset": f"views/{len(views):03d}.png",
                }
            )
    state = window.analysis_workspace.state
    canvas = getattr(window.central_tabs, "canvas", None)
    parameters = {
        "mode": window.mode_manager.state.mode.value,
        "standard": window.mode_manager.state.standard,
        "peak_search_method": state.peak_search_method,
        "roi_background_method": state.roi_background_method,
        "background_mode": state.background_mode,
        "background_scale": state.background_scale,
        "selected_libraries": asdict(window.library_manager.state),
        "resolved_libraries": asdict(
            window.library_manager.resolved_state(
                standard=(
                    window.mode_manager.state.standard
                    if window.mode_manager.state.mode.value == "standards"
                    else None
                )
            )
        ),
        "irradiation_segments": [
            asdict(segment) for segment in window._irradiation_segments
        ],
        "canvas": canvas.viewport_state().to_dict() if canvas is not None else None,
    }
    snapshot = {
        "schema": "fluxforge.gui_report_snapshot.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scientific_admission": False,
        "parameters": parameters,
        "inputs": inputs,
        "tables": tables,
        "views": views,
        "workspace": window.analysis_workspace.document.to_dict(),
    }
    # Detach all nested mutable arrays/maps and reject non-JSON numeric values.
    return json.loads(json.dumps(snapshot, allow_nan=False))
