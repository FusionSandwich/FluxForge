"""Elapsed Start/Stop/Power editor for activation irradiation histories."""

import json
from pathlib import Path

from fluxforge.gui.irradiation_history import (
    irradiation_history_payload,
    prepare_irradiation_history,
    read_irradiation_history_payload,
)
from fluxforge.gui.qt_compat import QT_AVAILABLE

if QT_AVAILABLE:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )

    class IrradiationHistoryDialog(QDialog):
        historyChanged = Signal(object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Irradiation History")
            self.resize(720, 480)
            self.applied_segments = ()
            layout = QVBoxLayout(self)
            guidance = QLabel(
                "Enter chronological elapsed times in seconds from a common zero. "
                "Power uses the same reference for every row (1 = reference power). "
                "Gaps are retained as zero-power shutdowns; the final Stop is "
                "the end of irradiation.",
                self,
            )
            guidance.setWordWrap(True)
            layout.addWidget(guidance)
            self.table = QTableWidget(0, 3, self)
            self.table.setObjectName("IrradiationHistoryTable")
            self.table.setHorizontalHeaderLabels(
                ["Start (s)", "Stop (s)", "Relative power"]
            )
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.itemChanged.connect(self._mark_pending)
            layout.addWidget(self.table)
            controls = QHBoxLayout()
            for attribute, text, name, callback in (
                (
                    "add_button",
                    "Add interval",
                    "IrradiationAddButton",
                    self.add_interval,
                ),
                (
                    "remove_button",
                    "Remove selected",
                    "IrradiationRemoveButton",
                    self.remove_selected,
                ),
                (
                    "load_button",
                    "Load JSON…",
                    "IrradiationLoadButton",
                    self._choose_load,
                ),
                (
                    "export_button",
                    "Export JSON…",
                    "IrradiationExportButton",
                    self._choose_export,
                ),
                (
                    "apply_button",
                    "Apply history",
                    "IrradiationApplyButton",
                    self._apply_clicked,
                ),
            ):
                button = QPushButton(text, self)
                button.setObjectName(name)
                button.clicked.connect(callback)
                setattr(self, attribute, button)
                controls.addWidget(button)
            layout.addLayout(controls)
            self.status_label = QLabel("Add an interval, then apply the history.", self)
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)
            note = QLabel(
                "Entered timing does not replace a certified operating record "
                "or supply uncertainty.",
                self,
            )
            note.setWordWrap(True)
            layout.addWidget(note)

        def _mark_pending(self):
            self.status_label.setText(
                "Edits pending. Apply history to validate the current rows."
            )

        def rows(self):
            return [
                tuple(
                    (
                        self.table.item(row, col).text()
                        if self.table.item(row, col)
                        else ""
                    )
                    for col in range(3)
                )
                for row in range(self.table.rowCount())
            ]

        def set_rows(self, rows):
            intervals, _ = prepare_irradiation_history(rows)
            self.table.blockSignals(True)
            try:
                self.table.setRowCount(len(intervals))
                for row, interval in enumerate(intervals):
                    for col, value in enumerate(
                        (interval.start_s, interval.stop_s, interval.relative_power)
                    ):
                        self.table.setItem(row, col, QTableWidgetItem(repr(value)))
            finally:
                self.table.blockSignals(False)
            self._mark_pending()

        def add_interval(self):
            row = self.table.rowCount()
            self.table.insertRow(row)
            # Leave Stop and Power blank: missing measurements are never supplied.
            previous = self.table.item(row - 1, 1) if row else None
            for col, value in enumerate((previous.text() if previous else "0", "", "")):
                self.table.setItem(row, col, QTableWidgetItem(value))
            self.table.setCurrentCell(row, 1)
            self._mark_pending()

        def remove_selected(self):
            for row in sorted(
                {item.row() for item in self.table.selectedIndexes()}, reverse=True
            ):
                self.table.removeRow(row)
            self._mark_pending()

        def segments(self):
            return prepare_irradiation_history(self.rows())[1]

        def apply_history(self):
            segments = self.segments()
            self.applied_segments = segments
            self.historyChanged.emit(segments)
            self.status_label.setText(
                f"Applied {len(segments)} duration segments, including shutdowns. "
                f"Elapsed time: {sum(segment.duration_s for segment in segments):g} s."
            )
            return segments

        def load_file(self, path):
            intervals = read_irradiation_history_payload(
                json.loads(Path(path).read_text(encoding="utf-8"))
            )
            self.set_rows(
                (interval.start_s, interval.stop_s, interval.relative_power)
                for interval in intervals
            )

        def export_json(self, path):
            payload = irradiation_history_payload(self.rows())
            Path(path).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )

        def _apply_clicked(self):
            try:
                self.apply_history()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid history", str(exc))

        def _choose_load(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Load irradiation history", "", "JSON (*.json)"
            )
            if path:
                try:
                    self.load_file(path)
                except (ValueError, OSError) as exc:
                    QMessageBox.warning(self, "Cannot load history", str(exc))

        def _choose_export(self):
            path, _ = QFileDialog.getSaveFileName(
                self, "Export irradiation history", "", "JSON (*.json)"
            )
            if path:
                try:
                    self.export_json(path)
                except (ValueError, OSError) as exc:
                    QMessageBox.warning(self, "Cannot export history", str(exc))

else:

    class IrradiationHistoryDialog:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Native Qt is unavailable.")
