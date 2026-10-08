"""Tabulated activity, mass, half-life and saturated SigPhi conversion."""

import json
from pathlib import Path

from fluxforge.gui.qt_compat import QT_AVAILABLE
from fluxforge.gui.reaction_rate_view import ReactionRateInput, reaction_rate_payload

if QT_AVAILABLE:
    from PySide6.QtCore import Qt
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

    class ReactionRateDialog(QDialog):
        def __init__(self, *, history_provider, activity_provider=None, parent=None):
            super().__init__(parent)
            self.history_provider = history_provider
            self.activity_provider = activity_provider or (lambda: ())
            self.current_payload = None
            self.setWindowTitle("Activity to Reaction Rate")
            self.resize(1120, 660)
            layout = QVBoxLayout(self)
            guide = QLabel(
                "Apply a history in Tools → Irradiation History. Enter activity "
                "at the end of irradiation (EOI), measured monitor mass, element "
                "mass fraction and target isotope fraction. Fractions use 0–1. "
                "Found activities retain their original count reference and "
                "are shown for comparison; declare EOI activity separately.",
                self,
            )
            guide.setWordWrap(True)
            layout.addWidget(guide)
            self.table = QTableWidget(0, 9, self)
            self.table.setObjectName("ReactionRateInputTable")
            self.table.setHorizontalHeaderLabels(
                [
                    "Product",
                    "Target element",
                    "Mass (mg)",
                    "Element fraction",
                    "Isotope fraction",
                    "Half-life (s)",
                    "EOI activity (Bq)",
                    "σ activity (Bq)",
                    "Found activity (Bq)",
                ]
            )
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.resizeColumnsToContents()
            self.table.itemChanged.connect(self._mark_pending)
            layout.addWidget(self.table)
            controls = QHBoxLayout()
            for attr, title, name, handler in (
                ("add_button", "Add row", "ReactionRateAddButton", self.add_row),
                (
                    "remove_button",
                    "Remove selected",
                    "ReactionRateRemoveButton",
                    self.remove_selected,
                ),
                (
                    "found_button",
                    "Add found isotopes",
                    "ReactionRateFoundButton",
                    self.add_found,
                ),
                (
                    "convert_button",
                    "Convert",
                    "ReactionRateConvertButton",
                    self._convert_clicked,
                ),
                (
                    "export_button",
                    "Export JSON…",
                    "ReactionRateExportButton",
                    self._choose_export,
                ),
            ):
                button = QPushButton(title, self)
                button.setObjectName(name)
                button.clicked.connect(handler)
                setattr(self, attr, button)
                controls.addWidget(button)
            layout.addLayout(controls)
            self.result_table = QTableWidget(0, 7, self)
            self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.result_table.setHorizontalHeaderLabels(
                [
                    "Product",
                    "Reaction",
                    "Target atoms",
                    "Rate at relative power 1 (s⁻¹)",
                    "SigPhi at relative power 1 (atom⁻¹ s⁻¹)",
                    "σ SigPhi (activity only)",
                    "Uncertainty scope",
                ]
            )
            layout.addWidget(self.result_table)
            self.status_label = QLabel("Add measured constraints, then convert.", self)
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)
            note = QLabel(
                "Displayed sigma is conditional on fixed mass, composition, "
                "half-life and timing. Blank sigma remains unavailable. These "
                "rates use relative power 1 in the applied history; its absolute "
                "power remains unspecified. These rows do not qualify a complete "
                "reaction-rate uncertainty budget.",
                self,
            )
            note.setWordWrap(True)
            layout.addWidget(note)

        def _mark_pending(self):
            self.current_payload = None
            self.result_table.setRowCount(0)
            self.status_label.setText(
                "Inputs changed. Convert again to update the results."
            )

        def add_row(self, values=None, *, found=None, source_id=None):
            if values is False:  # clicked(bool)
                values = None
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = values or ("",) * 8
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(value)))
            item = QTableWidgetItem("" if found is None else f"{found:g}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, source_id)
            item.setToolTip(
                "Original count-reference activity; not automatically EOI activity."
            )
            self.table.setItem(row, 8, item)
            self._mark_pending()

        def remove_selected(self):
            for row in sorted(
                {item.row() for item in self.table.selectedIndexes()}, reverse=True
            ):
                self.table.removeRow(row)
            self._mark_pending()

        def add_found(self):
            known = {
                self.table.item(row, 8).data(Qt.UserRole)
                for row in range(self.table.rowCount())
            }
            for result in self.activity_provider():
                source_id = f"{result.nuclide}:{result.line_energy_keV!r}"
                if source_id in known:
                    continue
                self.add_row(
                    (result.nuclide, "", "", "", "", repr(result.half_life_s), "", ""),
                    found=result.activity_bq,
                    source_id=source_id,
                )
                known.add(source_id)

        def inputs(self):
            result = []
            for row in range(self.table.rowCount()):
                values = [
                    (
                        self.table.item(row, col).text()
                        if self.table.item(row, col)
                        else ""
                    )
                    for col in range(8)
                ]
                sigma = None if not values[7].strip() else values[7]
                result.append(ReactionRateInput(*values[:7], sigma))
            return result

        def convert(self):
            payload = reaction_rate_payload(self.inputs(), self.history_provider())
            self.result_table.setRowCount(len(payload["rows"]))
            for index, row in enumerate(payload["rows"]):
                values = [
                    row["product"],
                    row["reaction_id"],
                    row["target_atoms"],
                    row["saturation_rate_per_s"],
                    row["sigphi_per_atom_s"],
                    row["sigphi_activity_sigma"],
                    row["uncertainty_scope"],
                ]
                for col, value in enumerate(values):
                    text = (
                        "Unavailable"
                        if value is None
                        else (
                            f"{value:.8g}"
                            if isinstance(value, (int, float))
                            else str(value)
                        )
                    )
                    self.result_table.setItem(index, col, QTableWidgetItem(text))
                    self.result_table.item(index, col).setToolTip(str(value))
            self.result_table.resizeColumnsToContents()
            self.current_payload = payload
            self.status_label.setText(
                f"Converted {len(payload['rows'])} rows with explicit EOI activity "
                "and current applied history."
            )
            return payload

        def export_json(self, path):
            # Recompute: an externally changed history must never export stale rates.
            payload = self.convert()
            Path(path).write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )

        def _convert_clicked(self):
            try:
                self.convert()
            except (ValueError, TypeError) as exc:
                self._mark_pending()
                QMessageBox.warning(self, "Cannot convert activity", str(exc))

        def _choose_export(self):
            path, _ = QFileDialog.getSaveFileName(
                self, "Export reaction rates", "", "JSON (*.json)"
            )
            if path:
                try:
                    self.export_json(path)
                except (ValueError, TypeError, OSError) as exc:
                    QMessageBox.warning(self, "Cannot export reaction rates", str(exc))

else:

    class ReactionRateDialog:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Native Qt is unavailable.")
